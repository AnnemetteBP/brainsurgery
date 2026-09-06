"""T3: mixed-precision, buffer-dropping, sharded export of GPT-2 (124M).

Route: plain script on safetensors + torch for the tensor edit, and
huggingface_hub's official `split_torch_state_dict_into_shards` for the
sharded layout + `model.safetensors.index.json` (the same helper
`transformers.save_pretrained` uses), so the layout is the canonical HF one.

Run from the sandbox root:  .venv/bin/python out/T3/solution.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

SANDBOX = Path(__file__).resolve().parents[2]
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST = SANDBOX / "out" / "T3"

N_LAYERS = 12
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67,108,864

# Exactly the projection matrices to cast -- enumerated, not regex-matched,
# so ln_*/wte/wpe/*.bias can never be swept in.
PROJECTIONS = [
    f"h.{i}.{suffix}"
    for i in range(N_LAYERS)
    for suffix in (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
]
BUFFERS = [f"h.{i}.attn.bias" for i in range(N_LAYERS)]

EXPECTED_SHAPES = {
    "attn.c_attn.weight": (768, 2304),
    "attn.c_proj.weight": (768, 768),
    "mlp.c_fc.weight": (768, 3072),
    "mlp.c_proj.weight": (3072, 768),
}


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def clean_output_dir() -> None:
    """Remove a previous checkpoint from DST, but never this script or the report."""
    DST.mkdir(parents=True, exist_ok=True)
    for path in DST.iterdir():
        if path.is_file() and (path.suffix == ".safetensors" or path.name.endswith(".index.json")):
            path.unlink()


def main() -> None:
    src = load_file(str(SRC))
    print(f"loaded {len(src)} tensors from {SRC}")

    # --- targeting checks: every intended name exists, nothing is missing ---
    missing = [k for k in PROJECTIONS + BUFFERS if k not in src]
    check(not missing, f"input is missing expected tensors: {missing}")
    for name in PROJECTIONS:
        suffix = name.split(".", 2)[2]
        check(
            tuple(src[name].shape) == EXPECTED_SHAPES[suffix],
            f"{name}: unexpected shape {tuple(src[name].shape)}",
        )

    projections = set(PROJECTIONS)
    buffers = set(BUFFERS)

    out: dict[str, torch.Tensor] = {}
    for name, tensor in src.items():
        if name in buffers:
            continue  # drop non-parameter causal-mask buffers
        if name in projections:
            out[name] = tensor.to(torch.bfloat16).contiguous()
        else:
            check(
                tensor.dtype == torch.float32,
                f"{name}: expected a float32 input tensor, got {tensor.dtype}",
            )
            out[name] = tensor.contiguous()

    # ---------------- required checks, before anything is written ----------
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    check(n_bf16 == 48, f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
    check(
        out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16,
        f"h.0.attn.c_attn.weight is {out['h.0.attn.c_attn.weight'].dtype}, expected bfloat16",
    )
    check(
        out["wte.weight"].dtype == torch.float32,
        f"wte.weight is {out['wte.weight'].dtype}, expected float32",
    )
    check(len(out) == 148, f"expected exactly 148 output tensors, got {len(out)}")

    # ---------------- supporting checks -----------------------------------
    check(set(out) == set(src) - buffers, "output key set is not input minus the buffers")
    non_f32 = {k for k, t in out.items() if k not in projections and t.dtype != torch.float32}
    check(not non_f32, f"non-projection tensors left non-float32: {sorted(non_f32)}")
    for name in PROJECTIONS:
        check(out[name].dtype == torch.bfloat16, f"{name} was not cast to bfloat16")
        check(tuple(out[name].shape) == tuple(src[name].shape), f"{name} changed shape")
    for name, tensor in out.items():
        if name not in projections:
            check(torch.equal(tensor, src[name]), f"{name} values changed")

    # ---------------- sharded write ---------------------------------------
    split = split_torch_state_dict_into_shards(out, max_shard_size=MAX_SHARD_BYTES)
    check(split.is_sharded, "expected a sharded layout, got a single file")

    for filename, names in split.filename_to_tensors.items():
        total = sum(out[n].numel() * out[n].element_size() for n in names)
        if total > MAX_SHARD_BYTES:
            check(
                len(names) == 1,
                f"{filename}: {total} bytes over the {MAX_SHARD_BYTES} budget "
                f"with {len(names)} tensors",
            )

    clean_output_dir()

    for filename, names in split.filename_to_tensors.items():
        save_file(
            {n: out[n] for n in names},
            str(DST / filename),
            metadata={"format": "pt"},
        )

    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    check(set(index["weight_map"]) == set(out), "weight_map does not cover every tensor")
    (DST / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    # ---------------- verify what actually landed on disk ------------------
    written: dict[str, torch.Tensor] = {}
    for filename in sorted(split.filename_to_tensors):
        shard = load_file(str(DST / filename))
        dup = set(shard) & set(written)
        check(not dup, f"tensor written to more than one shard: {sorted(dup)}")
        written.update(shard)

    check(len(written) == 148, f"wrote {len(written)} tensors, expected 148")
    check(
        sum(1 for t in written.values() if t.dtype == torch.bfloat16) == 48,
        "written checkpoint does not hold exactly 48 bfloat16 tensors",
    )
    check(
        written["h.0.attn.c_attn.weight"].dtype == torch.bfloat16,
        "h.0.attn.c_attn.weight not bfloat16 on disk",
    )
    check(written["wte.weight"].dtype == torch.float32, "wte.weight not float32 on disk")
    for name, tensor in written.items():
        check(torch.equal(tensor, out[name]), f"{name} does not round-trip bit-exactly")

    print(f"wrote {len(written)} tensors ({n_bf16} bfloat16) to {DST}")
    for filename in sorted(split.filename_to_tensors):
        names = split.filename_to_tensors[filename]
        total = sum(out[n].numel() * out[n].element_size() for n in names)
        print(f"  {filename}: {len(names):3d} tensors, {total:,} bytes of tensor data")


if __name__ == "__main__":
    main()
