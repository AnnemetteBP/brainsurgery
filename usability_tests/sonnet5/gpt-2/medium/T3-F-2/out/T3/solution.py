"""
T3: Mixed-precision export with sharding (GPT-2 124M), condition F.

Plain script on top of `torch` + `safetensors` (both in F-allowed.md). No
mergekit/transformers export path was used because this task needs precise,
name-exact control over which 48 tensors get cast and which 12 buffers get
dropped -- easier and more auditable as a direct safetensors rewrite than
coercing a generic dtype-export tool to do partial-precision + buffer drop +
custom shard sizing at once.

Steps:
  1. Load every tensor from inputs/base/model.safetensors.
  2. Drop the 12 non-parameter causal-mask buffers `h.<i>.attn.bias`.
  3. Cast exactly the 48 projection matrices to bfloat16; leave everything
     else (embeddings, norms, all biases) untouched in float32.
  4. Run the required checks; abort loudly if any fails.
  5. Greedily pack tensors into shards of at most 64 MiB of tensor data
     (oversized single tensors, i.e. wte.weight, get their own shard), write
     the shards plus model.safetensors.index.json.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")
SHARD_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB, tensor data only

NUM_LAYERS = 12
PROJ_SUFFIXES = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]
BF16_KEYS = {f"h.{i}.{suf}" for i in range(NUM_LAYERS) for suf in PROJ_SUFFIXES}
DROP_KEYS = {f"h.{i}.attn.bias" for i in range(NUM_LAYERS)}


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    sd = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def main() -> None:
    sd = load_state_dict(IN_PATH)
    assert len(sd) == 160, f"expected 160 input tensors, got {len(sd)}"
    assert BF16_KEYS <= sd.keys(), "missing expected projection tensors"
    assert DROP_KEYS <= sd.keys(), "missing expected buffer tensors"

    out = {}
    for k, t in sd.items():
        if k in DROP_KEYS:
            continue
        if k in BF16_KEYS:
            out[k] = t.to(torch.bfloat16).contiguous()
        else:
            out[k] = t.contiguous()

    # --- Required checks: fail loudly before writing anything. ---
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 48:
        sys.exit(f"FAIL: expected exactly 48 bfloat16 tensors, got {n_bf16}")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        sys.exit("FAIL: h.0.attn.c_attn.weight is not bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        sys.exit("FAIL: wte.weight is not float32")
    if len(out) != 148:
        sys.exit(f"FAIL: expected exactly 148 output tensors, got {len(out)}")
    for k in DROP_KEYS:
        if k in out:
            sys.exit(f"FAIL: buffer {k} was not dropped")
    for k in BF16_KEYS:
        if out[k].dtype != torch.bfloat16:
            sys.exit(f"FAIL: {k} is not bfloat16")
    for k, t in out.items():
        if k not in BF16_KEYS and t.dtype != torch.float32:
            sys.exit(f"FAIL: {k} should be float32 but is {t.dtype}")

    # --- Greedy shard packing, preserving key order; oversized tensors get
    # their own shard. ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    for k, t in out.items():
        nb = tensor_nbytes(t)
        if nb > SHARD_LIMIT_BYTES:
            if cur:
                shards.append(cur)
                cur, cur_bytes = {}, 0
            shards.append({k: t})
            continue
        if cur and cur_bytes + nb > SHARD_LIMIT_BYTES:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = t
        cur_bytes += nb
    if cur:
        shards.append(cur)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = len(shards)
    weight_map = {}
    total_size = 0
    digits = len(str(n_shards))
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:0{digits}d}-of-{n_shards:0{digits}d}.safetensors"
        save_file(shard, str(OUT_DIR / fname), metadata={"format": "pt"})
        for k, t in shard.items():
            weight_map[k] = fname
            total_size += tensor_nbytes(t)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"wrote {n_shards} shards, {len(out)} tensors, {total_size} bytes tensor data")


if __name__ == "__main__":
    main()
