"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf)

Standalone script (torch + safetensors only). Loads the sharded input
checkpoint, casts the 112 large projection matrices to bfloat16, leaves
everything else (embeddings, lm_head) as float32, and writes a new sharded
safetensors checkpoint with a valid index file, respecting a 256 MiB per
shard budget for tensor data.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parent.parent / "inputs" / "base"
OUT_DIR = HERE.parent / "T3"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB, tensor data only

# Exactly the 112 projection matrices named in TASK.md: per-layer self-attn
# q/k/v/o projections and mlp gate/up/down projections. Anchored with $ so
# this cannot accidentally match anything else (e.g. norms, biases).
PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)

DTYPE_BYTES = {
    torch.float32: 4,
    torch.bfloat16: 2,
}


def load_index(in_dir: Path) -> dict:
    with open(in_dir / "model.safetensors.index.json") as f:
        return json.load(f)


def main() -> None:
    index = load_index(IN_DIR)
    weight_map = index["weight_map"]
    names = list(weight_map.keys())

    # Open all shard files once, keyed by filename.
    shard_files = sorted(set(weight_map.values()))
    handles = {
        fn: safe_open(str(IN_DIR / fn), framework="pt", device="cpu") for fn in shard_files
    }

    tensors: dict[str, torch.Tensor] = {}
    proj_names = set()
    for name in names:
        fn = weight_map[name]
        t = handles[fn].get_tensor(name)
        if PROJ_RE.match(name):
            proj_names.add(name)
            t = t.to(torch.bfloat16)
        else:
            if t.dtype != torch.float32:
                raise RuntimeError(f"unexpected non-float32 input tensor {name}: {t.dtype}")
        tensors[name] = t.contiguous()

    # --- Required checks: fail loudly before writing anything. ---
    if len(proj_names) != 112:
        raise AssertionError(f"expected exactly 112 projection matrices, matched {len(proj_names)}")

    bf16_count = sum(1 for t in tensors.values() if t.dtype == torch.bfloat16)
    if bf16_count != 112:
        raise AssertionError(f"expected exactly 112 bfloat16 tensors, got {bf16_count}")

    q0 = "model.layers.0.self_attn.q_proj.weight"
    if tensors[q0].dtype != torch.bfloat16:
        raise AssertionError(f"{q0} must be bfloat16, got {tensors[q0].dtype}")

    embed = "model.embed_tokens.weight"
    if tensors[embed].dtype != torch.float32:
        raise AssertionError(f"{embed} must be float32, got {tensors[embed].dtype}")

    if len(tensors) != 114:
        raise AssertionError(f"expected exactly 114 tensors, got {len(tensors)}")

    # --- Sharding: first-fit-decreasing bin packing by tensor byte size. ---
    # A tensor larger than the shard budget gets its own shard alone.
    sizes = {name: t.numel() * DTYPE_BYTES[t.dtype] for name, t in tensors.items()}

    oversized = [n for n, sz in sizes.items() if sz > MAX_SHARD_BYTES]
    packable = [n for n in tensors if n not in oversized]
    packable.sort(key=lambda n: sizes[n], reverse=True)

    bins: list[list[str]] = [[n] for n in oversized]  # each oversized tensor alone
    bin_totals: list[int] = [sizes[n] for n in oversized]

    for name in packable:
        sz = sizes[name]
        placed = False
        for i, total in enumerate(bin_totals):
            # Never merge into a bin that holds an oversized tensor alone.
            if bins[i] and bins[i][0] in oversized:
                continue
            if total + sz <= MAX_SHARD_BYTES:
                bins[i].append(name)
                bin_totals[i] += sz
                placed = True
                break
        if not placed:
            bins.append([name])
            bin_totals.append(sz)

    if len(bins) == 0:
        raise AssertionError("no shards produced")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    num_shards = len(bins)
    weight_map_out = {}
    total_size = 0
    for idx, bin_names in enumerate(bins, start=1):
        shard_filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in bin_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename), metadata={"format": "pt"})
        for name in bin_names:
            weight_map_out[name] = shard_filename
            total_size += sizes[name]

    if len(weight_map_out) != 114:
        raise AssertionError(f"weight_map has {len(weight_map_out)} entries, expected 114")

    index_out = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index_out, f, indent=2, sort_keys=True)

    print(f"Wrote {len(tensors)} tensors across {num_shards} shard(s) to {OUT_DIR}")
    print(f"  bfloat16 tensors: {bf16_count}")
    print(f"  float32 tensors: {len(tensors) - bf16_count}")


if __name__ == "__main__":
    main()
