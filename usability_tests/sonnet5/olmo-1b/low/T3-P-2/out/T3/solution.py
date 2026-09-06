"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf)

Casts the 112 large projection matrices (q/k/v/o_proj, gate/up/down_proj in
each of 16 layers) to bfloat16, keeps everything else (embeddings, lm_head)
float32, and writes a sharded safetensors checkpoint with a
model.safetensors.index.json, respecting a 256 MiB per-shard tensor-data cap
(oversized single tensors get their own shard).
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "inputs", "base")
OUT_DIR = os.path.join(HERE)  # out/T3

SHARD_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MiB, tensor data only

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def main():
    index_path = os.path.join(IN_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        in_index = json.load(f)
    weight_map = in_index["weight_map"]

    # Load every tensor into memory, casting the target projection matrices.
    tensors = {}
    shard_files = sorted(set(weight_map.values()))
    open_files = {}
    try:
        for shard_file in shard_files:
            open_files[shard_file] = safe_open(
                os.path.join(IN_DIR, shard_file), framework="pt"
            )
        for name, shard_file in weight_map.items():
            t = open_files[shard_file].get_tensor(name)
            if PROJ_RE.match(name):
                t = t.to(torch.bfloat16)
            else:
                assert t.dtype == torch.float32, f"unexpected input dtype for {name}: {t.dtype}"
            tensors[name] = t
    finally:
        del open_files

    # ---- Required checks (fail loudly before writing) ----
    bf16_names = [n for n, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 112, f"expected 112 bfloat16 tensors, got {len(bf16_names)}"

    q0 = "model.layers.0.self_attn.q_proj.weight"
    assert tensors[q0].dtype == torch.bfloat16, f"{q0} is not bfloat16"

    embed_name = "model.embed_tokens.weight"
    assert tensors[embed_name].dtype == torch.float32, f"{embed_name} is not float32"

    assert len(tensors) == 114, f"expected 114 tensors total, got {len(tensors)}"

    for name in tensors:
        if not PROJ_RE.match(name):
            assert tensors[name].dtype == torch.float32, f"{name} should be float32"

    # ---- Shard assignment ----
    # Greedy bin-packing in a stable (index) order; a tensor bigger than the
    # limit gets its own shard.
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names_in_order = list(weight_map.keys())

    shards: list[list[str]] = []
    shard_sizes: list[int] = []
    for name in names_in_order:
        size = tensor_nbytes(tensors[name])
        placed = False
        if size <= SHARD_LIMIT_BYTES:
            for i in range(len(shards)):
                if shard_sizes[i] + size <= SHARD_LIMIT_BYTES:
                    shards[i].append(name)
                    shard_sizes[i] += size
                    placed = True
                    break
        if not placed:
            shards.append([name])
            shard_sizes.append(size)

    n_shards = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)

    out_weight_map = {}
    for i, names in enumerate(shards):
        shard_filename = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name].contiguous() for name in names}
        save_file(
            shard_tensors,
            os.path.join(OUT_DIR, shard_filename),
            metadata={"format": "pt"},
        )
        for name in names:
            out_weight_map[name] = shard_filename

    total_size = sum(tensor_nbytes(t) for t in tensors.values())
    out_index = {
        "metadata": {"total_size": total_size},
        "weight_map": out_weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(out_index, f, indent=2)

    print(f"Wrote {len(tensors)} tensors across {n_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
