"""
T3: Mixed-precision export with sharding (Pythia-1B).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). This
task is a precise per-tensor dtype/drop rule plus a byte-budget sharding
rule; a direct script gives exact control over both without fighting a
higher-level tool's own conventions.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

IN_PATH = Path(__file__).resolve().parents[2] / "inputs" / "base" / "model.safetensors"
OUT_DIR = Path(__file__).resolve().parent
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

# The 64 large projection matrices to cast to bfloat16.
BF16_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)

# The 48 non-parameter buffers to drop entirely.
DROP_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}

    with safe_open(str(IN_PATH), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            if DROP_PATTERN.match(key):
                continue
            t = f.get_tensor(key)
            if BF16_PATTERN.match(key):
                tensors[key] = t.to(torch.bfloat16)
            else:
                tensors[key] = t.to(torch.float32)

    # --- Required checks: fail loudly before writing anything. ---
    bf16_keys = [k for k, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_keys) == 64, f"expected 64 bfloat16 tensors, got {len(bf16_keys)}"
    assert tensors["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
    assert tensors["gpt_neox.embed_in.weight"].dtype == torch.float32
    assert len(tensors) == 196, f"expected 196 tensors, got {len(tensors)}"
    for k, t in tensors.items():
        if BF16_PATTERN.match(k):
            assert t.dtype == torch.bfloat16, k
        else:
            assert t.dtype == torch.float32, k
    for k in tensors:
        assert not DROP_PATTERN.match(k), f"buffer not dropped: {k}"

    # --- Greedy sharding by declared byte budget. ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names = list(tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in names:
        nbytes = tensor_nbytes(tensors[name])
        if nbytes > MAX_SHARD_BYTES:
            # Goes alone in its own shard.
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensors[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(tensors)} tensors across {n_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
