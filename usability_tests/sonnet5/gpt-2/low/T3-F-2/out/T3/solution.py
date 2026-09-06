"""T3: Mixed-precision export with sharding (GPT-2 124M).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md).
Casts the 48 large projection matrices to bfloat16, keeps everything else
(embeddings, norms, biases) in float32, drops the 12 non-parameter causal-
mask buffers, and writes a sharded safetensors checkpoint with an index
file, each shard <= 64 MiB of tensor data (a single oversize tensor gets
its own shard).
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE.parent / "T3"

MAX_SHARD_BYTES = 64 * 1024 * 1024  # 64 MiB, tensor data only

# Exactly the 48 large projection matrices to cast to bf16.
BF16_PATTERN = re.compile(
    r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)
# The 12 non-parameter causal-mask buffers to drop.
BUFFER_PATTERN = re.compile(r"^h\.\d+\.attn\.bias$")


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(IN_PATH), framework="pt") as f:
        for key in f.keys():
            if BUFFER_PATTERN.match(key):
                continue  # drop non-parameter buffers
            t = f.get_tensor(key)
            if BF16_PATTERN.match(key):
                t = t.to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            tensors[key] = t.contiguous()

    # --- Required checks: fail loudly before writing anything. ---
    n_bf16 = sum(1 for t in tensors.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 48, f"expected 48 bfloat16 tensors, got {n_bf16}"
    assert tensors["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert tensors["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(tensors) == 148, f"expected 148 tensors in output, got {len(tensors)}"
    for name in tensors:
        assert not BUFFER_PATTERN.match(name), f"buffer leaked into output: {name}"
    for name, t in tensors.items():
        if BF16_PATTERN.match(name):
            assert t.dtype == torch.bfloat16, f"{name} should be bfloat16"
        else:
            assert t.dtype == torch.float32, f"{name} should be float32"

    # --- Greedy bin-packing into shards, each <= MAX_SHARD_BYTES of tensor data. ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Stable order: as encountered in the input file.
    names = list(tensors.keys())

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_nbytes(tensors[name])
        if size > MAX_SHARD_BYTES:
            # Oversize tensor gets its own shard.
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename))
        for name, t in shard_tensors.items():
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(t)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"wrote {len(weight_map)} tensors across {n_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
