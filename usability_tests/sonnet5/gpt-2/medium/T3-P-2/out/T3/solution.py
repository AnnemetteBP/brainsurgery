"""
T3: Mixed-precision export with sharding (GPT-2 124M).

Cast the 48 large projection matrices to bfloat16, keep everything else
(float32) unchanged, drop the 12 non-parameter causal-mask buffers, and
write a sharded safetensors checkpoint with an index file where no shard
exceeds 64 MiB of tensor data (a single oversized tensor gets its own shard).
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
SHARD_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB, tensor data only

NUM_LAYERS = 12

# Exactly the 48 projection matrices that must become bfloat16.
BF16_PATTERN = re.compile(
    r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)

# The 12 non-parameter buffers to drop.
BUFFER_PATTERN = re.compile(r"^h\.\d+\.attn\.bias$")


def load_all_tensors(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    tensors = load_all_tensors(INPUT_PATH)

    assert len(tensors) == 160, f"expected 160 input tensors, got {len(tensors)}"

    # Drop non-parameter buffers.
    buffer_keys = [k for k in tensors if BUFFER_PATTERN.match(k)]
    assert len(buffer_keys) == NUM_LAYERS, (
        f"expected {NUM_LAYERS} causal-mask buffers, found {len(buffer_keys)}"
    )
    for k in buffer_keys:
        del tensors[k]

    # Cast the projection matrices to bfloat16; leave everything else as-is.
    bf16_keys = [k for k in tensors if BF16_PATTERN.match(k)]
    assert len(bf16_keys) == 4 * NUM_LAYERS, (
        f"expected {4 * NUM_LAYERS} projection matrices to cast, found {len(bf16_keys)}"
    )
    output = {}
    for k, t in tensors.items():
        if k in set(bf16_keys):
            output[k] = t.to(torch.bfloat16)
        else:
            output[k] = t

    # --- Required checks: fail loudly before writing anything ---
    n_bf16 = sum(1 for t in output.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 48, f"required check failed: expected 48 bfloat16 tensors, got {n_bf16}"

    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "required check failed: h.0.attn.c_attn.weight is not bfloat16"
    )

    assert output["wte.weight"].dtype == torch.float32, (
        "required check failed: wte.weight is not float32"
    )

    assert len(output) == 148, (
        f"required check failed: expected 148 output tensors, got {len(output)}"
    )

    for k in output:
        if BF16_PATTERN.match(k):
            assert output[k].dtype == torch.bfloat16, f"{k} should be bfloat16"
        else:
            assert output[k].dtype == torch.float32, f"{k} should remain float32"

    # --- Shard assignment ---
    # Greedy bin-packing in a stable (insertion) order: each shard fills up to
    # the byte limit; a tensor bigger than the limit alone gets its own shard.
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    shard_bytes: list[int] = []
    for k, t in output.items():
        nbytes = tensor_nbytes(t)
        placed = False
        if nbytes <= SHARD_LIMIT_BYTES:
            for i in range(len(shards)):
                if shard_bytes[i] + nbytes <= SHARD_LIMIT_BYTES:
                    shards[i].append(k)
                    shard_bytes[i] += nbytes
                    placed = True
                    break
        if not placed:
            shards.append([k])
            shard_bytes.append(nbytes)

    # Verify shard byte budgets.
    for i, keys in enumerate(shards):
        total = sum(tensor_nbytes(output[k]) for k in keys)
        if len(keys) > 1:
            assert total <= SHARD_LIMIT_BYTES, (
                f"shard {i} exceeds byte limit: {total} bytes"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, keys in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {k: output[k] for k in keys}
        save_file(shard_tensors, str(OUT_DIR / shard_name))
        for k in keys:
            weight_map[k] = shard_name
            total_size += tensor_nbytes(output[k])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(output)} tensors across {num_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
