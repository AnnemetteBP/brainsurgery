"""
T3: Mixed-precision export with sharding (Pythia-1B).

Loads inputs/base/model.safetensors, casts the 64 large projection matrices
to bfloat16, upcasts everything else to float32, drops the 48 non-parameter
buffers, and writes a sharded safetensors checkpoint (max 256 MiB of tensor
data per shard, with any single oversized tensor alone in its own shard)
plus a model.safetensors.index.json.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUTPUT_DIR = HERE

MAX_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB

# Exactly the 4 projection-matrix name patterns, per layer, that must become bf16.
BF16_PATTERNS = [
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.dense\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_h_to_4h\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_4h_to_h\.weight$"),
]

# Exactly the 3 non-parameter buffer name patterns, per layer, to delete.
BUFFER_PATTERNS = [
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.bias$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.masked_bias$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.rotary_emb\.inv_freq$"),
]


def is_bf16_target(name: str) -> bool:
    return any(p.match(name) for p in BF16_PATTERNS)


def is_buffer(name: str) -> bool:
    return any(p.match(name) for p in BUFFER_PATTERNS)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"input checkpoint not found: {INPUT_PATH}")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(INPUT_PATH), framework="pt") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    total_in = len(tensors)

    kept: dict[str, torch.Tensor] = {}
    dropped = []
    for name, t in tensors.items():
        if is_buffer(name):
            dropped.append(name)
            continue
        if is_bf16_target(name):
            kept[name] = t.to(torch.bfloat16)
        else:
            kept[name] = t.to(torch.float32)

    # --- Required checks: fail loudly before writing anything ---
    bf16_names = [n for n, t in kept.items() if t.dtype == torch.bfloat16]
    if len(bf16_names) != 64:
        raise AssertionError(f"expected exactly 64 bfloat16 tensors, got {len(bf16_names)}")

    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if kept[probe].dtype != torch.bfloat16:
        raise AssertionError(f"{probe} must be bfloat16, got {kept[probe].dtype}")

    if kept["gpt_neox.embed_in.weight"].dtype != torch.float32:
        raise AssertionError("gpt_neox.embed_in.weight must be float32")

    if len(kept) != 196:
        raise AssertionError(f"expected exactly 196 output tensors, got {len(kept)}")

    if len(dropped) != 48:
        raise AssertionError(f"expected exactly 48 dropped buffers, got {len(dropped)}")

    if total_in != len(kept) + len(dropped):
        raise AssertionError("tensor count mismatch: input != kept + dropped")

    for name, t in kept.items():
        if not is_bf16_target(name) and t.dtype != torch.float32:
            raise AssertionError(f"{name} should be float32 but is {t.dtype}")

    # --- Sharding ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names_in_order = list(kept.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in names_in_order:
        nbytes = tensor_nbytes(kept[name])
        if nbytes > MAX_SHARD_BYTES:
            # Oversized tensor: gets its own shard, alone.
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
    digits = max(5, len(str(n_shards)))
    weight_map: dict[str, str] = {}
    total_size = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:0{digits}d}-of-{n_shards:0{digits}d}.safetensors"
        shard_tensors = {n: kept[n].contiguous() for n in shard_names}
        save_file(shard_tensors, str(OUTPUT_DIR / shard_filename))
        for n in shard_names:
            weight_map[n] = shard_filename
            total_size += tensor_nbytes(kept[n])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(kept)} tensors across {n_shards} shards to {OUTPUT_DIR}")
    print(f"Dropped {len(dropped)} buffers")


if __name__ == "__main__":
    main()
