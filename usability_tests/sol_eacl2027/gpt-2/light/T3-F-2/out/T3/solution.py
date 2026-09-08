#!/usr/bin/env python3
"""Create the mixed-precision, sharded GPT-2 checkpoint for T3."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
MAX_SHARD_BYTES = 64 * 1024 * 1024
PROJECTION_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(state: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Greedily shard in sorted-name order; oversized tensors stand alone."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(state):
        tensor = state[name]
        size = tensor_bytes(tensor)
        if current and (size > MAX_SHARD_BYTES or current_bytes + size > MAX_SHARD_BYTES):
            shards.append(current)
            current = {}
            current_bytes = 0
        if size > MAX_SHARD_BYTES:
            shards.append({name: tensor})
        else:
            current[name] = tensor
            current_bytes += size

    if current:
        shards.append(current)
    return shards


def main() -> None:
    source = load_file(INPUT, device="cpu")
    projection_names = {
        f"h.{layer}.{suffix}"
        for layer in range(12)
        for suffix in PROJECTION_SUFFIXES
    }
    buffer_names = {f"h.{layer}.attn.bias" for layer in range(12)}

    assert len(source) == 160, f"expected 160 input tensors, got {len(source)}"
    assert projection_names <= source.keys(), "one or more projection matrices are missing"
    assert buffer_names <= source.keys(), "one or more causal-mask buffers are missing"

    state: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if name in buffer_names:
            continue
        state[name] = tensor.to(torch.bfloat16) if name in projection_names else tensor.to(torch.float32)

    # Required pre-write checks (plus exact-set checks to prevent overmatching).
    bf16_names = {name for name, tensor in state.items() if tensor.dtype == torch.bfloat16}
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert bf16_names == projection_names, "bfloat16 tensor set is not exactly the projection set"
    assert state["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert state["wte.weight"].dtype == torch.float32
    assert len(state) == 148, f"expected 148 output tensors, got {len(state)}"
    assert all(t.dtype == torch.float32 for n, t in state.items() if n not in projection_names)
    assert not (buffer_names & state.keys()), "a causal-mask buffer survived"

    shards = make_shards(state)
    for shard in shards:
        sizes = [tensor_bytes(tensor) for tensor in shard.values()]
        assert len(shard) == 1 or sum(sizes) <= MAX_SHARD_BYTES

    # Remove only checkpoint artifacts created by an earlier invocation.
    for old_shard in OUTPUT.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    index_path = OUTPUT / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, OUTPUT / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in state.values())},
        "weight_map": weight_map,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(state)} tensors in {shard_count} shards to {OUTPUT}")


if __name__ == "__main__":
    main()
