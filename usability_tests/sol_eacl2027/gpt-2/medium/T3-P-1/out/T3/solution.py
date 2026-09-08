import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
MAX_SHARD_BYTES = 64 * 1024 * 1024

PROJECTION_RE = re.compile(
    r"^h\.(?:[0-9]|1[01])\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight$"
)
BUFFER_RE = re.compile(r"^h\.(?:[0-9]|1[01])\.attn\.bias$")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name, tensor in tensors.items():
        size = tensor_nbytes(tensor)

        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current = {}
                current_bytes = 0
            current[name] = tensor
            current_bytes += size

    if current:
        shards.append(current)
    return shards


def main() -> None:
    source = load_file(INPUT, device="cpu")
    output: dict[str, torch.Tensor] = {}

    for name, tensor in source.items():
        if BUFFER_RE.fullmatch(name):
            continue
        dtype = torch.bfloat16 if PROJECTION_RE.fullmatch(name) else torch.float32
        output[name] = tensor.to(dtype=dtype).contiguous()

    # Required pre-write checks.
    bf16_names = [name for name, tensor in output.items() if tensor.dtype == torch.bfloat16]
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, found {len(bf16_names)}"
    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert output["wte.weight"].dtype == torch.float32
    assert len(output) == 148, f"expected 148 output tensors, found {len(output)}"

    # These additional checks guard against an accidental over-broad match or deletion.
    expected_projections = {
        f"h.{layer}.{module}.weight"
        for layer in range(12)
        for module in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    }
    expected_buffers = {f"h.{layer}.attn.bias" for layer in range(12)}
    assert set(bf16_names) == expected_projections
    assert set(source) - set(output) == expected_buffers
    assert all(
        tensor.dtype == torch.float32
        for name, tensor in output.items()
        if name not in expected_projections
    )

    shards = make_shards(output)
    shard_count = len(shards)
    weight_map: dict[str, str] = {}

    for number, shard in enumerate(shards, start=1):
        shard_bytes = sum(tensor_nbytes(tensor) for tensor in shard.values())
        assert shard_bytes <= MAX_SHARD_BYTES or len(shard) == 1
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename)
        weight_map.update({name: filename for name in shard})

    assert set(weight_map) == set(output)
    index = {
        "metadata": {"total_size": sum(tensor_nbytes(tensor) for tensor in output.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT_DIR / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(output)} tensors in {shard_count} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
