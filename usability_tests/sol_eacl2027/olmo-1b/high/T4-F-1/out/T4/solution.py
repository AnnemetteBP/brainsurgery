#!/usr/bin/env python3
"""Verify and task-vector merge the two OLMo fine-tunes."""

from __future__ import annotations

import json
import os
import struct
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_PATH = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_PATH = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_PATH = Path(__file__).resolve().parent / "model.safetensors"
TEMP_PATH = OUTPUT_PATH.with_suffix(".safetensors.tmp")
LAMBDA = 0.4
EXPECTED_TENSORS = 114

MLP_SHAPES = {
    "gate_proj": [8192, 2048],
    "up_proj": [8192, 2048],
    "down_proj": [2048, 8192],
}
MLP_NAMES = {
    f"model.layers.{layer}.mlp.{projection}.weight"
    for layer in range(16)
    for projection in MLP_SHAPES
}


class ShardedCheckpoint:
    """Read tensors from an indexed sharded safetensors checkpoint."""

    def __init__(self, directory: Path, stack: ExitStack) -> None:
        index_path = directory / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as stream:
            index = json.load(stream)
        self.weight_map: dict[str, str] = index["weight_map"]
        shard_names = sorted(set(self.weight_map.values()))
        self.shards = {
            name: stack.enter_context(safe_open(directory / name, framework="pt", device="cpu"))
            for name in shard_names
        }

        actual_keys = {key for shard in self.shards.values() for key in shard.keys()}
        assert actual_keys == set(self.weight_map), (
            "Base index and shard tensor names differ: "
            f"index-only={sorted(set(self.weight_map) - actual_keys)}, "
            f"shard-only={sorted(actual_keys - set(self.weight_map))}"
        )

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_slice(self, name: str):
        return self.shards[self.weight_map[name]].get_slice(name)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.shards[self.weight_map[name]].get_tensor(name)


def tensor_spec(checkpoint, name: str) -> tuple[list[int], str]:
    view = checkpoint.get_slice(name)
    return list(view.get_shape()), view.get_dtype()


def write_safetensors_streaming(
    path: Path,
    names: list[str],
    specs: dict[str, tuple[list[int], str]],
    base: ShardedCheckpoint,
    ft1,
    ft2,
) -> int:
    """Write one safetensors file while retaining only one result tensor."""
    bytes_per_element = {"F32": 4}
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for name in names:
        shape, dtype = specs[name]
        assert dtype in bytes_per_element, f"Unsupported output dtype for {name}: {dtype}"
        nbytes = bytes_per_element[dtype]
        for dimension in shape:
            nbytes *= dimension
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)

    merged_count = 0
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded_header)))
        stream.write(encoded_header)
        for name in names:
            base_tensor = base.get_tensor(name)
            if name in MLP_NAMES:
                assert base_tensor.dtype == torch.float32, f"Base tensor is not float32: {name}"
                ft1_tensor = ft1.get_tensor(name)
                ft2_tensor = ft2.get_tensor(name)
                result = (
                    base_tensor
                    + LAMBDA * (ft1_tensor - base_tensor)
                    + LAMBDA * (ft2_tensor - base_tensor)
                )
                merged_count += 1
            else:
                result = base_tensor

            result = result.contiguous()
            raw = memoryview(result.numpy()).cast("B")
            expected_bytes = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
            assert len(raw) == expected_bytes, f"Serialized size mismatch for {name}"
            stream.write(raw)

        stream.flush()
        os.fsync(stream.fileno())

    assert merged_count == 48, f"Expected to merge 48 tensors, merged {merged_count}"
    expected_size = 8 + len(encoded_header) + offset
    assert path.stat().st_size == expected_size, (
        f"Output byte size mismatch: expected {expected_size}, got {path.stat().st_size}"
    )
    return merged_count


def main() -> None:
    assert len(MLP_NAMES) == 48, f"Expected 48 MLP names, constructed {len(MLP_NAMES)}"
    if TEMP_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite stale temporary output: {TEMP_PATH}")

    with ExitStack() as stack:
        base = ShardedCheckpoint(BASE_DIR, stack)
        ft1 = stack.enter_context(safe_open(FT1_PATH, framework="pt", device="cpu"))
        ft2 = stack.enter_context(safe_open(FT2_PATH, framework="pt", device="cpu"))

        base_names = base.keys()
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        assert base_names == ft1_names == ft2_names, (
            "Checkpoint tensor names differ: "
            f"base_vs_ft1={sorted(base_names ^ ft1_names)}, "
            f"base_vs_ft2={sorted(base_names ^ ft2_names)}"
        )
        assert len(base_names) == EXPECTED_TENSORS, (
            f"Expected {EXPECTED_TENSORS} input tensors, found {len(base_names)}"
        )
        missing_mlp = MLP_NAMES - base_names
        assert not missing_mlp, f"Missing expected MLP tensors: {sorted(missing_mlp)}"

        names = sorted(base_names)
        specs: dict[str, tuple[list[int], str]] = {}
        for name in names:
            base_spec = tensor_spec(base, name)
            ft1_spec = tensor_spec(ft1, name)
            ft2_spec = tensor_spec(ft2, name)
            assert base_spec == ft1_spec == ft2_spec, (
                f"Shape/dtype mismatch for {name}: "
                f"base={base_spec}, ft1={ft1_spec}, ft2={ft2_spec}"
            )
            specs[name] = base_spec

        for layer in range(16):
            for projection, expected_shape in MLP_SHAPES.items():
                name = f"model.layers.{layer}.mlp.{projection}.weight"
                assert specs[name] == (expected_shape, "F32"), (
                    f"Unexpected MLP tensor spec for {name}: {specs[name]}"
                )

        # Complete the frozen-backbone precondition before creating any output.
        unchanged_names = sorted(base_names - MLP_NAMES)
        assert len(unchanged_names) == 66, (
            f"Expected 66 frozen tensors, found {len(unchanged_names)}"
        )
        for name in unchanged_names:
            base_tensor = base.get_tensor(name)
            assert torch.equal(base_tensor, ft1.get_tensor(name)), (
                f"Frozen tensor differs between base and ft1: {name}"
            )
            assert torch.equal(base_tensor, ft2.get_tensor(name)), (
                f"Frozen tensor differs between base and ft2: {name}"
            )

        merged_count = write_safetensors_streaming(
            TEMP_PATH, names, specs, base, ft1, ft2
        )
        assert merged_count == len(MLP_NAMES) == 48

        with safe_open(TEMP_PATH, framework="pt", device="cpu") as output:
            output_names = set(output.keys())
            assert output_names == base_names, (
                "Output tensor names differ from input tensor names: "
                f"missing={sorted(base_names - output_names)}, "
                f"extra={sorted(output_names - base_names)}"
            )
            assert len(output_names) == EXPECTED_TENSORS, (
                f"Expected {EXPECTED_TENSORS} output tensors, found {len(output_names)}"
            )
            for name in names:
                assert tensor_spec(output, name) == specs[name], (
                    f"Output shape/dtype mismatch for {name}: "
                    f"expected={specs[name]}, got={tensor_spec(output, name)}"
                )

    os.replace(TEMP_PATH, OUTPUT_PATH)
    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as output:
        assert len(output.keys()) == EXPECTED_TENSORS
    print(f"Wrote {OUTPUT_PATH} with {EXPECTED_TENSORS} tensors; merged {merged_count} MLP tensors.")


if __name__ == "__main__":
    main()
