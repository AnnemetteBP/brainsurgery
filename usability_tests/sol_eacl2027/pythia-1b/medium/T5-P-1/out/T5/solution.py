#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA weights and export a dense sharded checkpoint."""

import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
ISOLATED_TENSORS = {
    "gpt_neox.embed_in.weight",
    "embed_out.weight",
}

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def tensor_nbytes(handle, name):
    view = handle.get_slice(name)
    dtype = str(view.get_dtype()).split(".")[-1]
    if dtype not in DTYPE_BYTES:
        raise AssertionError(f"unsupported safetensors dtype for {name}: {dtype}")
    return DTYPE_BYTES[dtype] * int(torch.tensor(view.get_shape()).prod().item())


def adapter_pairs(adapter_keys):
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    prefix = "base_model.model."
    a_keys = {key[: -len(suffix_a)]: key for key in adapter_keys if key.endswith(suffix_a)}
    b_keys = {key[: -len(suffix_b)]: key for key in adapter_keys if key.endswith(suffix_b)}
    if set(a_keys) != set(b_keys):
        missing_a = sorted(set(b_keys) - set(a_keys))
        missing_b = sorted(set(a_keys) - set(b_keys))
        raise AssertionError(f"unpaired adapters: missing A={missing_a}, missing B={missing_b}")

    result = {}
    for stem in sorted(a_keys):
        if not stem.startswith(prefix):
            raise AssertionError(f"unexpected adapter namespace: {stem}")
        base_name = stem[len(prefix) :] + ".weight"
        if base_name in result:
            raise AssertionError(f"duplicate adapter target: {base_name}")
        result[base_name] = (a_keys[stem], b_keys[stem])
    return result


def make_shards(base, base_keys, sizes):
    """Pack ordinary tensors greedily; explicitly isolate the two embeddings."""
    shards = []
    current = []
    current_bytes = 0
    for name in base_keys:
        size = sizes[name]
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        elif name in ISOLATED_TENSORS:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
    if current:
        shards.append(current)

    for shard in shards:
        shard_bytes = sum(sizes[name] for name in shard)
        assert shard_bytes <= MAX_SHARD_BYTES or len(shard) == 1, (
            f"invalid shard of {shard_bytes} bytes: {shard}"
        )
    return shards


def main():
    config = json.loads(CONFIG_PATH.read_text())
    rank = config["r"]
    alpha = config["lora_alpha"]
    if config.get("fan_in_fan_out") is not False:
        raise AssertionError("this script expects fan_in_fan_out=false")
    scale = float(alpha) / float(rank)

    with safe_open(BASE_PATH, framework="pt", device="cpu") as base, safe_open(
        ADAPTER_PATH, framework="pt", device="cpu"
    ) as adapter:
        base_keys = sorted(base.keys())
        pairs = adapter_pairs(adapter.keys())

        # All mandated checks happen before the first output checkpoint write.
        assert len(pairs) == 16, f"expected 16 adapter pairs, found {len(pairs)}"
        assert len(base_keys) == 244, f"expected 244 output tensors, found {len(base_keys)}"
        assert not any("lora_" in name for name in base_keys), "LoRA key in output key set"
        probe = "gpt_neox.layers.0.attention.query_key_value.weight"
        assert list(base.get_slice(probe).get_shape()) == [6144, 2048], (
            f"wrong shape for {probe}: {base.get_slice(probe).get_shape()}"
        )
        assert set(pairs).issubset(base_keys), (
            f"adapter targets absent from base: {sorted(set(pairs) - set(base_keys))}"
        )

        sizes = {name: tensor_nbytes(base, name) for name in base_keys}
        shards = make_shards(base, base_keys, sizes)
        shard_count = len(shards)
        weight_map = {}

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for old_shard in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
            old_shard.unlink()
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()

        for number, names in enumerate(shards, start=1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {}
            for name in names:
                value = base.get_tensor(name)
                if name in pairs:
                    a_key, b_key = pairs[name]
                    a = adapter.get_tensor(a_key).float()
                    b = adapter.get_tensor(b_key).float()
                    if tuple(a.shape) != (rank, value.shape[1]):
                        raise AssertionError(f"unexpected A shape for {name}: {tuple(a.shape)}")
                    if tuple(b.shape) != (value.shape[0], rank):
                        raise AssertionError(f"unexpected B shape for {name}: {tuple(b.shape)}")
                    value = (value.float() + scale * (b @ a)).to(value.dtype)
                tensors[name] = value.contiguous()
                weight_map[name] = filename

            save_file(tensors, OUTPUT_DIR / filename)
            del tensors
            gc.collect()

    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Merged {len(pairs)} LoRA pairs into {len(base_keys)} tensors")
    print(f"Wrote {len(shards)} shards and {INDEX_PATH}")


if __name__ == "__main__":
    main()
