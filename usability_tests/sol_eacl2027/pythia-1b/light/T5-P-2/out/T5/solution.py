import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUT_DIR = ROOT / "out/T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024
DEDICATED_TENSORS = {
    "gpt_neox.embed_in.weight",
    "embed_out.weight",
}


def tensor_nbytes(tensor_slice):
    dtype_bytes = {
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
    dtype = tensor_slice.get_dtype()
    if dtype not in dtype_bytes:
        raise RuntimeError(f"Unsupported safetensors dtype: {dtype}")
    elements = 1
    for dimension in tensor_slice.get_shape():
        elements *= dimension
    return elements * dtype_bytes[dtype]


def main():
    config = json.loads(CONFIG_PATH.read_text())
    rank = config["r"]
    scale = config["lora_alpha"] / rank
    if config.get("fan_in_fan_out") is not False:
        raise RuntimeError("This script expects fan_in_fan_out=false")

    adapter = load_file(str(ADAPTER_PATH), device="cpu")
    a_suffix = ".lora_A.weight"
    prefix = "base_model.model."
    pairs = {}
    for name in adapter:
        if name.endswith(a_suffix):
            stem = name[: -len(a_suffix)]
            b_name = stem + ".lora_B.weight"
            if b_name not in adapter:
                raise RuntimeError(f"Missing B factor for {name}")
            base_name = stem[len(prefix) :] + ".weight" if stem.startswith(prefix) else stem + ".weight"
            pairs[base_name] = (name, b_name)

    paired_adapter_keys = {key for pair in pairs.values() for key in pair}
    if paired_adapter_keys != set(adapter):
        extras = sorted(set(adapter) - paired_adapter_keys)
        raise RuntimeError(f"Unpaired or unexpected adapter tensors: {extras}")
    if len(pairs) != 16:
        raise RuntimeError(f"Expected exactly 16 adapter pairs, found {len(pairs)}")

    with safe_open(str(BASE_PATH), framework="pt", device="cpu") as base:
        keys = list(base.keys())
        if len(keys) != 244:
            raise RuntimeError(f"Expected 244 output tensors, found {len(keys)} base tensors")
        if any("lora_" in key for key in keys):
            raise RuntimeError("Output key set contains a lora_ tensor")
        missing = sorted(set(pairs) - set(keys))
        if missing:
            raise RuntimeError(f"Adapter targets missing from base: {missing}")
        probe = "gpt_neox.layers.0.attention.query_key_value.weight"
        if base.get_slice(probe).get_shape() != [6144, 2048]:
            raise RuntimeError(f"Unexpected shape for {probe}: {base.get_slice(probe).get_shape()}")

        sizes = {key: tensor_nbytes(base.get_slice(key)) for key in keys}

    # Embedding matrices are explicitly required to occupy dedicated shards.
    groups = [[key] for key in keys if key in DEDICATED_TENSORS]
    current = []
    current_bytes = 0
    for key in keys:
        if key in DEDICATED_TENSORS:
            continue
        size = sizes[key]
        if size > MAX_SHARD_BYTES:
            raise RuntimeError(f"Tensor exceeds shard limit: {key} ({size} bytes)")
        if current and current_bytes + size > MAX_SHARD_BYTES:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
    if current:
        groups.append(current)

    weight_map = {}
    shard_count = len(groups)
    with safe_open(str(BASE_PATH), framework="pt", device="cpu") as base:
        for shard_number, group in enumerate(groups, 1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {}
            for key in group:
                value = base.get_tensor(key)
                if key in pairs:
                    a_name, b_name = pairs[key]
                    a = adapter[a_name].float()
                    b = adapter[b_name].float()
                    if a.shape != (rank, value.shape[1]) or b.shape != (value.shape[0], rank):
                        raise RuntimeError(f"Adapter/base shape mismatch for {key}")
                    value = (value.float() + scale * (b @ a)).to(value.dtype)
                tensors[key] = value
                weight_map[key] = filename
            shard_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            if shard_bytes > MAX_SHARD_BYTES:
                raise RuntimeError(f"Shard {filename} exceeds the 512 MiB limit")
            save_file(tensors, str(OUT_DIR / filename), metadata={"format": "pt"})

    index = {"metadata": {"total_size": sum(sizes.values())}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(f"Merged {len(pairs)} LoRA pairs into {len(keys)} tensors across {shard_count} shards")


if __name__ == "__main__":
    main()
