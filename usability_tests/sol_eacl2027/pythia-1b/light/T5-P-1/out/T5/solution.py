import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUT_DIR = Path("out/T5")
MAX_SHARD_BYTES = 512 * 1024 * 1024


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def main():
    with CONFIG_PATH.open() as handle:
        config = json.load(handle)

    rank = config["r"]
    scale = config["lora_alpha"] / rank
    fan_in_fan_out = config["fan_in_fan_out"]
    base = load_file(str(BASE_PATH), device="cpu")
    adapter = load_file(str(ADAPTER_PATH), device="cpu")

    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    prefix = "base_model.model."
    a_names = sorted(name for name in adapter if name.endswith(a_suffix))
    assert len(a_names) == 16, f"expected 16 LoRA A tensors, found {len(a_names)}"

    merged_count = 0
    for a_name in a_names:
        stem = a_name[: -len(a_suffix)]
        b_name = stem + b_suffix
        assert b_name in adapter, f"missing paired tensor {b_name}"
        assert stem.startswith(prefix), f"unexpected adapter name {a_name}"
        base_name = stem[len(prefix) :] + ".weight"
        assert base_name in base, f"missing base tensor {base_name}"

        a = adapter[a_name]
        b = adapter[b_name]
        assert a.shape[0] == rank and b.shape[1] == rank, (
            f"rank mismatch for {stem}: A={tuple(a.shape)}, B={tuple(b.shape)}"
        )
        delta = torch.matmul(b.float(), a.float()) * scale
        if fan_in_fan_out:
            delta = delta.T
        old = base[base_name]
        assert delta.shape == old.shape, (
            f"shape mismatch for {base_name}: delta={tuple(delta.shape)}, "
            f"base={tuple(old.shape)}"
        )
        base[base_name] = (old.float() + delta).to(old.dtype)
        merged_count += 1
        del delta, old
        gc.collect()

    # All required validation is deliberately completed before the first write.
    assert merged_count == 16, f"expected 16 merged pairs, got {merged_count}"
    assert len(adapter) == 2 * merged_count, "adapter contains unpaired or extra tensors"
    assert not any("lora_" in name for name in base), "LoRA tensor leaked into output"
    check_name = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert tuple(base[check_name].shape) == (6144, 2048), (
        f"wrong merged shape: {tuple(base[check_name].shape)}"
    )
    assert len(base) == 244, f"expected 244 output tensors, found {len(base)}"

    # The two embedding matrices are required to occupy dedicated shards.
    dedicated = {"gpt_neox.embed_in.weight", "embed_out.weight"}
    shards = []
    current = []
    current_bytes = 0
    for name in sorted(base):
        size = tensor_bytes(base[name])
        assert size <= MAX_SHARD_BYTES, f"single tensor exceeds shard cap: {name}"
        if name in dedicated:
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

    total = len(shards)
    weight_map = {}
    for number, names in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{total:05d}.safetensors"
        shard = {name: base[name] for name in names}
        shard_size = sum(tensor_bytes(tensor) for tensor in shard.values())
        assert shard_size <= MAX_SHARD_BYTES, f"oversized shard {filename}"
        save_file(shard, str(OUT_DIR / filename), metadata={"format": "pt"})
        weight_map.update({name: filename for name in names})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in base.values())},
        "weight_map": weight_map,
    }
    with (OUT_DIR / "model.safetensors.index.json").open("w") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Merged {merged_count} LoRA pairs and wrote {len(base)} tensors in {total} shards")


if __name__ == "__main__":
    main()
