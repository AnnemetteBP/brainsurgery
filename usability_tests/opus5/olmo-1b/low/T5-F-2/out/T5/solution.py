"""T5: merge a PEFT LoRA adapter into the OLMo-1B base weights and write a
sharded safetensors checkpoint (512 MiB shard budget).

Plain torch + safetensors + huggingface_hub sharding; see REPORT.md.
"""

import json
import os
import shutil

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
LORA = "inputs/lora"
OUT = "out/T5"
MAX_SHARD = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data


def load_base():
    index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        sd.update(load_file(os.path.join(BASE, shard)))
    # keep the base's key order from the index
    return {k: sd[k] for k in index["weight_map"]}


def main():
    cfg = json.load(open(os.path.join(LORA, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    assert not cfg.get("fan_in_fan_out", False), "fan_in_fan_out=True not handled"

    base = load_base()
    n_base = len(base)
    adapter = load_file(os.path.join(LORA, "adapter_model.safetensors"))

    pairs = {}
    for name in adapter:
        assert name.endswith(".weight"), f"unexpected adapter tensor {name}"
        stem, _, kind = name[: -len(".weight")].rpartition(".")
        assert kind in ("lora_A", "lora_B"), f"unexpected adapter tensor {name}"
        target = stem
        for prefix in ("base_model.model.",):
            if target.startswith(prefix):
                target = target[len(prefix) :]
        pairs.setdefault(target, {})[kind] = adapter[name]

    merged = 0
    for target, ab in sorted(pairs.items()):
        assert set(ab) == {"lora_A", "lora_B"}, f"incomplete adapter pair for {target}"
        key = target + ".weight"
        assert key in base, f"no base tensor for adapter target {target}"
        A, B = ab["lora_A"].float(), ab["lora_B"].float()
        W = base[key]
        assert W.dtype == torch.float32, f"{key} is {W.dtype}, expected float32"
        delta = scale * (B @ A)
        assert delta.shape == W.shape, f"delta {tuple(delta.shape)} != {tuple(W.shape)} for {key}"
        base[key] = (W.float() + delta).to(torch.float32).contiguous()
        merged += 1

    # ---- required checks -------------------------------------------------
    assert merged == 32, f"merged {merged} adapter pairs, expected 32"
    assert not [k for k in base if "lora_" in k], "adapter tensor left in the output"
    q0 = base["model.layers.0.self_attn.q_proj.weight"]
    assert tuple(q0.shape) == (2048, 2048), f"q_proj shape is {tuple(q0.shape)}"
    assert len(base) == 114, f"output has {len(base)} tensors, expected 114"
    assert len(base) == n_base, "tensor count changed against the base"

    split = split_torch_state_dict_into_shards(
        base,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD,
    )
    for fn, keys in split.filename_to_tensors.items():
        size = sum(base[k].numel() * base[k].element_size() for k in keys)
        assert size <= MAX_SHARD or len(keys) == 1, f"{fn} holds {size} bytes in {len(keys)} tensors"

    if os.path.isdir(OUT):
        for f in os.listdir(OUT):
            if f.endswith(".safetensors") or f == "model.safetensors.index.json":
                os.remove(os.path.join(OUT, f))
    os.makedirs(OUT, exist_ok=True)

    for fn, keys in split.filename_to_tensors.items():
        save_file({k: base[k] for k in keys}, os.path.join(OUT, fn), metadata={"format": "pt"})

    written = set()
    for fn in split.filename_to_tensors:
        written |= set(load_file(os.path.join(OUT, fn)))
    assert written == set(base), "written tensors differ from the merged state dict"

    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    for aux in ("config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json"):
        src = os.path.join(BASE, aux)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(OUT, aux))

    print(f"merged {merged} pairs, scale={scale}, {len(base)} tensors, "
          f"{len(split.filename_to_tensors)} shards")


if __name__ == "__main__":
    main()
