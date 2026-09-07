"""T5: fold a PEFT LoRA adapter into Pythia-1B base weights, write sharded safetensors."""
import json
import os

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
CONFIG = "inputs/lora/adapter_config.json"
OUT = "out/T5"
MAX_SHARD = 512 * 1024 * 1024

base = load_file(BASE)
adapter = load_file(ADAPTER)
cfg = json.load(open(CONFIG))
scale = cfg["lora_alpha"] / cfg["r"]
assert not cfg["fan_in_fan_out"], "fan_in_fan_out=True not handled"

# collect adapter pairs, mapping PEFT names to base names
pairs = {}
for name in adapter:
    if ".lora_A.weight" not in name and ".lora_B.weight" not in name:
        raise SystemExit(f"unexpected adapter tensor: {name}")
    stem, which = name.rsplit(".lora_", 1)
    stem = stem.removeprefix("base_model.model.")
    pairs.setdefault(stem, {})[which[0]] = adapter[name]

n_merged = 0
for stem, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        raise SystemExit(f"incomplete adapter pair for {stem}")
    target = stem + ".weight"
    if target not in base:
        raise SystemExit(f"no base tensor for adapter {stem}")
    W = base[target]
    A, B = ab["A"].float(), ab["B"].float()
    if B.shape[1] != A.shape[0]:
        raise SystemExit(f"rank mismatch at {stem}: {B.shape} @ {A.shape}")
    delta = scale * (B @ A)
    if delta.shape != W.shape:
        raise SystemExit(f"delta shape {tuple(delta.shape)} != base {tuple(W.shape)} at {stem}")
    base[target] = (W.float() + delta).to(W.dtype)
    n_merged += 1

# required checks, before writing anything
assert n_merged == 16, f"merged {n_merged} adapter pairs, expected 16"
assert not [k for k in base if "lora_" in k], "adapter tensors leaked into output"
probe = "gpt_neox.layers.0.attention.query_key_value.weight"
assert tuple(base[probe].shape) == (6144, 2048), f"{probe} has shape {tuple(base[probe].shape)}"
assert len(base) == 244, f"output has {len(base)} tensors, expected 244"

os.makedirs(OUT, exist_ok=True)
split = split_torch_state_dict_into_shards(
    base, filename_pattern="model{suffix}.safetensors", max_shard_size=MAX_SHARD
)
for filename, keys in split.filename_to_tensors.items():
    save_file(
        {k: base[k].contiguous() for k in keys},
        os.path.join(OUT, filename),
        metadata={"format": "pt"},
    )
index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)

# post-write verification: every name present exactly once, shard budget respected
written = {}
for filename in split.filename_to_tensors:
    shard = load_file(os.path.join(OUT, filename))
    total = sum(t.numel() * t.element_size() for t in shard.values())
    assert total <= MAX_SHARD or len(shard) == 1, f"{filename}: {total} bytes over budget"
    for k, v in shard.items():
        assert k not in written, f"duplicate tensor {k}"
        assert torch.equal(v, base[k]), f"mismatch for {k}"
        written[k] = filename
assert written == index["weight_map"], "index weight_map disagrees with the shards"
assert len(written) == 244, f"wrote {len(written)} tensors"
print(f"merged {n_merged} pairs, wrote {len(written)} tensors in {len(split.filename_to_tensors)} shards")
