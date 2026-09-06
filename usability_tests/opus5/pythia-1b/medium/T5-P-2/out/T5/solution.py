"""T5: merge a PEFT LoRA adapter into a Pythia-1B checkpoint and write it sharded."""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
CONFIG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
PREFIX = "base_model.model."

EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244
PROBE = "gpt_neox.layers.0.attention.query_key_value.weight"
PROBE_SHAPE = (6144, 2048)


def load(path):
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main():
    with open(CONFIG) as f:
        cfg = json.load(f)
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    scale = alpha / r

    base = load(BASE)
    adapter = load(ADAPTER)

    # Collect A/B pairs by their base tensor name.
    pairs = {}
    for name, t in adapter.items():
        if ".lora_A.weight" in name:
            stem, kind = name.rsplit(".lora_A.weight", 1)[0], "A"
        elif ".lora_B.weight" in name:
            stem, kind = name.rsplit(".lora_B.weight", 1)[0], "B"
        else:
            raise SystemExit(f"unrecognised adapter tensor: {name}")
        if not stem.startswith(PREFIX):
            raise SystemExit(f"adapter tensor without {PREFIX!r} prefix: {name}")
        target = stem[len(PREFIX):] + ".weight"
        pairs.setdefault(target, {})[kind] = t

    for target, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"incomplete LoRA pair for {target}: have {sorted(ab)}")
        if target not in base:
            raise SystemExit(f"adapter targets a tensor absent from the base: {target}")

    if len(pairs) != EXPECTED_PAIRS:
        raise SystemExit(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")

    # Merge: W += scale * B @ A, in float32, cast back to the base dtype.
    for target, ab in sorted(pairs.items()):
        w = base[target]
        a, b = ab["A"].float(), ab["B"].float()
        if a.shape[0] != r or b.shape[1] != r:
            raise SystemExit(f"{target}: factor ranks {a.shape}/{b.shape} disagree with r={r}")
        delta = scale * (b @ a)
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != w.shape:
            raise SystemExit(f"{target}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        base[target] = (w.float() + delta).to(w.dtype)

    # Required checks, before writing anything.
    leaked = [k for k in base if "lora_" in k]
    if leaked:
        raise SystemExit(f"adapter tensors leaked into the output: {leaked[:5]}")
    if tuple(base[PROBE].shape) != PROBE_SHAPE:
        raise SystemExit(f"{PROBE} has shape {tuple(base[PROBE].shape)}, expected {PROBE_SHAPE}")
    if len(base) != EXPECTED_TENSORS:
        raise SystemExit(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    # Greedy sharding in key order; an oversized tensor goes alone in its shard.
    names = sorted(base)
    sizes = {k: base[k].numel() * base[k].element_size() for k in names}
    shards, cur, cur_bytes = [], [], 0
    for k in names:
        n = sizes[k]
        if n > MAX_SHARD_BYTES:
            if cur:
                shards.append(cur)
            shards.append([k])
            cur, cur_bytes = [], 0
            continue
        if cur and cur_bytes + n > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += n
    if cur:
        shards.append(cur)

    total = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    weight_map = {}
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        shard_bytes = sum(sizes[k] for k in keys)
        if shard_bytes > MAX_SHARD_BYTES and len(keys) > 1:
            raise SystemExit(f"{fname} holds {shard_bytes} bytes, over the budget")
        save_file(
            {k: base[k].contiguous() for k in keys},
            os.path.join(OUT_DIR, fname),
            metadata={"format": "pt"},
        )
        for k in keys:
            weight_map[k] = fname

    if len(weight_map) != EXPECTED_TENSORS:
        raise SystemExit(f"weight_map covers {len(weight_map)} tensors, expected {EXPECTED_TENSORS}")
    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"merged {len(pairs)} LoRA pairs (scale={scale}), wrote {len(weight_map)} tensors "
          f"in {total} shards to {OUT_DIR}/")


if __name__ == "__main__":
    main()
