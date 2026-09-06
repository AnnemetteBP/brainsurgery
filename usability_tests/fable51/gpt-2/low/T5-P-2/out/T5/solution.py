"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
CONFIG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
MAX_SHARD = 104_857_600  # 100 MiB of tensor data
PREFIX = "base_model.model."


def fail(msg):
    print("CHECK FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


def main():
    cfg = json.load(open(CONFIG))
    scale = cfg["lora_alpha"] / cfg["r"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))

    base = load_file(BASE)
    adapter = load_file(ADAPTER)
    n_base = len(base)

    pat = re.compile(r"^(.*)\.lora_A\.weight$")
    merged = 0
    for a_name, A in adapter.items():
        m = pat.match(a_name)
        if not m:
            continue
        stem = m.group(1)
        b_name = stem + ".lora_B.weight"
        if b_name not in adapter:
            fail(f"missing lora_B for {a_name}")
        if not stem.startswith(PREFIX):
            fail(f"unexpected adapter prefix in {a_name}")
        target = stem[len(PREFIX):] + ".weight"
        if target not in base:
            fail(f"target {target} not in base")
        B = adapter[b_name]
        delta = (B.float() @ A.float()) * scale  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]
        W = base[target]
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != base {tuple(W.shape)} for {target}")
        base[target] = (W.float() + delta).to(W.dtype).contiguous()
        merged += 1

    # Required checks
    if merged != 12:
        fail(f"expected 12 merged adapter pairs, got {merged}")
    if any("lora_" in k for k in base):
        fail("lora_ tensor present in output")
    if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        fail(f"bad c_attn shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
    if len(base) != 160 or len(base) != n_base:
        fail(f"expected 160 output tensors, got {len(base)}")

    # Shard: greedy by insertion order, tensor data bytes only.
    shards, cur, cur_size = [], {}, 0
    for name, t in base.items():
        nbytes = t.numel() * t.element_size()
        if cur and cur_size + nbytes > MAX_SHARD:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[name] = t
        cur_size += nbytes
    if cur:
        shards.append(cur)

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map = {}
    total = 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k, t in shard.items():
            weight_map[k] = fname
            total += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"merged {merged} pairs, wrote {len(weight_map)} tensors in {n} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
