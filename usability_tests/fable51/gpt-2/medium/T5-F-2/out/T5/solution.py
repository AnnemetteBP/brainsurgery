"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded safetensors checkpoint.

Plain script on top of safetensors + torch (no model instantiation).
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA_DIR = os.path.join(ROOT, "inputs", "lora")
OUT = os.path.join(ROOT, "out", "T5")
MAX_SHARD = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    cfg = json.load(open(os.path.join(LORA_DIR, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    scale = alpha / r

    with safe_open(BASE, "pt") as f:
        base = {k: f.get_tensor(k) for k in f.keys()}
    with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), "pt") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}

    n_base = len(base)
    pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
    merged = 0
    for a_name, A in lora.items():
        m = pat.match(a_name)
        if not m:
            continue
        module = m.group(1)
        b_name = f"base_model.model.{module}.lora_B.weight"
        target = f"{module}.weight"
        if b_name not in lora:
            fail(f"missing lora_B for {a_name}")
        if target not in base:
            fail(f"adapter target {target} not in base")
        B = lora[b_name]
        W = base[target]
        if A.dtype != torch.float32 or B.dtype != torch.float32 or W.dtype != torch.float32:
            fail(f"non-float32 tensor in pair {target}")
        delta = scale * (B @ A)  # [out, in] (nn.Linear convention)
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != base {tuple(W.shape)} for {target}")
        base[target] = (W + delta).contiguous()
        merged += 1

    # ---- required checks (before writing) ----
    if merged != 12:
        fail(f"expected 12 merged adapter pairs, got {merged}")
    if 2 * merged != len(lora):
        fail(f"adapter has {len(lora)} tensors, expected {2 * merged}")
    if any("lora_" in k for k in base):
        fail("lora_ tensor name in output")
    if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        fail(f"h.0.attn.c_attn.weight shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
    if base["h.0.attn.c_attn.weight"].dtype != torch.float32:
        fail("merged weight is not float32")
    if len(base) != 160 or len(base) != n_base:
        fail(f"output has {len(base)} tensors, expected 160")

    # ---- shard: greedy in base key order, oversized tensors alone ----
    shards, cur, cur_bytes = [], {}, 0
    for k, t in base.items():
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > MAX_SHARD:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = t
        cur_bytes += nbytes
    if cur:
        shards.append(cur)
    for s in shards:
        total = sum(t.numel() * t.element_size() for t in s.values())
        if total > MAX_SHARD and len(s) != 1:
            fail("shard over budget with more than one tensor")

    os.makedirs(OUT, exist_ok=True)
    n = len(shards)
    weight_map, total_size = {}, 0
    for i, s in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(s, os.path.join(OUT, fname), metadata={"format": "pt"})
        for k, t in s.items():
            weight_map[k] = fname
            total_size += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"merged {merged} pairs, wrote {n} shards, {len(weight_map)} tensors to {OUT}")


if __name__ == "__main__":
    main()
