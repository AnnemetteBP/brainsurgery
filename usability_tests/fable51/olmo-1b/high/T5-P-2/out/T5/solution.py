"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
LORA_DIR = os.path.join(ROOT, "inputs", "lora")
OUT_DIR = os.path.join(ROOT, "out", "T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # ---- adapter config ----
    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        cfg = json.load(f)
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # ---- base index: names -> shard file, in base order ----
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
        base_index = json.load(f)
    base_map = base_index["weight_map"]
    base_names = list(base_map.keys())
    if len(base_names) != EXPECTED_TENSORS:
        fail(f"base has {len(base_names)} tensors, expected {EXPECTED_TENSORS}")

    # ---- load base tensors (fp32, ~5 GB) ----
    tensors = {}
    for shard in sorted(set(base_map.values())):
        with safe_open(os.path.join(BASE_DIR, shard), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    if set(tensors) != set(base_names):
        fail("base shards do not match the index")

    # ---- load adapter and pair up A/B ----
    lora = {}
    with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k)
    pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
    pairs = {}
    for k in lora:
        m = pat.match(k)
        if not m:
            continue
        base_name = m.group(1) + ".weight"
        b_key = k.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in lora:
            fail(f"missing lora_B for {k}")
        pairs[base_name] = (lora[k], lora[b_key])
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}")
    if 2 * len(pairs) != len(lora):
        fail(f"adapter has {len(lora)} tensors but {len(pairs)} pairs were paired")

    # ---- merge ----
    merged = 0
    for base_name, (A, B) in pairs.items():
        if base_name not in tensors:
            fail(f"adapter target {base_name} not in base")
        W = tensors[base_name]
        if A.dtype != torch.float32 or B.dtype != torch.float32 or W.dtype != torch.float32:
            fail(f"non-float32 tensor involved in merge of {base_name}")
        delta = B.float() @ A.float()  # [out, r] @ [r, in] -> [out, in]
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {base_name}")
        tensors[base_name] = (W + scale * delta).contiguous()
        merged += 1
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} weights, expected {EXPECTED_PAIRS}")

    # ---- required checks before writing ----
    if any("lora_" in k for k in tensors):
        fail("adapter tensor name in output")
    q0 = tensors["model.layers.0.self_attn.q_proj.weight"]
    if tuple(q0.shape) != (2048, 2048):
        fail(f"layers.0 q_proj shape {tuple(q0.shape)}")
    if q0.dtype != torch.float32:
        fail(f"layers.0 q_proj dtype {q0.dtype}")
    if len(tensors) != EXPECTED_TENSORS:
        fail(f"output has {len(tensors)} tensors, expected {EXPECTED_TENSORS}")
    if set(tensors) != set(base_names):
        fail("output key set differs from base")

    # ---- plan shards (greedy, base order) ----
    # A tensor goes alone in its own shard if it exceeds the budget, or if it is so large
    # (more than half the budget: embed_tokens / lm_head at 412 MB) that the task spec
    # says it is stored on its own.
    ALONE_BYTES = MAX_SHARD_BYTES // 2
    def nbytes(t):
        return t.numel() * t.element_size()

    shards = []
    cur, cur_size = [], 0
    for name in base_names:
        sz = nbytes(tensors[name])
        if sz > ALONE_BYTES:
            if cur:
                shards.append(cur)
                cur, cur_size = [], 0
            shards.append([name])
            continue
        if cur_size + sz > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += sz
    if cur:
        shards.append(cur)
    for s in shards:
        tot = sum(nbytes(tensors[n]) for n in s)
        if tot > MAX_SHARD_BYTES and len(s) != 1:
            fail(f"shard exceeds budget: {tot}")

    # ---- write ----
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map = {}
    total_size = 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        chunk = {k: tensors[k].contiguous() for k in names}
        save_file(chunk, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname
            total_size += nbytes(tensors[k])
        print(f"wrote {fname}: {len(names)} tensors, {sum(nbytes(tensors[k]) for k in names)} bytes")
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    # ---- post-write verification ----
    seen = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt") as f:
            for k in f.keys():
                seen[k] = f.get_slice(k).get_shape()
    if len(seen) != EXPECTED_TENSORS or set(seen) != set(base_names):
        fail("written checkpoint key set mismatch")
    if any("lora_" in k for k in seen):
        fail("written checkpoint contains lora_ tensors")
    print(f"OK: {len(seen)} tensors in {n} shards, {merged} weights merged, scale={scale}")


if __name__ == "__main__":
    main()
