"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and export sharded.

Standalone PyTorch + safetensors script. Reads inputs/base/model.safetensors and
inputs/lora/*, writes out/T5/model-XXXXX-of-XXXXX.safetensors plus
out/T5/model.safetensors.index.json.
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_PATH = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA_PATH = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
LORA_CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
EXPECTED_TENSORS = 244
EXPECTED_PAIRS = 16
PEFT_PREFIX = "base_model.model."


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    with open(LORA_CFG) as f:
        cfg = json.load(f)
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # Load base in file order (keys() is sorted by safetensors; keep that order).
    base: dict[str, torch.Tensor] = {}
    with safe_open(BASE_PATH, framework="pt", device="cpu") as f:
        base_keys = list(f.keys())
        for k in base_keys:
            base[k] = f.get_tensor(k)
    print(f"base tensors: {len(base)}")
    if len(base) != EXPECTED_TENSORS:
        fail(f"base has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    # Load adapter and pair A/B by module.
    lora: dict[str, torch.Tensor] = {}
    with safe_open(LORA_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k)
    print(f"adapter tensors: {len(lora)}")

    pat = re.compile(r"^(.*)\.lora_(A|B)\.weight$")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        m = pat.match(k)
        if m is None:
            fail(f"unrecognized adapter tensor name: {k}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = v

    merged = 0
    for mod, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            fail(f"incomplete LoRA pair for {mod}: {sorted(ab)}")
        if not mod.startswith(PEFT_PREFIX):
            fail(f"adapter module {mod} lacks PEFT prefix {PEFT_PREFIX!r}")
        base_name = mod[len(PEFT_PREFIX):] + ".weight"
        if base_name not in base:
            fail(f"adapter target {base_name} not found in base")
        A = ab["A"].to(torch.float32)
        B = ab["B"].to(torch.float32)
        W = base[base_name]
        if A.shape != (r, W.shape[1]) or B.shape != (W.shape[0], r):
            fail(f"shape mismatch for {base_name}: W={tuple(W.shape)} "
                 f"A={tuple(A.shape)} B={tuple(B.shape)} r={r}")
        delta = scale * (B @ A)  # [out, in], float32
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != W shape {tuple(W.shape)} for {base_name}")
        new_W = (W.to(torch.float32) + delta).to(W.dtype).contiguous()
        if new_W.dtype != W.dtype or new_W.shape != W.shape:
            fail(f"merged tensor {base_name} changed shape/dtype")
        base[base_name] = new_W
        merged += 1
    print(f"merged pairs: {merged}")

    # Required checks, before writing.
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
    bad = [k for k in base if "lora_" in k]
    if bad:
        fail(f"output contains adapter names: {bad[:5]}")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if probe not in base or tuple(base[probe].shape) != (6144, 2048):
        fail(f"{probe} missing or wrong shape: "
             f"{tuple(base[probe].shape) if probe in base else None}")
    if base[probe].dtype != torch.float16:
        fail(f"{probe} dtype {base[probe].dtype}, expected float16")
    if len(base) != EXPECTED_TENSORS:
        fail(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")
    if set(base) != set(base_keys):
        fail("output key set differs from base key set")

    # Shard: greedy in key order, 512 MiB of tensor data per shard.
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    cur: list[str] = []
    cur_size = 0
    for k in base_keys:
        sz = nbytes(base[k])
        if cur and cur_size + sz > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(k)
        cur_size += sz
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        shard_bytes = sum(nbytes(base[k]) for k in keys)
        if shard_bytes > MAX_SHARD_BYTES and len(keys) > 1:
            fail(f"shard {fname} exceeds budget with {len(keys)} tensors")
        total_size += shard_bytes
        for k in keys:
            weight_map[k] = fname

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {k: base[k].contiguous() for k in keys}
        save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        print(f"wrote {fname}: {len(keys)} tensors, "
              f"{sum(nbytes(t) for t in tensors.values()):,} bytes")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    # Post-write verification.
    seen = 0
    for i in range(1, n + 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt", device="cpu") as f:
            ks = list(f.keys())
            seen += len(ks)
            for k in ks:
                if weight_map[k] != fname:
                    fail(f"index mismatch for {k}")
    if seen != EXPECTED_TENSORS or len(weight_map) != EXPECTED_TENSORS:
        fail(f"wrote {seen} tensors / index has {len(weight_map)}, expected {EXPECTED_TENSORS}")
    print(f"OK: {seen} tensors in {n} shards, total_size={total_size:,}")


if __name__ == "__main__":
    main()
