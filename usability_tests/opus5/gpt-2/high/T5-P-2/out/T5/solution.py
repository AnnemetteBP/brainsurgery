"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint.

Reads inputs/base/model.safetensors and inputs/lora/{adapter_config.json,
adapter_model.safetensors}, folds the adapter in ("merge and unload") and writes a
sharded safetensors checkpoint plus an index to out/T5/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent          # .../out/T5
ROOT = HERE.parent.parent                        # sandbox root
BASE_PATH = ROOT / "inputs" / "base" / "model.safetensors"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = HERE
SHARD_LIMIT = 100 * 1024 * 1024                  # 104_857_600 bytes of tensor data
PEFT_PREFIX = "base_model.model."
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160


def die(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


# ---------------------------------------------------------------- load inputs
cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
r = cfg["r"]
alpha = cfg["lora_alpha"]
fan_in_fan_out = cfg["fan_in_fan_out"]
if not r:
    die("adapter_config.json has r == 0")
scale = alpha / r
print(f"adapter: r={r} lora_alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")
if fan_in_fan_out is not True:
    die(f"expected fan_in_fan_out=true (Conv1D [in, out] base layout), got {fan_in_fan_out!r}")

base = load_file(str(BASE_PATH))
adapter = load_file(str(LORA_DIR / "adapter_model.safetensors"))
print(f"loaded base: {len(base)} tensors; adapter: {len(adapter)} tensors")

if len(base) != EXPECTED_TENSORS:
    die(f"base has {len(base)} tensors, expected {EXPECTED_TENSORS}")

# ------------------------------------------------------------- pair up A / B
# base_model.model.h.<i>.<module>.lora_{A,B}.weight  ->  h.<i>.<module>.weight
pat = re.compile(r"^(?P<target>.+)\.lora_(?P<which>[AB])\.weight$")
pairs: dict[str, dict[str, torch.Tensor]] = {}
for name, tensor in adapter.items():
    if not name.startswith(PEFT_PREFIX):
        die(f"adapter tensor {name!r} does not start with {PEFT_PREFIX!r}")
    stripped = name[len(PEFT_PREFIX):]
    m = pat.match(stripped)
    if m is None:
        die(f"adapter tensor {name!r} is not a lora_A/lora_B weight")
    target = m.group("target") + ".weight"
    slot = pairs.setdefault(target, {})
    if m.group("which") in slot:
        die(f"duplicate lora_{m.group('which')} for target {target!r}")
    slot[m.group("which")] = tensor

for target, slot in pairs.items():
    if set(slot) != {"A", "B"}:
        die(f"target {target!r} has an incomplete adapter pair: {sorted(slot)}")

if len(pairs) != EXPECTED_PAIRS:
    die(f"found {len(pairs)} adapter pairs, expected exactly {EXPECTED_PAIRS}")
print(f"check ok: exactly {len(pairs)} complete adapter pairs")

# ------------------------------------------------------------------- merging
merged = dict(base)
for target in sorted(pairs, key=lambda k: (len(k), k)):
    if target not in merged:
        die(f"adapter targets {target!r}, which is not in the base checkpoint")
    w = merged[target]
    a = pairs[target]["A"]
    b = pairs[target]["B"]
    if a.shape[0] != r or b.shape[1] != r:
        die(f"{target}: adapter rank mismatch, A{tuple(a.shape)} B{tuple(b.shape)} vs r={r}")
    if w.dtype != torch.float32 or a.dtype != torch.float32 or b.dtype != torch.float32:
        die(f"{target}: expected float32 everywhere, got "
            f"{w.dtype}/{a.dtype}/{b.dtype}")
    # B @ A is [out, in]; the Conv1D base weight is [in, out], hence the transpose.
    delta = (b.to(torch.float32) @ a.to(torch.float32)).T.contiguous() * scale
    if delta.shape != w.shape:
        die(f"{target}: delta {tuple(delta.shape)} does not match base {tuple(w.shape)}")
    merged[target] = (w.to(torch.float32) + delta).contiguous()
    print(f"merged {target}: {tuple(w.shape)} += {scale} * (B @ A).T")

# ------------------------------------------------------------ required checks
lora_names = [k for k in merged if "lora_" in k]
if lora_names:
    die(f"{len(lora_names)} adapter tensor name(s) leaked into the output: {lora_names[:3]}")
print("check ok: no tensor name contains 'lora_'")

probe = "h.0.attn.c_attn.weight"
if probe not in merged:
    die(f"{probe} missing from the output")
if tuple(merged[probe].shape) != (768, 2304):
    die(f"{probe} has shape {tuple(merged[probe].shape)}, expected (768, 2304)")
print(f"check ok: {probe} still has shape (768, 2304)")

if len(merged) != EXPECTED_TENSORS:
    die(f"output has {len(merged)} tensors, expected {EXPECTED_TENSORS}")
if set(merged) != set(base):
    die("output key set differs from the base key set")
print(f"check ok: output has exactly {EXPECTED_TENSORS} tensors, same names as the base")

for name, tensor in merged.items():
    if tensor.dtype != base[name].dtype or tensor.shape != base[name].shape:
        die(f"{name}: shape/dtype drifted from the base")

# ------------------------------------------------------------------ sharding
def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


shards: list[list[str]] = []
current: list[str] = []
current_bytes = 0
for name in sorted(merged):
    size = nbytes(merged[name])
    if size > SHARD_LIMIT:
        # too big for any shard: give it one of its own
        if current:
            shards.append(current)
            current, current_bytes = [], 0
        shards.append([name])
        continue
    if current and current_bytes + size > SHARD_LIMIT:
        shards.append(current)
        current, current_bytes = [], 0
    current.append(name)
    current_bytes += size
if current:
    shards.append(current)

n = len(shards)
names = [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]

for stale in OUT_DIR.glob("*.safetensors"):
    stale.unlink()
stale_index = OUT_DIR / "model.safetensors.index.json"
if stale_index.exists():
    stale_index.unlink()

weight_map: dict[str, str] = {}
total_size = 0
for filename, keys in zip(names, shards):
    payload = sum(nbytes(merged[k]) for k in keys)
    if payload > SHARD_LIMIT and len(keys) > 1:
        die(f"{filename}: {payload} bytes over the {SHARD_LIMIT} byte budget")
    save_file({k: merged[k] for k in keys}, str(OUT_DIR / filename),
              metadata={"format": "pt"})
    for k in keys:
        weight_map[k] = filename
    total_size += payload
    print(f"wrote {filename}: {len(keys)} tensors, {payload} bytes")

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
stale_index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
print(f"wrote model.safetensors.index.json: {len(weight_map)} entries, "
      f"total_size={total_size}")

# ------------------------------------------------------- verify what was written
seen: dict[str, str] = {}
for filename in names:
    path = OUT_DIR / filename
    payload = 0
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            payload += nbytes(t)
            if k in seen:
                die(f"{k} appears in both {seen[k]} and {filename}")
            seen[k] = filename
            if t.shape != base[k].shape or t.dtype != base[k].dtype:
                die(f"{k} in {filename}: shape/dtype mismatch after write")
            if not torch.equal(t, merged[k]):
                die(f"{k} in {filename}: values differ from what was merged")
    if payload > SHARD_LIMIT and len(keys) > 1:
        die(f"{filename} holds {payload} bytes in {len(keys)} tensors, over budget")
    if payload > SHARD_LIMIT:
        print(f"  {filename}: single oversized tensor {keys[0]} ({payload} bytes), alone")

if seen != weight_map:
    die("index weight_map does not match the tensors actually present in the shards")
if len(seen) != EXPECTED_TENSORS:
    die(f"shards hold {len(seen)} tensors, expected {EXPECTED_TENSORS}")
if any("lora_" in k for k in seen):
    die("an adapter tensor was written to the output")

unchanged = [k for k in base if k not in pairs]
for k in unchanged:
    if not torch.equal(merged[k], base[k]):
        die(f"{k} was modified but should be untouched")
print(f"check ok: {len(unchanged)} unchanged tensors are bit-identical to the base")

print(f"OK: {n} shards + index in {OUT_DIR}, {len(seen)} tensors, "
      f"{len(pairs)} merged weights")
