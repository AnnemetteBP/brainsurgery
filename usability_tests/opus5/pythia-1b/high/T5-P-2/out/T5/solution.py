#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into a Pythia-1B checkpoint and export it sharded.

Reads  inputs/base/model.safetensors  and  inputs/lora/{adapter_model.safetensors,adapter_config.json}
Writes out/T5/model-0000X-of-0000N.safetensors + out/T5/model.safetensors.index.json
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent          # out/T5
ROOT = HERE.parent.parent                       # sandbox root
BASE_PATH = ROOT / "inputs" / "base" / "model.safetensors"
LORA_PATH = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CFG = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = HERE

MAX_SHARD_BYTES = 512 * 1024 * 1024             # 536,870,912
EXPECTED_TENSORS = 244
EXPECTED_PAIRS = 16
PEFT_PREFIX = "base_model.model."


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# ---------------------------------------------------------------- load inputs
for p in (BASE_PATH, LORA_PATH, LORA_CFG):
    check(p.exists(), f"missing input {p}")

cfg = json.loads(LORA_CFG.read_text())
r = int(cfg["r"])
alpha = float(cfg["lora_alpha"])
fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
check(r > 0, f"adapter_config r must be positive, got {r}")
scale = alpha / r
print(f"adapter: r={r} lora_alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")
check(abs(scale - 2.0) < 1e-12, f"expected scale alpha/r == 2, got {scale}")

base = load_file(str(BASE_PATH))
lora = load_file(str(LORA_PATH))
print(f"loaded base: {len(base)} tensors, adapter: {len(lora)} tensors")

base_names = set(base)
base_dtypes = {k: v.dtype for k, v in base.items()}
base_shapes = {k: tuple(v.shape) for k, v in base.items()}
check(len(base) == EXPECTED_TENSORS, f"base has {len(base)} tensors, expected {EXPECTED_TENSORS}")

# --------------------------------------------------- pair up the lora factors
# PEFT names: base_model.model.<base name without .weight>.lora_{A,B}.weight
pairs: dict[str, dict[str, torch.Tensor]] = {}
for name, tensor in lora.items():
    check(name.startswith(PEFT_PREFIX), f"adapter tensor {name!r} lacks prefix {PEFT_PREFIX!r}")
    stem = name[len(PEFT_PREFIX):]
    if stem.endswith(".lora_A.weight"):
        side, target = "A", stem[: -len(".lora_A.weight")] + ".weight"
    elif stem.endswith(".lora_B.weight"):
        side, target = "B", stem[: -len(".lora_B.weight")] + ".weight"
    else:
        fail(f"unrecognised adapter tensor name {name!r}")
    check(target not in pairs or side not in pairs[target], f"duplicate lora_{side} for {target}")
    pairs.setdefault(target, {})[side] = tensor

complete = {t: d for t, d in pairs.items() if set(d) == {"A", "B"}}
incomplete = sorted(set(pairs) - set(complete))
check(not incomplete, f"adapter has factors without a matching pair: {incomplete}")

# REQUIRED CHECK 1: exactly 16 adapter pairs.
check(
    len(complete) == EXPECTED_PAIRS,
    f"found {len(complete)} complete adapter pairs, expected {EXPECTED_PAIRS}",
)

# ------------------------------------------------------------------ the merge
merged: list[str] = []
for target in sorted(complete):
    check(target in base_names, f"adapter targets {target!r} which is not in the base checkpoint")
    A = complete[target]["A"]
    B = complete[target]["B"]
    check(A.ndim == 2 and B.ndim == 2, f"{target}: lora factors must be 2-D")
    check(
        A.shape[0] == r and B.shape[1] == r,
        f"{target}: factor ranks {tuple(B.shape)} @ {tuple(A.shape)} disagree with r={r}",
    )

    W = base[target]
    orig_dtype, orig_shape = W.dtype, W.shape

    # float32 throughout: B @ A is [out, in]; with fan_in_fan_out the base is
    # stored [in, out] (Conv1D-style) and the delta has to be transposed.
    delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
    if fan_in_fan_out:
        delta = delta.T
    check(
        delta.shape == orig_shape,
        f"{target}: delta {tuple(delta.shape)} does not match base {tuple(orig_shape)}",
    )

    base[target] = (W.to(torch.float32) + delta).to(orig_dtype)
    check(base[target].shape == orig_shape, f"{target}: shape changed by the merge")
    check(base[target].dtype == orig_dtype, f"{target}: dtype changed by the merge")
    check(torch.isfinite(base[target]).all(), f"{target}: merge produced non-finite values")
    merged.append(target)

print(f"merged {len(merged)} tensors, e.g. {merged[0]}")
check(len(merged) == EXPECTED_PAIRS, f"merged {len(merged)} tensors, expected {EXPECTED_PAIRS}")

# ----------------------------------------------------- pre-write sanity gates
# REQUIRED CHECK 2: nothing adapter-shaped leaks into the output.
leaked = sorted(k for k in base if "lora_" in k)
check(not leaked, f"adapter tensors present in the output state dict: {leaked}")

# REQUIRED CHECK 3: the reference qkv weight keeps its shape.
probe = "gpt_neox.layers.0.attention.query_key_value.weight"
check(probe in base, f"{probe} missing from the output state dict")
check(
    tuple(base[probe].shape) == (6144, 2048),
    f"{probe} has shape {tuple(base[probe].shape)}, expected (6144, 2048)",
)
check(base[probe].dtype == torch.float16, f"{probe} has dtype {base[probe].dtype}, expected float16")

# REQUIRED CHECK 4: exactly 244 tensors, and the same names as the base.
check(len(base) == EXPECTED_TENSORS, f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")
check(set(base) == base_names, "output key set differs from the base key set")
for k, t in base.items():
    check(tuple(t.shape) == base_shapes[k], f"{k}: shape changed ({tuple(t.shape)})")
    check(t.dtype == base_dtypes[k], f"{k}: dtype changed ({t.dtype})")

# --------------------------------------------------------------- shard layout
# Greedy fill in a deterministic (sorted) order: a tensor bigger than the limit
# gets a shard to itself, otherwise a tensor that would overflow the current
# shard starts a new one.
order = sorted(base)
shards: list[list[str]] = []
cur: list[str] = []
cur_size = 0
for name in order:
    size = nbytes(base[name])
    if size > MAX_SHARD_BYTES:
        if cur:
            shards.append(cur)
            cur, cur_size = [], 0
        shards.append([name])
        continue
    if cur and cur_size + size > MAX_SHARD_BYTES:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += size
if cur:
    shards.append(cur)

n_shards = len(shards)
check(n_shards >= 1, "no shards were produced")
shard_names = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
total_size = sum(nbytes(t) for t in base.values())
for names, fname in zip(shards, shard_names):
    payload = sum(nbytes(base[n]) for n in names)
    check(names, f"{fname} would be empty")
    check(
        payload <= MAX_SHARD_BYTES or len(names) == 1,
        f"{fname} holds {payload} bytes across {len(names)} tensors, over the {MAX_SHARD_BYTES} limit",
    )
    print(f"{fname}: {len(names)} tensors, {payload} bytes")

weight_map = {n: fname for names, fname in zip(shards, shard_names) for n in names}
check(len(weight_map) == EXPECTED_TENSORS, f"weight_map covers {len(weight_map)} tensors")
check(set(weight_map) == base_names, "weight_map key set differs from the base key set")

# --------------------------------------------------------------------- write
if OUT_DIR.exists():
    for stale in OUT_DIR.iterdir():
        if stale.name in {"solution.py", "REPORT.md"}:
            continue
        (shutil.rmtree if stale.is_dir() else os.remove)(stale)
OUT_DIR.mkdir(parents=True, exist_ok=True)

for names, fname in zip(shards, shard_names):
    shard = {n: base[n].contiguous() for n in names}
    save_file(shard, str(OUT_DIR / fname), metadata={"format": "pt"})

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
(OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"wrote {n_shards} shards + index, total_size={total_size}")

# -------------------------------------------------------- post-write verify
del base, lora, complete, pairs

index_back = json.loads((OUT_DIR / "model.safetensors.index.json").read_text())
wmap = index_back["weight_map"]
check(set(wmap.values()) == set(shard_names), "index references shard files that were not written")

seen: set[str] = set()
with safe_open(str(BASE_PATH), framework="pt") as src:
    for fname in shard_names:
        path = OUT_DIR / fname
        check(path.exists(), f"{fname} was not written")
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            payload = 0
            for k in keys:
                out_t = f.get_tensor(k)
                payload += nbytes(out_t)
                check(wmap.get(k) == fname, f"{k} is in {fname} but the index says {wmap.get(k)}")
                check(k not in seen, f"{k} appears in more than one shard")
                seen.add(k)
                check("lora_" not in k, f"adapter tensor {k} present in {fname}")
                check(tuple(out_t.shape) == base_shapes[k], f"{k}: shape mismatch on reload")
                check(out_t.dtype == base_dtypes[k], f"{k}: dtype mismatch on reload")
                src_t = src.get_tensor(k)
                if k in merged:
                    ref = src_t.to(torch.float32)
                    err = torch.linalg.norm(out_t.to(torch.float32) - ref) / torch.linalg.norm(ref)
                    check(err > 0, f"{k} was supposed to change but is identical to the base")
                else:
                    check(torch.equal(out_t, src_t), f"{k}: unchanged tensor differs from the base")
            check(
                payload <= MAX_SHARD_BYTES or len(keys) == 1,
                f"{fname} holds {payload} bytes, over the {MAX_SHARD_BYTES} limit",
            )

check(len(seen) == EXPECTED_TENSORS, f"shards hold {len(seen)} tensors, expected {EXPECTED_TENSORS}")
check(seen == base_names, "reloaded key set differs from the base key set")
check(seen == set(wmap), "reloaded key set differs from the index weight_map")
check(
    sorted(p.name for p in OUT_DIR.glob("*.safetensors")) == sorted(shard_names),
    "unexpected safetensors files in the output directory",
)

print(
    f"OK: {len(seen)} tensors in {n_shards} shards, {len(merged)} merged, "
    f"{len(seen) - len(merged)} bit-identical to the base"
)
