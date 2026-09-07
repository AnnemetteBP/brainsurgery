"""T5: fold a PEFT LoRA adapter into GPT-2 base weights and export sharded.

Direct checkpoint surgery: read base + adapter with safetensors, merge, write
sharded safetensors plus an index. Deliberately avoids instantiating a
transformers model (the base key set has no `transformer.` prefix and keeps the
`attn.bias` causal-mask buffers, which a save_pretrained round-trip would not
reproduce exactly).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("inputs/base/model.safetensors")
LORA = Path("inputs/lora/adapter_model.safetensors")
LORA_CFG = Path("inputs/lora/adapter_config.json")
OUT = Path("out/T5")

SHARD_BUDGET = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data
ADAPTER_RE = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def main() -> None:
    cfg = json.loads(LORA_CFG.read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if r <= 0:
        fail(f"adapter r must be positive, got {r}")
    scale = alpha / r
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    with safe_open(BASE, framework="pt") as bf:
        base_keys = list(bf.keys())
        state = {k: bf.get_tensor(k) for k in base_keys}
    with safe_open(LORA, framework="pt") as lf:
        lora = {k: lf.get_tensor(k) for k in lf.keys()}

    n_base_in = len(state)
    print(f"read {n_base_in} base tensors, {len(lora)} adapter tensors")

    # --- pair up the adapter factors -------------------------------------
    pairs = []
    for key in sorted(lora):
        m = ADAPTER_RE.match(key)
        if not m:
            continue
        target = f"{m.group(1)}.weight"
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in lora:
            fail(f"adapter {key} has no matching lora_B tensor")
        if target not in state:
            fail(f"adapter target {target} is not a base tensor")
        pairs.append((target, key, b_key))

    unpaired = set(lora) - {k for _, k, _ in pairs} - {k for _, _, k in pairs}
    if unpaired:
        fail(f"adapter tensors not consumed by any pair: {sorted(unpaired)}")
    if len(pairs) != 12:
        fail(f"expected exactly 12 adapter pairs, found {len(pairs)}")
    if len({t for t, _, _ in pairs}) != len(pairs):
        fail("two adapter pairs target the same base tensor")
    print(f"check ok: {len(pairs)} adapter pairs")

    # --- merge ------------------------------------------------------------
    for target, a_key, b_key in pairs:
        A = lora[a_key].to(torch.float32)   # [r, in]
        B = lora[b_key].to(torch.float32)   # [out, r]
        W = state[target]
        if W.dtype != torch.float32:
            fail(f"{target} is {W.dtype}, expected float32")
        if A.shape[0] != r or B.shape[1] != r:
            fail(f"{a_key}/{b_key} ranks {A.shape}/{B.shape} disagree with r={r}")
        delta = scale * (B @ A)             # [out, in]
        if fan_in_fan_out:
            delta = delta.T                # [in, out], matching Conv1D layout
        if delta.shape != W.shape:
            fail(f"{target}: delta {tuple(delta.shape)} != weight {tuple(W.shape)}")
        state[target] = (W.to(torch.float32) + delta).contiguous()

    # --- post-merge checks ------------------------------------------------
    leaked = [k for k in state if "lora_" in k]
    if leaked:
        fail(f"adapter/intermediate tensor names in output: {leaked}")
    probe = "h.0.attn.c_attn.weight"
    if tuple(state[probe].shape) != (768, 2304):
        fail(f"{probe} shape is {tuple(state[probe].shape)}, expected (768, 2304)")
    if state[probe].dtype != torch.float32:
        fail(f"{probe} dtype is {state[probe].dtype}, expected float32")
    if len(state) != 160:
        fail(f"output has {len(state)} tensors, expected 160")
    if set(state) != set(base_keys):
        fail("output key set differs from the base key set")
    print("check ok: 160 tensors, no lora_ names, c_attn shape and dtype intact")

    # --- shard ------------------------------------------------------------
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    cur: list[str] = []
    cur_size = 0
    for k in base_keys:
        size = nbytes(state[k])
        if size > SHARD_BUDGET:
            if cur:
                shards.append(cur)
                cur, cur_size = [], 0
            shards.append([k])  # oversized tensor goes alone
            continue
        if cur and cur_size + size > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(k)
        cur_size += size
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("model-*.safetensors"):
        old.unlink()
    for i, keys in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(nbytes(state[k]) for k in keys)
        if size > SHARD_BUDGET and len(keys) > 1:
            fail(f"shard {name} holds {size} bytes over the {SHARD_BUDGET} budget")
        total += size
        save_file({k: state[k] for k in keys}, OUT / name, metadata={"format": "pt"})
        weight_map.update({k: name for k in keys})
        print(f"wrote {name}: {len(keys)} tensors, {size} bytes")

    if len(weight_map) != 160:
        fail(f"weight_map covers {len(weight_map)} tensors, expected 160")
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
        + "\n"
    )
    print(f"wrote index: {n} shards, {total} bytes total")


if __name__ == "__main__":
    main()
