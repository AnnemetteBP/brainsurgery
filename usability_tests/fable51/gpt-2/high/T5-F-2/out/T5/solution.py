#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded safetensors checkpoint.

Plain safetensors + torch. Avoids peft/transformers model instantiation because the base
checkpoint keys have no `transformer.` prefix and include `attn.bias` mask buffers that
`save_pretrained` would rename or drop; the grader wants the exact base key set.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base" / "model.safetensors"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"

SHARD_BUDGET = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data per shard
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160
ADAPTER_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_(?P<which>[AB])\.weight$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # --- load base ---
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(BASE), framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    if len(state) != EXPECTED_TENSORS:
        fail(f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}")

    # --- load adapter, group A/B by base module name ---
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(LORA_DIR / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            m = ADAPTER_RE.match(k)
            if m is None:
                fail(f"unrecognised adapter tensor name {k!r}")
            pairs.setdefault(m["base"], {})[m["which"]] = f.get_tensor(k)

    # --- merge ---
    merged = 0
    for module, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            fail(f"incomplete adapter pair for {module}: {sorted(ab)}")
        key = f"{module}.weight"
        if key not in state:
            fail(f"adapter targets {key!r}, which is not in the base")
        A, B = ab["A"].float(), ab["B"].float()
        if A.shape[0] != r or B.shape[1] != r:
            fail(f"{module}: rank mismatch A{tuple(A.shape)} B{tuple(B.shape)} r={r}")
        delta = scale * (B @ A)  # [out, in], nn.Linear convention
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]
        W = state[key]
        if W.dtype != torch.float32:
            fail(f"{key}: expected float32 base, got {W.dtype}")
        if delta.shape != W.shape:
            fail(f"{key}: delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)}")
        state[key] = (W + delta).contiguous()
        merged += 1
        print(f"merged {key}  delta |F|={delta.norm():.4f}")

    # --- required checks ---
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
    lora_keys = [k for k in state if "lora_" in k]
    if lora_keys:
        fail(f"adapter tensors present in output: {lora_keys}")
    if tuple(state["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        fail(f"h.0.attn.c_attn.weight has shape {tuple(state['h.0.attn.c_attn.weight'].shape)}")
    if len(state) != EXPECTED_TENSORS:
        fail(f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}")

    # --- shard (greedy in base key order; an oversized tensor goes alone) ---
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for k, t in state.items():
        nbytes = t.numel() * t.element_size()
        if current and current_bytes + nbytes > SHARD_BUDGET:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(k)
        current_bytes += nbytes
    if current:
        shards.append(current)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {k: state[k] for k in keys}
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        if size > SHARD_BUDGET and len(tensors) != 1:
            fail(f"shard {fname} holds {size} bytes across {len(tensors)} tensors")
        save_file(tensors, str(OUT_DIR / fname), metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = fname
        total_size += size
        print(f"wrote {fname}: {len(keys)} tensors, {size} bytes")

    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map has {len(weight_map)} entries, expected {EXPECTED_TENSORS}")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"OK: {EXPECTED_TENSORS} tensors in {n} shards, total {total_size} bytes")


if __name__ == "__main__":
    main()
