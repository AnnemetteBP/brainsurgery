#!/usr/bin/env python3
"""T5: fold a PEFT LoRA adapter into GPT-2 base weights and export sharded.

Operates directly on the checkpoint files (safetensors + torch), because the
base checkpoint contains the ``h.<i>.attn.bias`` causal-mask buffers, which a
``PeftModel.merge_and_unload()`` -> ``save_pretrained()`` round-trip would not
reproduce in the output key set.  Sharding uses huggingface_hub's standard
splitter, the same one ``save_pretrained`` uses.

Run from the sandbox root:  .venv/bin/python out/T5/solution.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = Path("inputs/base/model.safetensors")
LORA_DIR = Path("inputs/lora")
OUT_DIR = Path("out/T5")
INDEX_NAME = "model.safetensors.index.json"

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB of tensor data, headers excluded
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160
PROBE = "h.0.attn.c_attn.weight"
PROBE_SHAPE = (768, 2304)

LORA_RE = re.compile(r"^base_model\.model\.(?P<target>.+)\.lora_(?P<factor>[AB])\.weight$")


class CheckFailed(RuntimeError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def main() -> int:
    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    check(isinstance(r, int) and r > 0, f"bad r in adapter_config.json: {r!r}")
    scale = alpha / r
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    state = load_file(str(BASE))
    check(len(state) == EXPECTED_TENSORS, f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}")
    base_keys = set(state)

    adapter = load_file(str(LORA_DIR / "adapter_model.safetensors"))

    # ---- group adapter tensors into (A, B) pairs keyed by base tensor name ----
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in adapter.items():
        m = LORA_RE.match(name)
        check(m is not None, f"unrecognised adapter tensor name: {name}")
        target = f"{m.group('target')}.weight"
        slot = pairs.setdefault(target, {})
        check(m.group("factor") not in slot, f"duplicate lora_{m.group('factor')} for {target}")
        slot[m.group("factor")] = tensor

    check(len(pairs) == EXPECTED_PAIRS, f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}")
    for target, slot in pairs.items():
        check(set(slot) == {"A", "B"}, f"incomplete adapter pair for {target}: {sorted(slot)}")
        check(target in base_keys, f"adapter target {target} is not a base tensor")

    # ---- merge ----
    merged: dict[str, torch.Tensor] = {}
    for target in sorted(pairs):
        A = pairs[target]["A"].to(torch.float32)
        B = pairs[target]["B"].to(torch.float32)
        check(A.ndim == 2 and B.ndim == 2, f"{target}: adapter factors must be 2-D")
        check(A.shape[0] == r and B.shape[1] == r, f"{target}: rank mismatch {tuple(A.shape)} {tuple(B.shape)}")
        base = state[target]
        check(base.dtype == torch.float32, f"{target}: base dtype is {base.dtype}, expected float32")

        delta = scale * (B @ A)  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # base uses the Conv1D [in, out] layout
        check(
            tuple(delta.shape) == tuple(base.shape),
            f"{target}: delta {tuple(delta.shape)} does not match base {tuple(base.shape)}",
        )
        state[target] = (base + delta).contiguous()
        merged[target] = state[target]
    print(f"merged {len(merged)} adapted weights: {sorted(merged)}")

    # ---- required checks, before writing ----
    check(len(merged) == EXPECTED_PAIRS, f"merged {len(merged)} weights, expected {EXPECTED_PAIRS}")
    offenders = [k for k in state if "lora_" in k]
    check(not offenders, f"adapter/intermediate tensors would be written: {offenders}")
    check(PROBE in state, f"{PROBE} missing from output")
    check(
        tuple(state[PROBE].shape) == PROBE_SHAPE,
        f"{PROBE} has shape {tuple(state[PROBE].shape)}, expected {list(PROBE_SHAPE)}",
    )
    check(len(state) == EXPECTED_TENSORS, f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}")
    check(set(state) == base_keys, "output key set differs from the base key set")

    # ---- sharded write ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale checkpoint files only; this directory also holds the solution.
    for stale in [*OUT_DIR.glob("*.safetensors"), *OUT_DIR.glob(INDEX_NAME)]:
        stale.unlink()

    split = split_torch_state_dict_into_shards(state, max_shard_size=MAX_SHARD_BYTES)
    check(split.is_sharded, "output was not sharded")
    for filename, keys in split.filename_to_tensors.items():
        shard = {k: state[k].contiguous() for k in keys}
        save_file(shard, str(OUT_DIR / filename), metadata={"format": "pt"})
    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    (OUT_DIR / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(split.filename_to_tensors)} shards + index to {OUT_DIR}")

    verify(state, merged)
    print("OK")
    return 0


def verify(expected: dict[str, torch.Tensor], merged: dict[str, torch.Tensor]) -> None:
    """Re-read what was written and re-assert every requirement."""
    index = json.loads((OUT_DIR / INDEX_NAME).read_text())
    weight_map = index["weight_map"]
    check(set(weight_map) == set(expected), "weight_map does not cover exactly the output tensors")
    check(len(weight_map) == EXPECTED_TENSORS, f"weight_map has {len(weight_map)} entries")

    shards = sorted({Path(p).name for p in weight_map.values()})
    on_disk = sorted(p.name for p in OUT_DIR.glob("*.safetensors"))
    check(shards == on_disk, f"shard files on disk {on_disk} != referenced {shards}")

    seen: dict[str, torch.Tensor] = {}
    for shard_name in shards:
        with safe_open(str(OUT_DIR / shard_name), framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        payload = sum(t.numel() * t.element_size() for t in tensors.values())
        check(
            payload <= MAX_SHARD_BYTES or len(tensors) == 1,
            f"{shard_name}: {payload} bytes of tensor data over the {MAX_SHARD_BYTES} budget "
            f"with {len(tensors)} tensors",
        )
        for k, t in tensors.items():
            check(k not in seen, f"{k} appears in more than one shard")
            check(weight_map[k] == shard_name, f"{k} is in {shard_name}, weight_map says {weight_map[k]}")
            seen[k] = t

    check(len(seen) == EXPECTED_TENSORS, f"read back {len(seen)} tensors, expected {EXPECTED_TENSORS}")
    check(not [k for k in seen if "lora_" in k], "an adapter tensor reached the output")
    check(tuple(seen[PROBE].shape) == PROBE_SHAPE, f"{PROBE} shape changed on disk")

    for k, want in expected.items():
        got = seen[k]
        check(got.dtype == want.dtype, f"{k}: dtype {got.dtype} != {want.dtype}")
        check(tuple(got.shape) == tuple(want.shape), f"{k}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        check(torch.equal(got, want), f"{k}: value differs from what was merged")

    base = load_file(str(BASE))
    untouched = [k for k in base if k not in merged]
    for k in untouched:
        check(torch.equal(seen[k], base[k]), f"{k} was supposed to be unchanged but differs")
    check(
        len(untouched) == EXPECTED_TENSORS - EXPECTED_PAIRS,
        f"{len(untouched)} unchanged tensors, expected {EXPECTED_TENSORS - EXPECTED_PAIRS}",
    )
    print(f"verified: {len(untouched)} tensors bit-exact, {len(merged)} merged, {len(shards)} shards")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
