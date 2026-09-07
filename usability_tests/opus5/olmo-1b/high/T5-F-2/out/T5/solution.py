#!/usr/bin/env python3
"""T5: fold a PEFT LoRA adapter into OLMo-1B base weights and export sharded.

Works directly on the checkpoint files: no model is instantiated. Tools used:
  - safetensors  : read the base shards / adapter, write the output shards
  - torch        : float32 matmul for B @ A
  - huggingface_hub.split_torch_state_dict_into_shards : the canonical HF
    shard packer, driven with max_shard_size = 512 MiB

Every "Required check" from TASK.md is a hard assertion that runs *before*
anything is written, plus a re-read verification pass afterwards.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

BASE_DIR = Path("inputs/base")
LORA_DIR = Path("inputs/lora")
OUT_DIR = Path("out/T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
EXPECTED_TENSORS = 114
EXPECTED_PAIRS = 32
INDEX_NAME = "model.safetensors.index.json"

# base_model.model.<base module path>.lora_{A,B}.weight
ADAPTER_RE = re.compile(
    r"^base_model\.model\.(?P<module>model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj))"
    r"\.lora_(?P<factor>[AB])\.weight$"
)


class CheckFailed(RuntimeError):
    """A required check did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def load_base() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    index = json.loads((BASE_DIR / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        for name, tensor in load_file(BASE_DIR / shard).items():
            check(name not in state, f"duplicate tensor across base shards: {name}")
            state[name] = tensor
    check(
        len(state) == EXPECTED_TENSORS,
        f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}",
    )
    check(
        set(state) == set(weight_map),
        "base shard contents disagree with the index weight_map",
    )
    return state, weight_map


def load_adapter() -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    check(r > 0, f"adapter r must be positive, got {r}")
    check(
        cfg.get("fan_in_fan_out", False) is False,
        "fan_in_fan_out is true: the factors would need transposing; refusing to guess",
    )
    scale = alpha / r

    raw = load_file(LORA_DIR / "adapter_model.safetensors")
    factors: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in raw.items():
        m = ADAPTER_RE.match(name)
        check(m is not None, f"unrecognised adapter tensor name: {name}")
        assert m is not None
        slot = factors.setdefault(m["module"], {})
        check(m["factor"] not in slot, f"duplicate lora_{m['factor']} for {m['module']}")
        slot[m["factor"]] = tensor

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for module, slot in factors.items():
        check(set(slot) == {"A", "B"}, f"{module}: incomplete adapter pair {sorted(slot)}")
        a, b = slot["A"], slot["B"]
        check(a.ndim == 2 and b.ndim == 2, f"{module}: adapter factors must be 2-D")
        check(a.shape[0] == r, f"{module}: lora_A rows {a.shape[0]} != r {r}")
        check(b.shape[1] == r, f"{module}: lora_B cols {b.shape[1]} != r {r}")
        pairs[module] = (a, b)

    check(
        len(pairs) == EXPECTED_PAIRS,
        f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}",
    )
    check(
        len(raw) == 2 * EXPECTED_PAIRS,
        f"adapter has {len(raw)} tensors, expected {2 * EXPECTED_PAIRS}",
    )
    return pairs, scale


def merge(
    state: dict[str, torch.Tensor],
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> int:
    merged = 0
    for module, (a, b) in sorted(pairs.items()):
        key = f"{module}.weight"
        check(key in state, f"adapter targets {key}, which is not in the base checkpoint")
        base = state[key]
        check(
            base.dtype == torch.float32 and a.dtype == torch.float32 and b.dtype == torch.float32,
            f"{key}: expected float32 base and adapter factors",
        )
        # fan_in_fan_out = false: base is [out, in] like the factors, so B @ A adds directly.
        delta = (b.to(torch.float32) @ a.to(torch.float32)).mul_(scale)
        check(
            delta.shape == base.shape,
            f"{key}: delta {tuple(delta.shape)} != base {tuple(base.shape)}",
        )
        state[key] = (base + delta).to(torch.float32).contiguous()
        merged += 1
    check(merged == EXPECTED_PAIRS, f"merged {merged} pairs, expected {EXPECTED_PAIRS}")
    return merged


def required_checks(state: dict[str, torch.Tensor], base_keys: set[str], merged: int) -> None:
    check(merged == EXPECTED_PAIRS, f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")

    leaked = sorted(k for k in state if "lora_" in k)
    check(not leaked, f"adapter/intermediate tensors leaked into the output: {leaked}")

    probe = "model.layers.0.self_attn.q_proj.weight"
    check(probe in state, f"{probe} missing from the output")
    check(
        tuple(state[probe].shape) == (2048, 2048),
        f"{probe} has shape {tuple(state[probe].shape)}, expected (2048, 2048)",
    )

    check(
        len(state) == EXPECTED_TENSORS,
        f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}",
    )
    check(set(state) == base_keys, "output key set differs from the base key set")
    bad = sorted(k for k, v in state.items() if v.dtype != torch.float32)
    check(not bad, f"non-float32 tensors in the output: {bad}")


def write_sharded(state: dict[str, torch.Tensor]) -> None:
    # Clear only checkpoint artefacts of a previous run. Never wipe OUT_DIR
    # wholesale: this script lives in it.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in [*OUT_DIR.glob("*.safetensors"), OUT_DIR / INDEX_NAME]:
        stale.unlink(missing_ok=True)

    split = split_torch_state_dict_into_shards(
        {k: state[k] for k in sorted(state)},
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD_BYTES,
    )
    for filename, keys in split.filename_to_tensors.items():
        save_file(
            {k: state[k].contiguous() for k in keys},
            OUT_DIR / filename,
            metadata={"format": "pt"},
        )
    index = {
        "metadata": {"total_size": split.metadata["total_size"]},
        "weight_map": dict(sorted(split.tensor_to_filename.items())),
    }
    (OUT_DIR / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def verify_written(reference: dict[str, torch.Tensor]) -> None:
    """Re-read what landed on disk and re-run every check against it."""
    index = json.loads((OUT_DIR / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shard_files = sorted(p.name for p in OUT_DIR.glob("*.safetensors"))
    check(
        set(shard_files) == set(weight_map.values()),
        "shard files on disk do not match the index weight_map",
    )

    state: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        tensors = load_file(OUT_DIR / shard)
        payload = sum(t.numel() * t.element_size() for t in tensors.values())
        check(
            payload <= MAX_SHARD_BYTES,
            f"{shard} holds {payload} bytes of tensor data, over the {MAX_SHARD_BYTES} limit",
        )
        for name, tensor in tensors.items():
            check(name not in state, f"{name} appears in more than one output shard")
            check(
                weight_map[name] == shard,
                f"{name} indexed to {weight_map[name]}, found in {shard}",
            )
            state[name] = tensor

    required_checks(state, set(reference), EXPECTED_PAIRS)
    for name, tensor in state.items():
        ref = reference[name]
        check(tensor.shape == ref.shape, f"{name}: shape changed on write")
        check(tensor.dtype == ref.dtype, f"{name}: dtype changed on write")
        check(torch.equal(tensor, ref), f"{name}: values changed on write")
    print(f"verified {len(state)} tensors across {len(shard_files)} shards")


def main() -> int:
    base, _ = load_base()
    base_keys = set(base)
    pairs, scale = load_adapter()
    print(f"adapter: {len(pairs)} pairs, scale = alpha/r = {scale}")

    original = {k: base[k].clone() for k in (f"{m}.weight" for m in pairs)}
    merged = merge(base, pairs, scale)
    required_checks(base, base_keys, merged)

    # The 82 untouched tensors must still be the objects we read; the 32 merged
    # ones must actually have moved.
    for key, before in original.items():
        check(not torch.equal(base[key], before), f"{key} was not modified by the merge")

    write_sharded(base)
    verify_written(base)
    print(f"wrote {OUT_DIR}: {len(base)} tensors, {merged} merged")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
