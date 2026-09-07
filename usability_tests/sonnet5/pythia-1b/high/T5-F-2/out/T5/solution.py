#!/usr/bin/env python
"""T5: LoRA adapter merge with sharded export (Pythia-1B).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). We
merge directly on the checkpoint tensors rather than going through
`peft.merge_and_unload` (which would require instantiating the full
GPTNeoX model) because the spec pins down the exact arithmetic --
`scale = lora_alpha / r`, `B @ A` with no transpose since
`fan_in_fan_out = false`, float32 accumulation, cast back to the base
dtype -- and a direct tensor-level script makes every one of those steps
and every required check explicit and auditable.

Usage: .venv/bin/python out/T5/solution.py
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = REPO_ROOT / "inputs" / "base" / "model.safetensors"
LORA_PATH = REPO_ROOT / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CONFIG_PATH = REPO_ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = REPO_ROOT / "out" / "T5"

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data per shard
SHARD_NAME_TMPL = "model-{index:05d}-of-{total:05d}.safetensors"

# Matches e.g. "base_model.model.gpt_neox.layers.3.attention.query_key_value.lora_A.weight"
LORA_A_RE = re.compile(
    r"^base_model\.model\.(?P<base_name>.+)\.lora_A\.weight$"
)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    if not BASE_PATH.is_file():
        fail(f"base checkpoint not found: {BASE_PATH}")
    if not LORA_PATH.is_file():
        fail(f"adapter checkpoint not found: {LORA_PATH}")

    lora_config = json.loads(LORA_CONFIG_PATH.read_text())
    r = lora_config["r"]
    lora_alpha = lora_config["lora_alpha"]
    fan_in_fan_out = lora_config.get("fan_in_fan_out", False)
    if fan_in_fan_out:
        # Our merge assumes the nn.Linear [out, in] layout throughout; a
        # Conv1D-style adapter would need A and B transposed before the
        # matmul. Fail loudly instead of silently producing wrong values.
        fail("fan_in_fan_out=true is not handled by this script")
    scale = lora_alpha / r

    base = load_all(BASE_PATH)
    adapter = load_all(LORA_PATH)

    base_names = set(base.keys())
    if len(base_names) != 244:
        fail(f"expected 244 base tensors, found {len(base_names)}")

    # --- pair up lora_A / lora_B tensors and resolve them to base tensor names ---
    pairs: dict[str, tuple[str, str]] = {}  # base_name -> (a_key, b_key)
    unmatched = []
    for key in adapter:
        m = LORA_A_RE.match(key)
        if not m:
            continue
        base_name = m.group("base_name") + ".weight"
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in adapter:
            unmatched.append(key)
            continue
        pairs[base_name] = (key, b_key)

    if unmatched:
        fail(f"lora_A tensors with no matching lora_B: {unmatched}")

    # every adapter tensor must have been consumed by exactly one pair
    consumed = set()
    for a_key, b_key in pairs.values():
        consumed.add(a_key)
        consumed.add(b_key)
    leftover = set(adapter.keys()) - consumed
    if leftover:
        fail(f"unrecognized/unpaired adapter tensors: {sorted(leftover)}")

    # required check: exactly 16 adapter pairs
    if len(pairs) != 16:
        fail(f"expected exactly 16 adapter pairs, found {len(pairs)}")

    for base_name in pairs:
        if base_name not in base_names:
            fail(f"adapter targets base tensor not present in checkpoint: {base_name}")

    # --- merge ---
    merged = dict(base)  # shallow copy; unmerged tensors pass through unchanged
    for base_name, (a_key, b_key) in sorted(pairs.items()):
        A = adapter[a_key]
        B = adapter[b_key]
        W = base[base_name]

        if W.dtype != torch.float16:
            fail(f"unexpected base dtype for {base_name}: {W.dtype}")
        if A.shape[1] != W.shape[1] or B.shape[0] != W.shape[0] or A.shape[0] != B.shape[1]:
            fail(
                f"shape mismatch for {base_name}: base={tuple(W.shape)} "
                f"A={tuple(A.shape)} B={tuple(B.shape)}"
            )

        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
        new_weight = (W.to(torch.float32) + delta).to(torch.float16)

        if new_weight.shape != W.shape:
            fail(f"merged shape for {base_name} changed: {W.shape} -> {new_weight.shape}")

        merged[base_name] = new_weight.contiguous()

    # required check: layer 0 qkv weight shape preserved
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(merged[probe].shape) != (6144, 2048):
        fail(f"{probe} has shape {tuple(merged[probe].shape)}, expected (6144, 2048)")

    # required check: no adapter or intermediate tensor leaked into the output
    lora_leftovers = [k for k in merged if "lora_" in k]
    if lora_leftovers:
        fail(f"lora tensors present in merged output: {lora_leftovers}")

    # required check: output tensor count matches base tensor count exactly
    if set(merged.keys()) != base_names:
        fail("merged tensor name set differs from base tensor name set")
    if len(merged) != 244:
        fail(f"expected 244 output tensors, found {len(merged)}")

    # required check: every non-adapted base tensor is bit-exact unchanged
    for name, tensor in base.items():
        if name in pairs:
            continue
        if not torch.equal(merged[name], tensor):
            fail(f"non-adapted tensor changed unexpectedly: {name}")

    # --- shard and write ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    names_in_order = list(merged.keys())  # base file's own order
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names_in_order:
        t = merged[name]
        nbytes = t.numel() * t.element_size()
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, shard_names in enumerate(shards, start=1):
        shard_file = SHARD_NAME_TMPL.format(index=idx, total=total_shards)
        shard_tensors = {name: merged[name] for name in shard_names}
        shard_bytes = sum(t.numel() * t.element_size() for t in shard_tensors.values())
        if shard_bytes > MAX_SHARD_BYTES and len(shard_names) > 1:
            fail(f"shard {shard_file} exceeds budget with more than one tensor")
        total_size += shard_bytes
        save_file(shard_tensors, str(OUT_DIR / shard_file), metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_file

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"Merged {len(pairs)} LoRA pairs (scale={scale}).")
    print(f"Wrote {len(merged)} tensors across {total_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
