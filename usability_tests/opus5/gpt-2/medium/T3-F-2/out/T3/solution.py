"""T3: mixed-precision sharded export of GPT-2 (124M).

Plain torch + safetensors. Casts exactly the 48 projection matrices to
bfloat16, keeps everything else float32, drops the causal-mask buffers, and
writes a greedily-packed sharded checkpoint with an HF-style index.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67,108,864

# Exactly the four projection matrices per layer. Dots are escaped and the
# pattern is anchored so it cannot reach ln_*, wte/wpe or the .bias tensors.
BF16_RE = re.compile(
    r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)
# The causal-mask buffer, not a parameter.
DROP_RE = re.compile(r"^h\.\d+\.attn\.bias$")

EXPECTED_BF16 = 48
EXPECTED_DROPPED = 12
EXPECTED_TOTAL = 148


def build() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    dropped = []
    with safe_open(IN_PATH, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            if DROP_RE.match(k):
                dropped.append(k)
                continue
            t = f.get_tensor(k)
            if BF16_RE.match(k):
                t = t.to(torch.bfloat16)
            elif t.dtype != torch.float32:
                t = t.to(torch.float32)
            out[k] = t.contiguous()

    if len(dropped) != EXPECTED_DROPPED:
        raise AssertionError(
            f"expected to drop {EXPECTED_DROPPED} buffers, dropped {len(dropped)}: {dropped}"
        )
    return out


def check(sd: dict[str, torch.Tensor]) -> None:
    """Required checks. Every one of these fails loudly before anything is written."""
    bf16 = sorted(k for k, v in sd.items() if v.dtype == torch.bfloat16)
    if len(bf16) != EXPECTED_BF16:
        raise AssertionError(f"expected {EXPECTED_BF16} bfloat16 tensors, got {len(bf16)}: {bf16}")
    if sd["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        raise AssertionError("h.0.attn.c_attn.weight is not bfloat16")
    if sd["wte.weight"].dtype != torch.float32:
        raise AssertionError(f"wte.weight is {sd['wte.weight'].dtype}, expected float32")
    if len(sd) != EXPECTED_TOTAL:
        raise AssertionError(f"expected {EXPECTED_TOTAL} tensors, got {len(sd)}")

    # Nothing but the 48 projections may have left float32, and no parameter
    # may have gone missing.
    bad = [k for k, v in sd.items() if v.dtype != torch.float32 and not BF16_RE.match(k)]
    if bad:
        raise AssertionError(f"non-float32 tensors outside the bf16 set: {bad}")
    missing = [k for k in sd if BF16_RE.match(k) and sd[k].dtype != torch.bfloat16]
    if missing:
        raise AssertionError(f"projection matrices not cast: {missing}")
    with safe_open(IN_PATH, framework="pt") as f:
        src = set(f.keys())
    lost = sorted(k for k in src - set(sd) if not DROP_RE.match(k))
    if lost:
        raise AssertionError(f"dropped non-buffer tensors: {lost}")
    if set(sd) - src:
        raise AssertionError(f"invented tensor names: {sorted(set(sd) - src)}")


def shard(sd: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedy packing in key order; an oversized tensor lands alone in its shard."""
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for k, v in sd.items():
        n = v.numel() * v.element_size()
        if cur and cur_bytes + n > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += n
    if cur:
        shards.append(cur)

    for i, s in enumerate(shards):
        total = sum(sd[k].numel() * sd[k].element_size() for k in s)
        if total > MAX_SHARD_BYTES and len(s) != 1:
            raise AssertionError(f"shard {i} is {total} bytes over budget with {len(s)} tensors")
    return shards


def main() -> None:
    sd = build()
    check(sd)
    shards = shard(sd)

    if os.path.isdir(OUT_DIR):
        for name in os.listdir(OUT_DIR):
            if name.endswith(".safetensors") or name == "model.safetensors.index.json":
                os.remove(os.path.join(OUT_DIR, name))
    os.makedirs(OUT_DIR, exist_ok=True)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        part = {k: sd[k] for k in keys}
        save_file(part, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = fname
            total_size += sd[k].numel() * sd[k].element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=False)
        f.write("\n")

    verify(sd, n)
    print(f"wrote {len(sd)} tensors into {n} shards, {total_size} bytes")


def verify(sd: dict[str, torch.Tensor], n_shards: int) -> None:
    """Re-read what was written and confirm the checks hold on disk."""
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    wm = index["weight_map"]
    if set(wm) != set(sd):
        raise AssertionError("index weight_map does not cover exactly the output tensors")

    seen: dict[str, torch.Tensor] = {}
    for fname in sorted(set(wm.values())):
        path = os.path.join(OUT_DIR, fname)
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            total = 0
            for k in keys:
                t = f.get_tensor(k)
                seen[k] = t
                total += t.numel() * t.element_size()
        if total > MAX_SHARD_BYTES and len(keys) != 1:
            raise AssertionError(f"{fname}: {total} bytes over budget with {len(keys)} tensors")
    check(seen)

    with safe_open(IN_PATH, framework="pt") as f:
        for k, t in seen.items():
            ref = f.get_tensor(k)
            want = ref.to(torch.bfloat16) if BF16_RE.match(k) else ref
            if t.dtype != want.dtype or t.shape != want.shape:
                raise AssertionError(f"{k}: dtype/shape mismatch")
            if not torch.equal(t, want):
                raise AssertionError(f"{k}: values are not bit-exact")


if __name__ == "__main__":
    main()
