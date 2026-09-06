"""T3: mixed-precision sharded export of GPT-2 (124M).

Plain torch + safetensors. Casts exactly the 48 per-layer projection matrices
to bfloat16, drops the 12 causal-mask buffers, leaves every other tensor
float32 and bit-identical, then writes HF-style shards with an index file.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data per shard
N_LAYERS = 12

# Explicit, anchored name patterns: only the four projection matrices per layer.
PROJ_RE = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
# Non-parameter causal-mask buffers to drop.
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    with safe_open(SRC, framework="pt") as f:
        src_keys = sorted(f.keys())
        tensors = {k: f.get_tensor(k) for k in src_keys}

    if len(tensors) != 160:
        fail(f"expected 160 input tensors, got {len(tensors)}")

    out: dict[str, torch.Tensor] = {}
    n_cast = 0
    n_dropped = 0
    for name, t in tensors.items():
        if BUFFER_RE.match(name):
            n_dropped += 1
            continue
        if PROJ_RE.match(name):
            if t.dtype != torch.float32 or t.ndim != 2:
                fail(f"{name}: unexpected dtype/shape {t.dtype} {tuple(t.shape)}")
            out[name] = t.to(torch.bfloat16).contiguous()
            n_cast += 1
        else:
            if t.dtype != torch.float32:
                fail(f"{name}: expected float32 passthrough, got {t.dtype}")
            out[name] = t.contiguous()

    # ---- Required checks (before writing) ----
    bf16 = [k for k, v in out.items() if v.dtype == torch.bfloat16]
    if len(bf16) != 48:
        fail(f"expected exactly 48 bfloat16 tensors, got {len(bf16)}: {bf16}")
    if n_cast != 4 * N_LAYERS:
        fail(f"cast {n_cast} matrices, expected {4 * N_LAYERS}")
    if n_dropped != N_LAYERS:
        fail(f"dropped {n_dropped} buffers, expected {N_LAYERS}")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        fail("h.0.attn.c_attn.weight is not bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        fail(f"wte.weight is {out['wte.weight'].dtype}, expected float32")
    if len(out) != 148:
        fail(f"output has {len(out)} tensors, expected 148")
    non_bf16 = [k for k, v in out.items() if v.dtype != torch.bfloat16 and v.dtype != torch.float32]
    if non_bf16:
        fail(f"unexpected dtypes: {non_bf16}")
    # Every surviving name must exist in the source with the same shape (names unchanged).
    for k, v in out.items():
        if k not in tensors or tuple(v.shape) != tuple(tensors[k].shape):
            fail(f"{k}: name/shape drift")
    # Bit-exact passthrough for float32 tensors.
    for k, v in out.items():
        if v.dtype == torch.float32 and not torch.equal(v, tensors[k]):
            fail(f"{k}: float32 values changed")

    # ---- Greedy sharding in sorted key order ----
    shards: list[dict[str, torch.Tensor]] = []
    sizes: list[int] = []
    cur: dict[str, torch.Tensor] = {}
    cur_size = 0
    for name, t in out.items():
        nbytes = t.numel() * t.element_size()
        if nbytes > MAX_SHARD_BYTES:
            # Oversized tensor goes alone in its own shard.
            if cur:
                shards.append(cur); sizes.append(cur_size)
                cur, cur_size = {}, 0
            shards.append({name: t}); sizes.append(nbytes)
            continue
        if cur and cur_size + nbytes > MAX_SHARD_BYTES:
            shards.append(cur); sizes.append(cur_size)
            cur, cur_size = {}, 0
        cur[name] = t
        cur_size += nbytes
    if cur:
        shards.append(cur); sizes.append(cur_size)

    # Sharding invariants.
    for sd, sz in zip(shards, sizes):
        if sz > MAX_SHARD_BYTES and len(sd) != 1:
            fail(f"shard exceeds 64 MiB with {len(sd)} tensors")
    if sum(len(s) for s in shards) != 148:
        fail("shard partition does not cover 148 tensors")

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map: dict[str, str] = {}
    total = 0
    for i, sd in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            fail(f"destination already exists: {path}")
        save_file(sd, path, metadata={"format": "pt"})
        for k, t in sd.items():
            weight_map[k] = fname
            total += t.numel() * t.element_size()

    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)

    print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes of tensor data")
    for i, sz in enumerate(sizes, start=1):
        print(f"  shard {i}: {len(shards[i-1])} tensors, {sz} bytes")


if __name__ == "__main__":
    main()
