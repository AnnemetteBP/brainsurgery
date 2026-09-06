"""T3: mixed-precision export of OLMo-1B-0724-hf with sharding.

Plain torch + safetensors. Casts exactly the 112 per-layer projection
matrices to bfloat16 (explicit allowlist, no regex), keeps everything else
float32 with unchanged bytes, and writes a sharded safetensors checkpoint
with a HuggingFace-style index file under out/T3/.
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SANDBOX = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_DIR = os.path.join(SANDBOX, "inputs", "base")
OUT_DIR = os.path.join(SANDBOX, "out", "T3")

NUM_LAYERS = 16
PROJ_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
BF16_NAMES = {
    f"model.layers.{i}.{suffix}" for i in range(NUM_LAYERS) for suffix in PROJ_SUFFIXES
}
EXPECTED_BF16 = 112
EXPECTED_TOTAL = 114
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def layer_sort_key(name: str):
    """embed_tokens, layers in numeric order, then everything else (lm_head)."""
    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if name == "model.embed_tokens.weight":
        return (0, 0, name)
    if m:
        return (1, int(m.group(1)), m.group(2))
    return (2, 0, name)


def load_input() -> dict[str, torch.Tensor]:
    with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(os.path.join(IN_DIR, shard), framework="pt", device="cpu") as s:
            for name in s.keys():
                if name in tensors:
                    fail(f"duplicate tensor {name} across input shards")
                tensors[name] = s.get_tensor(name)
    if set(tensors) != set(index["weight_map"]):
        fail("input index weight_map disagrees with shard contents")
    return tensors


def main() -> None:
    src = load_input()
    if len(src) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} input tensors, found {len(src)}")
    missing = BF16_NAMES - set(src)
    if missing:
        fail(f"{len(missing)} projection matrices missing from input, e.g. {sorted(missing)[:3]}")
    for name, t in src.items():
        if t.dtype != torch.float32:
            fail(f"input tensor {name} is {t.dtype}, expected float32")

    out: dict[str, torch.Tensor] = {}
    for name in sorted(src, key=layer_sort_key):
        t = src[name]
        if name in BF16_NAMES:
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            out[name] = t.contiguous()  # unchanged float32 values

    # ---- Required checks (before writing) ----
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != EXPECTED_BF16:
        fail(f"expected exactly {EXPECTED_BF16} bfloat16 tensors, got {n_bf16}")
    if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        fail("model.layers.0.self_attn.q_proj.weight is not bfloat16")
    if out["model.embed_tokens.weight"].dtype != torch.float32:
        fail("model.embed_tokens.weight is not float32")
    if len(out) != EXPECTED_TOTAL:
        fail(f"expected exactly {EXPECTED_TOTAL} output tensors, got {len(out)}")
    # Extra sanity: names unchanged, no float32 tensor got touched, bf16 set exact.
    if set(out) != set(src):
        fail("tensor name set changed")
    if {n for n, t in out.items() if t.dtype == torch.bfloat16} != BF16_NAMES:
        fail("the set of bfloat16 tensors is not exactly the projection allowlist")
    for name, t in out.items():
        if t.dtype == torch.float32 and not torch.equal(t, src[name]):
            fail(f"float32 tensor {name} changed value")
        if t.shape != src[name].shape:
            fail(f"shape of {name} changed")

    # ---- Sharding: greedy fill in order, oversized tensors alone ----
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, t in out.items():
        nbytes = t.numel() * t.element_size()
        if nbytes > SHARD_BUDGET:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current_bytes + nbytes > SHARD_BUDGET:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    for group in shards:
        total = sum(out[n].numel() * out[n].element_size() for n in group)
        if len(group) > 1 and total > SHARD_BUDGET:
            fail(f"shard exceeds budget: {total} bytes")
    if sum(len(g) for g in shards) != EXPECTED_TOTAL:
        fail("sharding lost or duplicated tensors")

    # ---- Write ----
    os.makedirs(OUT_DIR, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, group in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            fail(f"destination already exists: {path}")
        save_file({n: out[n] for n in group}, path, metadata={"format": "pt"})
        for n in group:
            weight_map[n] = fname
            total_size += out[n].numel() * out[n].element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    # ---- Verify what was written ----
    seen: dict[str, torch.dtype] = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt", device="cpu") as s:
            data_bytes = 0
            for n in s.keys():
                t = s.get_tensor(n)
                seen[n] = t.dtype
                data_bytes += t.numel() * t.element_size()
                if weight_map[n] != fname:
                    fail(f"index maps {n} to a different shard than the one holding it")
                if not torch.equal(t, out[n]):
                    fail(f"written tensor {n} differs from intended")
            if len(list(s.keys())) > 1 and data_bytes > SHARD_BUDGET:
                fail(f"written shard {fname} exceeds budget")
    if len(seen) != EXPECTED_TOTAL:
        fail(f"written checkpoint has {len(seen)} tensors, expected {EXPECTED_TOTAL}")
    if sum(1 for d in seen.values() if d == torch.bfloat16) != EXPECTED_BF16:
        fail("written checkpoint bfloat16 count wrong")

    print(f"OK: wrote {EXPECTED_TOTAL} tensors ({n_bf16} bf16) in {n_shards} shards, "
          f"{total_size} bytes of tensor data -> {OUT_DIR}")
    for i, group in enumerate(shards, start=1):
        b = sum(out[n].numel() * out[n].element_size() for n in group)
        print(f"  shard {i}: {len(group)} tensors, {b} bytes")


if __name__ == "__main__":
    main()
