#!/usr/bin/env python3
"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Casts exactly the per-layer projection matrices to bfloat16, keeps every other
tensor untouched in float32, and writes the result to out/T3/ as a sharded
safetensors checkpoint with a model.safetensors.index.json.

Run from the sandbox root:  python out/T3/solution.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T3")

MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

# The projection matrices to cast, named exactly as the task lists them.
# Suffix -> expected shape. Nothing else in the checkpoint is touched.
PROJECTIONS: dict[str, tuple[int, ...]] = {
    "self_attn.q_proj.weight": (2048, 2048),
    "self_attn.k_proj.weight": (2048, 2048),
    "self_attn.v_proj.weight": (2048, 2048),
    "self_attn.o_proj.weight": (2048, 2048),
    "mlp.gate_proj.weight": (8192, 2048),
    "mlp.up_proj.weight": (8192, 2048),
    "mlp.down_proj.weight": (2048, 8192),
}

EXPECTED_TOTAL = 114
EXPECTED_CAST = 112

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def die(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def main() -> None:
    # ---------------------------------------------------------------- inputs
    index_path = IN_DIR / "model.safetensors.index.json"
    if not index_path.is_file():
        die(f"missing input index {index_path}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    # Source of truth for names/shapes/dtypes: the shard headers themselves,
    # cross-checked against the index so a stale index cannot go unnoticed.
    header: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
    src_of: dict[str, Path] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = IN_DIR / shard_name
        if not shard_path.is_file():
            die(f"missing input shard {shard_path}")
        with safe_open(shard_path, framework="pt") as f:
            for name in f.keys():
                if name in header:
                    die(f"tensor {name!r} appears in more than one input shard")
                sl = f.get_slice(name)
                header[name] = (sl.get_dtype(), tuple(sl.get_shape()))
                src_of[name] = shard_path

    if set(header) != set(weight_map):
        only_files = sorted(set(header) - set(weight_map))
        only_index = sorted(set(weight_map) - set(header))
        die(f"index/shards disagree; only in files: {only_files}, only in index: {only_index}")
    if len(header) != EXPECTED_TOTAL:
        die(f"input has {len(header)} tensors, expected {EXPECTED_TOTAL}")

    bad_dtype = sorted(n for n, (dt, _) in header.items() if dt != "F32")
    if bad_dtype:
        die(f"input tensors are not all float32: {bad_dtype[:5]}")

    # ------------------------------------------------------- select the casts
    layers = sorted({int(m.group(1)) for n in header if (m := LAYER_RE.match(n))})
    if layers != list(range(len(layers))):
        die(f"layer indices are not contiguous from 0: {layers}")

    to_cast: set[str] = set()
    for i in layers:
        for suffix, want_shape in PROJECTIONS.items():
            name = f"model.layers.{i}.{suffix}"
            if name not in header:
                die(f"expected projection {name!r} is not in the checkpoint")
            got_shape = header[name][1]
            if got_shape != want_shape:
                die(f"{name} has shape {got_shape}, expected {want_shape}")
            to_cast.add(name)

    keep = sorted(set(header) - to_cast)
    print(f"layers: {len(layers)}  cast to bfloat16: {len(to_cast)}  kept float32: {len(keep)}")
    print(f"kept in float32: {keep}")

    # ------------------------------------------------- plan the sharded layout
    # Deterministic, name-sorted order; greedy packing up to MAX_SHARD_BYTES.
    # A tensor bigger than the budget lands alone in its own shard.
    def nbytes(name: str) -> int:
        _, shape = header[name]
        n = 1
        for d in shape:
            n *= d
        return n * (2 if name in to_cast else 4)

    plan: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name in sorted(header):
        size = nbytes(name)
        if cur and cur_bytes + size > MAX_SHARD_BYTES:
            plan.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += size
    if cur:
        plan.append(cur)

    n_shards = len(plan)
    shard_names = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
    out_map = {name: shard_names[i] for i, names in enumerate(plan) for name in names}
    planned_dtype = {n: (torch.bfloat16 if n in to_cast else torch.float32) for n in header}
    total_size = sum(nbytes(n) for n in header)

    for i, names in enumerate(plan):
        size = sum(nbytes(n) for n in names)
        if size > MAX_SHARD_BYTES and len(names) > 1:
            die(f"shard {shard_names[i]} holds {size} bytes over budget with {len(names)} tensors")
        print(f"  {shard_names[i]}: {len(names):3d} tensors, {size:,} bytes")

    # ------------------------------------------------ required checks (before writing)
    n_bf16 = sum(1 for dt in planned_dtype.values() if dt is torch.bfloat16)
    if n_bf16 != EXPECTED_CAST:
        die(f"{n_bf16} tensors would be bfloat16, expected exactly {EXPECTED_CAST}")
    probe = "model.layers.0.self_attn.q_proj.weight"
    if planned_dtype.get(probe) is not torch.bfloat16:
        die(f"{probe} is not bfloat16")
    emb = "model.embed_tokens.weight"
    if planned_dtype.get(emb) is not torch.float32:
        die(f"{emb} is not float32")
    if len(out_map) != EXPECTED_TOTAL:
        die(f"output would hold {len(out_map)} tensors, expected {EXPECTED_TOTAL}")
    if set(out_map) != set(header):
        die("output key set differs from the input key set")
    print(f"checks passed: {EXPECTED_CAST} bfloat16, {EXPECTED_TOTAL} tensors, {n_shards} shards")

    # ----------------------------------------------------------------- write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.safetensors"):
        stale.unlink()

    # One input shard is opened at a time per output shard, so peak memory
    # stays at roughly one output shard rather than the whole 5 GB checkpoint.
    for shard_name, names in zip(shard_names, plan):
        by_src: dict[Path, list[str]] = defaultdict(list)
        for name in names:
            by_src[src_of[name]].append(name)
        tensors: dict[str, torch.Tensor] = {}
        for src, src_names in by_src.items():
            with safe_open(src, framework="pt") as f:
                for name in src_names:
                    t = f.get_tensor(name)
                    if name in to_cast:
                        t = t.to(torch.bfloat16)
                    tensors[name] = t.contiguous().clone()
        save_file(tensors, OUT_DIR / shard_name, metadata={"format": "pt"})
        del tensors
        print(f"wrote {shard_name}")

    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": out_map}, indent=2) + "\n"
    )
    print("wrote model.safetensors.index.json")

    # ------------------------------------------------- verify what landed on disk
    seen: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard_name in shard_names:
        path = OUT_DIR / shard_name
        data_bytes = 0
        n_in_shard = 0
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                if name in seen:
                    die(f"{name} written to more than one shard")
                sl = f.get_slice(name)
                shape = tuple(sl.get_shape())
                dt = sl.get_dtype()
                seen[name] = (dt, shape)
                n = 1
                for d in shape:
                    n *= d
                data_bytes += n * (2 if dt == "BF16" else 4)
                n_in_shard += 1
                if out_map[name] != shard_name:
                    die(f"{name} is in {shard_name} but the index says {out_map[name]}")
        if data_bytes > MAX_SHARD_BYTES and n_in_shard > 1:
            die(f"{shard_name} on disk holds {data_bytes} bytes over budget")

    if len(seen) != EXPECTED_TOTAL:
        die(f"output holds {len(seen)} tensors, expected {EXPECTED_TOTAL}")
    n_bf16_disk = sum(1 for dt, _ in seen.values() if dt == "BF16")
    if n_bf16_disk != EXPECTED_CAST:
        die(f"output holds {n_bf16_disk} bfloat16 tensors, expected {EXPECTED_CAST}")
    if seen[probe][0] != "BF16":
        die(f"on disk {probe} is {seen[probe][0]}, expected BF16")
    if seen[emb][0] != "F32":
        die(f"on disk {emb} is {seen[emb][0]}, expected F32")
    for name, (dt, shape) in seen.items():
        if shape != header[name][1]:
            die(f"{name} shape changed: {header[name][1]} -> {shape}")
        want = "BF16" if name in to_cast else "F32"
        if dt != want:
            die(f"{name} is {dt} on disk, expected {want}")

    # Values: bit-exact for the kept tensors, exact round-trip for the cast ones.
    for name in sorted(seen):
        with safe_open(OUT_DIR / out_map[name], framework="pt") as f:
            got = f.get_tensor(name)
        with safe_open(src_of[name], framework="pt") as f:
            ref = f.get_tensor(name)
        want = ref.to(torch.bfloat16) if name in to_cast else ref
        if not torch.equal(got, want):
            die(f"{name} values differ from the expected result")

    print(f"OK: {EXPECTED_TOTAL} tensors, {EXPECTED_CAST} bfloat16, {n_shards} shards, "
          f"{total_size:,} bytes of tensor data")


if __name__ == "__main__":
    sys.exit(main())
