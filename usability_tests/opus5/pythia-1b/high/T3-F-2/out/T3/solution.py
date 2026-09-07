#!/usr/bin/env python
"""T3: mixed-precision sharded export of Pythia-1B.

Route: plain script on torch + safetensors, with the sharding delegated to
huggingface_hub's `split_torch_state_dict_into_shards` (the same splitter
`transformers.save_pretrained` uses), so the layout is the canonical HF one.

Targeting is done with an explicit, exhaustive key list built from the layer
indices, never with a regex, so it cannot over-match embeddings/norms/biases.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T3")

N_LAYERS = 16
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB of tensor data

PROJECTION_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)
BUFFER_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)

PROJECTIONS = {
    f"gpt_neox.layers.{i}.{s}" for i in range(N_LAYERS) for s in PROJECTION_SUFFIXES
}
BUFFERS = {f"gpt_neox.layers.{i}.{s}" for i in range(N_LAYERS) for s in BUFFER_SUFFIXES}

EXPECTED_SHAPES = {
    "attention.query_key_value.weight": (6144, 2048),
    "attention.dense.weight": (2048, 2048),
    "mlp.dense_h_to_4h.weight": (8192, 2048),
    "mlp.dense_4h_to_h.weight": (2048, 8192),
}


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def main() -> None:
    src = load_file(str(SRC))
    print(f"loaded {len(src)} tensors from {SRC}")

    # --- targeting must resolve exactly, no silent misses ---------------------
    missing = sorted((PROJECTIONS | BUFFERS) - set(src))
    if missing:
        fail(f"{len(missing)} targeted names absent from input, first: {missing[0]}")
    if len(PROJECTIONS) != 64:
        fail(f"built {len(PROJECTIONS)} projection names, expected 64")
    if len(BUFFERS) != 48:
        fail(f"built {len(BUFFERS)} buffer names, expected 48")

    for name in sorted(PROJECTIONS):
        suffix = name.split(".", 3)[3]
        want = EXPECTED_SHAPES[suffix]
        got = tuple(src[name].shape)
        if got != want:
            fail(f"{name} has shape {got}, expected {want}")

    # --- build the output state dict -----------------------------------------
    out: dict[str, torch.Tensor] = {}
    for name, tensor in src.items():
        if name in BUFFERS:
            continue
        dtype = torch.bfloat16 if name in PROJECTIONS else torch.float32
        out[name] = tensor.to(dtype).contiguous()

    # --- required checks, before anything is written --------------------------
    n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
    if n_bf16 != 64:
        fail(f"{n_bf16} bfloat16 tensors, expected exactly 64")

    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if out[probe].dtype is not torch.bfloat16:
        fail(f"{probe} is {out[probe].dtype}, expected bfloat16")

    if out["gpt_neox.embed_in.weight"].dtype is not torch.float32:
        fail(f"gpt_neox.embed_in.weight is {out['gpt_neox.embed_in.weight'].dtype}, expected float32")

    if len(out) != 196:
        fail(f"{len(out)} output tensors, expected exactly 196")

    # belt and braces: nothing but the 64 projections escaped float32, and no
    # parameter was dropped along with the buffers.
    off = sorted(k for k, t in out.items() if t.dtype is not torch.float32) 
    if set(off) != PROJECTIONS:
        fail(f"non-float32 set differs from the 64 projections ({len(off)} tensors)")
    if set(src) - set(out) != BUFFERS:
        fail("dropped key set differs from the 48 buffers")

    # --- shard ---------------------------------------------------------------
    plan = split_torch_state_dict_into_shards(
        out,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD_BYTES,
    )

    for filename, keys in plan.filename_to_tensors.items():
        size = sum(out[k].numel() * out[k].element_size() for k in keys)
        if size > MAX_SHARD_BYTES and len(keys) > 1:
            fail(f"{filename} holds {size} bytes over the {MAX_SHARD_BYTES} budget in {len(keys)} tensors")

    DST.mkdir(parents=True, exist_ok=True)
    for stale in DST.glob("*.safetensors"):
        stale.unlink()
    (DST / "model.safetensors.index.json").unlink(missing_ok=True)

    written = 0
    for filename, keys in plan.filename_to_tensors.items():
        shard = {k: out[k] for k in keys}
        save_file(shard, str(DST / filename), metadata={"format": "pt"})
        size = sum(t.numel() * t.element_size() for t in shard.values())
        print(f"  {filename}: {len(shard):3d} tensors, {size:,} bytes")
        written += len(shard)

    if written != 196:
        fail(f"wrote {written} tensors across shards, expected 196")

    if plan.is_sharded:
        index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
    else:
        index = {
            "metadata": plan.metadata,
            "weight_map": {k: next(iter(plan.filename_to_tensors)) for k in out},
        }
    if sorted(index["weight_map"]) != sorted(out):
        fail("weight_map does not cover every output tensor")
    (DST / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    print(f"wrote {len(plan.filename_to_tensors)} shards + index to {DST}")


if __name__ == "__main__":
    main()
