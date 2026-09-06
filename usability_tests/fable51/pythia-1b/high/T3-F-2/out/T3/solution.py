"""T3: mixed-precision sharded export of Pythia-1B (plain torch + safetensors).

Explicit name sets are used instead of regexes so that nothing outside the 64
projection matrices can be cast, and nothing outside the 48 buffers dropped.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
OUT = HERE  # out/T3/
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard
N_LAYERS = 16

PROJ_SUFFIXES = (
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
TO_BF16 = {f"gpt_neox.layers.{i}.{s}" for i in range(N_LAYERS) for s in PROJ_SUFFIXES}
TO_DROP = {f"gpt_neox.layers.{i}.{s}" for i in range(N_LAYERS) for s in BUFFER_SUFFIXES}
assert len(TO_BF16) == 64 and len(TO_DROP) == 48


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    out_tensors: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        missing = (TO_BF16 | TO_DROP) - set(keys)
        if missing:
            fail(f"expected tensors absent from input: {sorted(missing)}")
        for k in keys:
            if k in TO_DROP:
                continue
            t = f.get_tensor(k)
            if k in TO_BF16:
                t = t.to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            out_tensors[k] = t.contiguous()

    # ---- required checks (fail before anything is written) ----
    n_bf16 = sum(1 for t in out_tensors.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 64:
        fail(f"expected 64 bfloat16 tensors, got {n_bf16}")
    if out_tensors["gpt_neox.layers.0.attention.query_key_value.weight"].dtype != torch.bfloat16:
        fail("layers.0 query_key_value.weight is not bfloat16")
    if out_tensors["gpt_neox.embed_in.weight"].dtype != torch.float32:
        fail("embed_in.weight is not float32")
    if len(out_tensors) != 196:
        fail(f"expected 196 output tensors, got {len(out_tensors)}")
    if set(out_tensors) != set(keys) - TO_DROP:
        fail("output key set differs from input keys minus the 48 buffers")
    n_f32 = sum(1 for t in out_tensors.values() if t.dtype == torch.float32)
    if n_f32 != 196 - 64:
        fail(f"expected {196 - 64} float32 tensors, got {n_f32}")

    # ---- greedy sharding: a tensor over budget gets its own shard ----
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for k, t in out_tensors.items():
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nbytes
        if cur_bytes > SHARD_BUDGET:  # oversized tensor: close its shard immediately
            shards.append(cur)
            cur, cur_bytes = [], 0
    if cur:
        shards.append(cur)

    for shard in shards:
        size = sum(out_tensors[k].numel() * out_tensors[k].element_size() for k in shard)
        if size > SHARD_BUDGET and len(shard) != 1:
            fail(f"shard exceeds budget with {len(shard)} tensors")

    # ---- write shards and index ----
    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: out_tensors[k] for k in shard}, os.path.join(OUT, fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
            total_size += out_tensors[k].numel() * out_tensors[k].element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    print(f"wrote {n} shards, {len(weight_map)} tensors, {total_size} bytes to {OUT}")


if __name__ == "__main__":
    main()
