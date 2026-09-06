"""T3: mixed-precision export of Pythia-1B with sharding."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|"
    r"mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    sd = load_file(SRC)
    print(f"loaded {len(sd)} tensors")

    out = {}
    n_proj = n_buf = 0
    for name, t in sd.items():
        if BUFFER_RE.match(name):
            n_buf += 1
            continue
        if PROJ_RE.match(name):
            n_proj += 1
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            out[name] = t.to(torch.float32).contiguous()

    # Required checks (fail loudly before writing).
    if n_proj != 64:
        fail(f"expected 64 projection matrices, matched {n_proj}")
    if n_buf != 48:
        fail(f"expected 48 buffers dropped, dropped {n_buf}")
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 64:
        fail(f"expected exactly 64 bfloat16 tensors, got {n_bf16}")
    if out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype != torch.bfloat16:
        fail("layer 0 query_key_value.weight is not bfloat16")
    if out["gpt_neox.embed_in.weight"].dtype != torch.float32:
        fail("embed_in.weight is not float32")
    if len(out) != 196:
        fail(f"expected 196 output tensors, got {len(out)}")
    n_other = sum(1 for t in out.values() if t.dtype not in (torch.bfloat16, torch.float32))
    if n_other:
        fail(f"{n_other} tensors are neither bfloat16 nor float32")

    # Shard greedily in original key order; oversized tensors go alone.
    shards, cur, cur_bytes = [], {}, 0
    for name, t in out.items():
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[name] = t
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(t.numel() * t.element_size() for t in shard.values())
        assert size <= SHARD_BUDGET or len(shard) == 1, f"{fname} over budget"
        save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for name in shard:
            weight_map[name] = fname
        total += size
        print(f"{fname}: {len(shard)} tensors, {size} bytes")

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"wrote {n} shards, {len(weight_map)} tensors, index written")


if __name__ == "__main__":
    main()
