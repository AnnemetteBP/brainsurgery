"""T3: mixed-precision sharded export of Pythia-1B (plain torch + safetensors)."""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base" / "model.safetensors"
OUT = ROOT / "out" / "T3"
MAX_SHARD = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"CHECK FAILED: {msg}", file=sys.stderr)
        sys.exit(1)


src = load_file(str(SRC))
check(len(src) == 244, f"expected 244 input tensors, got {len(src)}")

out: dict[str, torch.Tensor] = {}
n_bf16 = n_dropped = 0
for name, t in src.items():
    if BUFFER_RE.match(name):
        n_dropped += 1
        continue
    if PROJ_RE.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
        n_bf16 += 1
    else:
        out[name] = t.to(torch.float32).contiguous()

# Required checks (before writing).
check(n_dropped == 48, f"expected to drop 48 buffers, dropped {n_dropped}")
check(n_bf16 == 64, f"expected 64 bfloat16 casts, got {n_bf16}")
check(sum(t.dtype == torch.bfloat16 for t in out.values()) == 64, "exactly 64 bf16 tensors")
check(all(t.dtype in (torch.bfloat16, torch.float32) for t in out.values()), "only bf16/f32")
check(out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16,
      "layer0 qkv must be bf16")
check(out["gpt_neox.embed_in.weight"].dtype == torch.float32, "embed_in must be f32")
check(out["embed_out.weight"].dtype == torch.float32, "embed_out must be f32")
check(len(out) == 196, f"expected 196 output tensors, got {len(out)}")
for name, t in out.items():
    check(t.shape == src[name].shape, f"shape changed for {name}")

# Greedy sharding in sorted key order; oversized tensors get their own shard.
shards: list[list[str]] = []
cur: list[str] = []
cur_bytes = 0
for name in sorted(out):
    nbytes = out[name].numel() * out[name].element_size()
    if cur and cur_bytes + nbytes > MAX_SHARD:
        shards.append(cur)
        cur, cur_bytes = [], 0
    cur.append(name)
    cur_bytes += nbytes
if cur:
    shards.append(cur)

OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.safetensors"):
    old.unlink()
total = len(shards)
weight_map: dict[str, str] = {}
total_size = 0
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    data = {n: out[n] for n in names}
    size = sum(t.numel() * t.element_size() for t in data.values())
    check(size <= MAX_SHARD or len(data) == 1, f"shard {fname} exceeds budget")
    save_file(data, str(OUT / fname), metadata={"format": "pt"})
    for n in names:
        weight_map[n] = fname
    total_size += size

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
(OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

# Post-write verification.
check(len(weight_map) == 196, "index must map 196 tensors")
reloaded = {}
for f in sorted(set(weight_map.values())):
    reloaded.update(load_file(str(OUT / f)))
check(set(reloaded) == set(out), "reloaded key set mismatch")
for n, t in out.items():
    check(reloaded[n].dtype == t.dtype and torch.equal(reloaded[n], t), f"roundtrip mismatch {n}")
print(f"OK: {len(out)} tensors, {n_bf16} bf16, {total} shards, {total_size} bytes")
for i, names in enumerate(shards, 1):
    print(f"  shard {i}: {len(names)} tensors, "
          f"{sum(out[n].numel()*out[n].element_size() for n in names)} bytes")
