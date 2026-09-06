"""T3 baseline for Pythia-1B: bfloat16 projection matrices, float32 everything else, 256 MiB (268,435,456 bytes) shards."""

import json
import re
import sys
from pathlib import Path

import torch

from safetensors.torch import load_file, save_file


def load_checkpoint(path):
    """Load a single .safetensors file or a sharded directory with model.safetensors.index.json."""
    path = Path(path)
    if path.is_file():
        return load_file(str(path))
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        sd = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(str(path / shard)))
        return sd
    return load_file(str(path / "model.safetensors"))


def save_sharded_safetensors(sd, out_dir, max_bytes):
    """Pack tensors in order into shards of at most max_bytes of tensor data (an oversized
    tensor goes alone in its own shard) and write model.safetensors.index.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards, cur, cur_size = [], {}, 0
    for name, tensor in sd.items():
        size = tensor.numel() * tensor.element_size()
        if cur and cur_size + size > max_bytes:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[name] = tensor.contiguous()
        cur_size += size
    if cur:
        shards.append(cur)
    weight_map = {}
    for idx, shard in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out_dir / shard_name))
        for name in shard:
            weight_map[name] = shard_name
    total = sum(t.numel() * t.element_size() for t in sd.values())
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T3")
MAX_SHARD = 268435456

sd = load_checkpoint("inputs/base/model.safetensors")
matrix_re = re.compile(r"gpt_neox\.layers\.(\d+)\.(attention\.query_key_value\.weight|attention\.dense\.weight|mlp\.dense_h_to_4h\.weight|mlp\.dense_4h_to_h\.weight)")
buffer_re = re.compile(r"gpt_neox\.layers\.(\d+)\.(attention\.bias|attention\.masked_bias|attention\.rotary_emb\.inv_freq)")

out = {}
for name, tensor in sd.items():
    if buffer_re.fullmatch(name):
        continue
    out[name] = tensor.float().to(torch.bfloat16) if matrix_re.fullmatch(name) else tensor.float()

assert sum(t.dtype == torch.bfloat16 for t in out.values()) == 64
assert out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
assert out["gpt_neox.embed_in.weight"].dtype == torch.float32
assert len(out) == 196, len(out)

save_sharded_safetensors(out, out_dir, MAX_SHARD)
