"""T2 baseline for Pythia-1B: remove attention head 6 from every layer."""

import json
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

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T2")
N_LAYERS = 16
DROP_HEAD = 6
# name, axis holding heads, number of concatenated segments, segment width, head block width
SPECS = [
    ("attention.query_key_value.weight", 0, 1, 6144, 768),
    ("attention.query_key_value.bias", 0, 1, 6144, 768),
    ("attention.dense.weight", 1, 1, 2048, 256),
]


def keep_index(segments: int, seg_size: int, block: int) -> torch.Tensor:
    keep = []
    for seg in range(segments):
        for h in range(seg_size // block):
            if h != DROP_HEAD:
                start = seg * seg_size + h * block
                keep.append(torch.arange(start, start + block))
    return torch.cat(keep)


sd = load_checkpoint("inputs/base/model.safetensors")
for layer in range(N_LAYERS):
    for rel, dim, segments, seg_size, block in SPECS:
        name = "gpt_neox.layers.{i}.".format(i=layer) + rel
        sd[name] = sd[name].index_select(dim, keep_index(segments, seg_size, block)).contiguous()

assert sd["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (5376, 2048)
assert sd["gpt_neox.layers.0.attention.query_key_value.bias"].shape == (5376,)
assert sd["gpt_neox.layers.0.attention.dense.weight"].shape == (2048, 1792)
assert len(sd) == 244

out_dir.mkdir(parents=True, exist_ok=True)
save_file(sd, str(out_dir / "model.safetensors"))
