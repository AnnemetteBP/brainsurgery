"""T4 baseline for Pythia-1B: task-vector merge of two fine-tunes, lambda 0.4 each, MLP tensors only."""

import json
import re
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

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T4")
LAMBDA = 0.4

base = load_checkpoint("inputs/base/model.safetensors")
ft1 = load_checkpoint("inputs/ft1/model.safetensors")
ft2 = load_checkpoint("inputs/ft2/model.safetensors")
mlp_re = re.compile(r"gpt_neox\.layers\.(\d+)\.(mlp\.dense_h_to_4h\.weight|mlp\.dense_h_to_4h\.bias|mlp\.dense_4h_to_h\.weight|mlp\.dense_4h_to_h\.bias)")

assert set(base) == set(ft1) == set(ft2)
for name in base:
    if not mlp_re.fullmatch(name):
        assert torch.equal(base[name], ft1[name]), f"ft1 differs on shared tensor {name}"
        assert torch.equal(base[name], ft2[name]), f"ft2 differs on shared tensor {name}"

out = dict(base)
merged = 0
for name in base:
    if mlp_re.fullmatch(name):
        b, f1, f2 = base[name].float(), ft1[name].float(), ft2[name].float()
        out[name] = (b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)).to(base[name].dtype)
        merged += 1
assert merged == 64
assert len(out) == 244

out_dir.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_dir / "model.safetensors"))
