"""T3: mixed-precision sharded export of GPT-2 (124M).

Plain torch + safetensors + huggingface_hub's canonical shard splitter.
"""
import json, os, re, sys
import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import split_torch_state_dict_into_shards

SRC = "inputs/base/model.safetensors"
DST = "out/T3"
MAX_SHARD = 64 * 1024 * 1024

PROJ = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUF = re.compile(r"^h\.(\d+)\.attn\.bias$")

sd = load_file(SRC)
out = {}
for k, v in sd.items():
    if BUF.match(k):
        continue
    out[k] = v.to(torch.bfloat16) if PROJ.match(k) else v.to(torch.float32)

# --- required checks: fail loudly before writing ---
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
assert n_bf16 == 48, f"expected 48 bfloat16 tensors, got {n_bf16}"
assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, "h.0.attn.c_attn.weight not bfloat16"
assert out["wte.weight"].dtype == torch.float32, "wte.weight not float32"
assert len(out) == 148, f"expected 148 tensors, got {len(out)}"
assert sum(1 for k in sd if BUF.match(k)) == 12
assert set(sd) - set(out) == {f"h.{i}.attn.bias" for i in range(12)}, "wrong keys dropped"

os.makedirs(DST, exist_ok=True)
split = split_torch_state_dict_into_shards(
    out, filename_pattern="model{suffix}.safetensors", max_shard_size=MAX_SHARD
)
for fname, keys in split.filename_to_tensors.items():
    save_file({k: out[k].contiguous() for k in keys}, os.path.join(DST, fname),
              metadata={"format": "pt"})

index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)

# --- post-write verification ---
seen = {}
for fname in set(index["weight_map"].values()):
    d = load_file(os.path.join(DST, fname))
    tot = sum(t.numel() * t.element_size() for t in d.values())
    if len(d) > 1:
        assert tot <= MAX_SHARD, f"{fname}: {tot} bytes > {MAX_SHARD}"
    seen.update(d)
assert len(seen) == 148 and set(seen) == set(out)
for k in out:
    assert seen[k].dtype == out[k].dtype and torch.equal(seen[k], out[k]), k
print(f"OK: {len(out)} tensors, {n_bf16} bf16, {len(set(index['weight_map'].values()))} shards")
