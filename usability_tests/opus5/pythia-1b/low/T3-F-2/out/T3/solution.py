"""T3: mixed-precision sharded export of Pythia-1B.

Plain torch + safetensors + huggingface_hub's shard splitter (the same helper
transformers uses in save_pretrained), so the shard layout/index is the
standard one serving stacks expect.
"""
import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "inputs" / "base" / "model.safetensors"
DST = ROOT / "out" / "T3"

MAX_SHARD = 256 * 1024 * 1024

BF16 = re.compile(
    r"^gpt_neox\.layers\.\d+\."
    r"(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
DROP = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def die(msg):
    raise SystemExit(f"CHECK FAILED: {msg}")


def main():
    src = load_file(str(SRC))
    out = {}
    dropped = 0
    for name, t in src.items():
        if DROP.match(name):
            dropped += 1
            continue
        out[name] = t.to(torch.bfloat16) if BF16.match(name) else t.to(torch.float32)

    # --- required checks: fail loudly before writing anything ---
    n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
    if n_bf16 != 64:
        die(f"expected 64 bfloat16 tensors, got {n_bf16}")
    k = "gpt_neox.layers.0.attention.query_key_value.weight"
    if out[k].dtype is not torch.bfloat16:
        die(f"{k} is {out[k].dtype}, expected bfloat16")
    if out["gpt_neox.embed_in.weight"].dtype is not torch.float32:
        die("gpt_neox.embed_in.weight is not float32")
    if len(out) != 196:
        die(f"expected 196 output tensors, got {len(out)}")
    if dropped != 48:
        die(f"expected to drop 48 buffers, dropped {dropped}")
    non_bf16 = [n for n, t in out.items() if t.dtype not in (torch.bfloat16, torch.float32)]
    if non_bf16:
        die(f"non-{{bf16,fp32}} tensors remain: {non_bf16[:5]}")

    DST.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(
        out, max_shard_size=MAX_SHARD, filename_pattern="model{suffix}.safetensors"
    )
    for filename, keys in split.filename_to_tensors.items():
        shard = {k: out[k].contiguous() for k in keys}
        nbytes = sum(t.numel() * t.element_size() for t in shard.values())
        if nbytes > MAX_SHARD and len(shard) > 1:
            die(f"shard {filename} holds {nbytes} bytes over the limit with {len(shard)} tensors")
        save_file(shard, str(DST / filename), metadata={"format": "pt"})

    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    (DST / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    mapped = set(index["weight_map"])
    if mapped != set(out):
        die("weight_map does not cover exactly the output tensors")
    print(f"wrote {len(out)} tensors into {len(split.filename_to_tensors)} shard(s) -> {DST}")


if __name__ == "__main__":
    sys.exit(main())
