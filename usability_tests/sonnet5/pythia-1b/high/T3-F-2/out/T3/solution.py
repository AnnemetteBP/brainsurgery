"""T3: mixed-precision export with sharding for Pythia-1B.

Plain script on top of `safetensors` + `torch` (see F-allowed.md: transformers'
sharded dtype export was considered but it does not support per-tensor dtype
overrides in one pass, only a single output dtype for the whole model; a
direct script gives exact control over which tensors get bfloat16 vs
float32 and over the shard packing).

Steps:
  1. Load every tensor from inputs/base/model.safetensors.
  2. Drop the 48 non-parameter buffers (attention.bias, attention.masked_bias,
     attention.rotary_emb.inv_freq for each of the 16 layers).
  3. Cast the 64 projection weight matrices to bfloat16.
  4. Upcast everything else to float32.
  5. Run the required checks (fail loudly before writing anything).
  6. Greedily pack tensors into shards of at most 256 MiB of tensor data,
     giving any tensor that alone exceeds the limit its own shard.
  7. Write the shards plus model.safetensors.index.json.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # out/T3 -> out -> sandbox root
INPUT_PATH = REPO_ROOT / "inputs" / "base" / "model.safetensors"
OUTPUT_DIR = REPO_ROOT / "out" / "T3"

SHARD_LIMIT_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes, tensor data only

# Non-parameter buffers to drop, per layer.
BUFFER_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)

# The 4 projection matrices per layer that must become bfloat16.
BF16_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\."
    r"(attention\.(query_key_value|dense)\.weight|"
    r"mlp\.(dense_h_to_4h|dense_4h_to_h)\.weight)$"
)


def build_output_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if BUFFER_PATTERN.match(key):
            continue
        if BF16_PATTERN.match(key):
            out[key] = tensor.to(torch.bfloat16).contiguous()
        else:
            out[key] = tensor.to(torch.float32).contiguous()
    return out


def run_checks(state_dict: dict[str, torch.Tensor]) -> None:
    bf16_count = sum(1 for t in state_dict.values() if t.dtype == torch.bfloat16)
    assert bf16_count == 64, f"expected exactly 64 bfloat16 tensors, got {bf16_count}"

    key = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert state_dict[key].dtype == torch.bfloat16, f"{key} must be bfloat16"

    key = "gpt_neox.embed_in.weight"
    assert state_dict[key].dtype == torch.float32, f"{key} must be float32"

    assert len(state_dict) == 196, f"expected exactly 196 output tensors, got {len(state_dict)}"

    # Every other tensor must be float32 (no stray dtypes leaked through).
    for k, t in state_dict.items():
        if t.dtype not in (torch.bfloat16, torch.float32):
            raise AssertionError(f"{k} has unexpected dtype {t.dtype}")

    # No dropped parameter: every buffer key we removed must actually be a
    # buffer name, and every non-buffer input key must still be present.
    raw_keys = set(load_file(INPUT_PATH).keys())
    dropped = raw_keys - set(state_dict.keys())
    assert len(dropped) == 48, f"expected to drop exactly 48 buffers, dropped {len(dropped)}"
    assert all(BUFFER_PATTERN.match(k) for k in dropped), "dropped a non-buffer tensor"


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def plan_shards(state_dict: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedy bin-packing in sorted-key order; oversized tensors get their own shard."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for key in sorted(state_dict.keys()):
        size = tensor_nbytes(state_dict[key])
        if size > SHARD_LIMIT_BYTES:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([key])
            continue
        if current and current_size + size > SHARD_LIMIT_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(key)
        current_size += size

    if current:
        shards.append(current)
    return shards


def write_shards(state_dict: dict[str, torch.Tensor], shards: list[list[str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, keys in enumerate(shards, start=1):
        filename = f"model-{idx:05d}-of-{total:05d}.safetensors"
        shard_dict = {k: state_dict[k] for k in keys}
        save_file(shard_dict, OUTPUT_DIR / filename, metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = filename
            total_size += tensor_nbytes(state_dict[k])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def main() -> None:
    raw = load_file(INPUT_PATH)
    state_dict = build_output_state_dict(raw)
    run_checks(state_dict)
    shards = plan_shards(state_dict)
    write_shards(state_dict, shards)
    print(f"wrote {len(state_dict)} tensors in {len(shards)} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
