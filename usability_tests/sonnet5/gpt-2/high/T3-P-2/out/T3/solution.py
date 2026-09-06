"""
T3: Mixed-precision export with sharding (GPT-2, 124M).

- Cast the 48 per-layer projection matrices (attn.c_attn.weight,
  attn.c_proj.weight, mlp.c_fc.weight, mlp.c_proj.weight) to bfloat16.
- Keep everything else (embeddings, layer norms, biases) in float32.
- Drop the 12 non-parameter causal-mask buffers h.<i>.attn.bias.
- Write a sharded safetensors checkpoint (<=64 MiB of tensor data per shard,
  except a single oversized tensor which gets its own shard) plus an index.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")

SHARD_CAP_BYTES = 64 * 1024 * 1024  # 64 MiB

# Exactly the 48 projection matrices that must become bfloat16.
PROJ_RE = re.compile(
    r"^h\.\d+\.(attn\.c_attn\.weight|attn\.c_proj\.weight|mlp\.c_fc\.weight|mlp\.c_proj\.weight)$"
)
# Non-parameter buffers to drop (causal-mask buffer per layer).
BUFFER_RE = re.compile(r"^h\.\d+\.attn\.bias$")


def build_output(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if BUFFER_RE.match(name):
            continue
        if PROJ_RE.match(name):
            out[name] = tensor.to(torch.bfloat16).contiguous()
        else:
            out[name] = tensor.contiguous()
    return out


def plan_shards(tensors: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedily pack tensor names into shards of at most SHARD_CAP_BYTES of
    tensor data each. A tensor bigger than the cap on its own is placed alone
    in its own shard (the only way a shard can exceed the cap)."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in sorted(tensors):
        size = tensors[name].numel() * tensors[name].element_size()
        if current and current_size + size > SHARD_CAP_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
        if current_size > SHARD_CAP_BYTES:
            # Either a single oversized tensor, or we just tipped over the
            # cap; close the shard immediately either way.
            shards.append(current)
            current = []
            current_size = 0
    if current:
        shards.append(current)
    return shards


def main() -> None:
    state_dict = load_file(IN_PATH)
    assert len(state_dict) == 160, f"expected 160 input tensors, got {len(state_dict)}"

    out = build_output(state_dict)

    # --- Required checks: fail loudly before writing anything. ---
    bf16_names = [n for n, t in out.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 48, f"expected exactly 48 bfloat16 tensors, got {len(bf16_names)}"
    assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert out["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(out) == 148, f"expected 148 output tensors, got {len(out)}"
    for name in bf16_names:
        assert PROJ_RE.match(name), f"unexpected tensor cast to bfloat16: {name}"
    for name, tensor in out.items():
        if not PROJ_RE.match(name):
            assert tensor.dtype == torch.float32, f"{name} should be float32, got {tensor.dtype}"
    for i in range(12):
        assert f"h.{i}.attn.bias" not in out, f"buffer h.{i}.attn.bias should have been dropped"

    # --- Shard and write. ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_groups = plan_shards(out)
    num_shards = len(shard_groups)

    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, names in enumerate(shard_groups, start=1):
        shard_filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: out[name] for name in names}
        save_file(shard_tensors, OUT_DIR / shard_filename, metadata={"format": "pt"})
        for name, tensor in shard_tensors.items():
            weight_map[name] = shard_filename
            total_size += tensor.numel() * tensor.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(out)} tensors across {num_shards} shards to {OUT_DIR}")
    print(f"bfloat16 tensors: {len(bf16_names)}")


if __name__ == "__main__":
    main()
