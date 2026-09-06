"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf).

Casts the 112 per-layer projection matrices to bfloat16, keeps everything
else (embeddings, lm_head) in float32, and writes a sharded safetensors
checkpoint with an index file, respecting a 256 MiB per-shard tensor-data
budget (oversized tensors get their own shard).
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/rootkidd/Projects/brainsurgery/usability_tests/sonnet5/olmo-1b/high/T3-P-2")
INPUT_DIR = REPO_ROOT / "inputs" / "base"
OUTPUT_DIR = REPO_ROOT / "out" / "T3"

SHARD_BUDGET_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes

# Exactly the 112 large projection matrices: q/k/v/o_proj and gate/up/down_proj
# per layer. Anchored, escaped-dot regex so it cannot also match embeddings,
# norms or biases.
PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_state_dict(input_dir: Path) -> dict[str, torch.Tensor]:
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Group tensor names by the shard file that holds them so each shard is
    # opened once.
    by_file: dict[str, list[str]] = {}
    for name, shard_file in weight_map.items():
        by_file.setdefault(shard_file, []).append(name)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file, names in by_file.items():
        with safe_open(input_dir / shard_file, framework="pt") as f:
            for name in names:
                state_dict[name] = f.get_tensor(name)
    return state_dict


def pack_shards(names: list[str], sizes: dict[str, int], budget: int) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in names:
        size = sizes[name]
        if size > budget:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
            continue
        if current and current_size + size > budget:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)
    return shards


def layer_sort_key(name: str):
    m = re.match(r"^model\.layers\.(\d+)\.", name)
    if m:
        return (1, int(m.group(1)), name)
    return (0, 0, name)


def main() -> None:
    state_dict = load_state_dict(INPUT_DIR)
    assert len(state_dict) == 114, f"expected 114 input tensors, got {len(state_dict)}"

    proj_names = {n for n in state_dict if PROJ_RE.match(n)}
    assert len(proj_names) == 112, f"expected 112 projection matrices, got {len(proj_names)}"

    # Sanity check: every input tensor is float32 to start with.
    for name, t in state_dict.items():
        assert t.dtype == torch.float32, f"unexpected input dtype for {name}: {t.dtype}"

    out_state: dict[str, torch.Tensor] = {}
    for name, t in state_dict.items():
        if name in proj_names:
            out_state[name] = t.to(torch.bfloat16).contiguous()
        else:
            out_state[name] = t.contiguous()

    # ---- Required checks (fail loudly before writing anything) ----
    bf16_count = sum(1 for t in out_state.values() if t.dtype == torch.bfloat16)
    assert bf16_count == 112, f"expected exactly 112 bfloat16 tensors, got {bf16_count}"

    q0 = "model.layers.0.self_attn.q_proj.weight"
    assert out_state[q0].dtype == torch.bfloat16, f"{q0} must be bfloat16"

    assert out_state["model.embed_tokens.weight"].dtype == torch.float32, (
        "model.embed_tokens.weight must be float32"
    )

    assert len(out_state) == 114, f"expected 114 output tensors, got {len(out_state)}"

    # Cross-check every non-projection tensor is untouched float32 and values
    # are bit-identical to the input.
    for name, t in out_state.items():
        if name not in proj_names:
            assert t.dtype == torch.float32, f"{name} must remain float32"
            assert torch.equal(t, state_dict[name]), f"{name} values changed unexpectedly"

    # Cross-check bfloat16 values match round-to-nearest-even casting exactly.
    for name in proj_names:
        expected = state_dict[name].to(torch.bfloat16)
        assert torch.equal(out_state[name], expected), f"{name} bfloat16 cast mismatch"

    # ---- Shard packing ----
    dtype_size = {torch.bfloat16: 2, torch.float32: 4}
    sizes = {name: t.numel() * dtype_size[t.dtype] for name, t in out_state.items()}

    ordered_names = sorted(out_state.keys(), key=layer_sort_key)
    shards = pack_shards(ordered_names, sizes, SHARD_BUDGET_BYTES)

    for shard_names in shards:
        shard_total = sum(sizes[n] for n in shard_names)
        if len(shard_names) > 1:
            assert shard_total <= SHARD_BUDGET_BYTES, (
                f"shard with {len(shard_names)} tensors exceeds budget: {shard_total} bytes"
            )

    # ---- Write output ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: out_state[name] for name in shard_names}
        save_file(shard_tensors, OUTPUT_DIR / shard_filename, metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += sizes[name]

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    assert len(weight_map) == 114, f"index has {len(weight_map)} entries, expected 114"

    # ---- Post-write verification: re-read everything back ----
    verify_names: set[str] = set()
    per_file: dict[str, list[str]] = {}
    for name, shard_file in weight_map.items():
        per_file.setdefault(shard_file, []).append(name)

    bf16_seen = 0
    for shard_file, names in per_file.items():
        with safe_open(OUTPUT_DIR / shard_file, framework="pt") as f:
            for name in names:
                t = f.get_tensor(name)
                assert t.shape == state_dict[name].shape, f"{name} shape mismatch on reload"
                if name in proj_names:
                    assert t.dtype == torch.bfloat16
                    bf16_seen += 1
                    assert torch.equal(t, state_dict[name].to(torch.bfloat16))
                else:
                    assert t.dtype == torch.float32
                    assert torch.equal(t, state_dict[name])
                verify_names.add(name)

    assert bf16_seen == 112
    assert len(verify_names) == 114

    print(f"Wrote {num_shards} shard(s) + index to {OUTPUT_DIR}")
    print(f"Total tensors: {len(weight_map)}, bfloat16: {bf16_seen}, float32: {114 - bf16_seen}")
    print(f"Total tensor bytes: {total_size}")


if __name__ == "__main__":
    main()
