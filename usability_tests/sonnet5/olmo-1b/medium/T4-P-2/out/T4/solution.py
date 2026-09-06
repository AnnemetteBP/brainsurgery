"""T4: Task-vector merge of two fine-tunes (OLMo-1B-0724-hf).

Merges ft1 and ft2 into base by task arithmetic on the 48 MLP tensors:

    out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

with lambda = 0.4, computed in float32. Every other tensor is copied
unchanged from base. Fails loudly if the frozen-backbone assumption (base,
ft1, ft2 identical outside the 48 MLP tensors) does not hold.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
INPUTS = SANDBOX / "inputs"
OUT_DIR = HERE
LAMBDA = 0.4
NUM_LAYERS = 16

MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_base_state_dict(base_dir: Path) -> dict[str, torch.Tensor]:
    index_path = base_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        with safe_open(base_dir / shard_name, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    if set(state_dict.keys()) != set(weight_map.keys()):
        raise RuntimeError(
            "Base shard contents do not match model.safetensors.index.json weight_map."
        )
    return state_dict


def load_single_file_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def is_mlp_tensor(name: str) -> bool:
    match = MLP_RE.match(name)
    if match is None:
        return False
    layer = int(match.group(1))
    return 0 <= layer < NUM_LAYERS


def main() -> None:
    base = load_base_state_dict(INPUTS / "base")
    ft1 = load_single_file_state_dict(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_single_file_state_dict(INPUTS / "ft2" / "model.safetensors")

    # --- Step 1: verify same tensor names across all three checkpoints. ---
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())
    if not (base_keys == ft1_keys == ft2_keys):
        raise RuntimeError(
            "Tensor name mismatch between checkpoints: "
            f"base-only={base_keys - ft1_keys - ft2_keys}, "
            f"ft1-only={ft1_keys - base_keys}, "
            f"ft2-only={ft2_keys - base_keys}"
        )
    if len(base_keys) != 114:
        raise RuntimeError(f"Expected 114 tensors in base, found {len(base_keys)}.")

    mlp_names = {name for name in base_keys if is_mlp_tensor(name)}
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected 48 MLP tensors, found {len(mlp_names)}: {sorted(mlp_names)}")

    # Verify every non-MLP tensor is identical (dtype, shape, bit-exact values)
    # across base, ft1 and ft2 -- the frozen-backbone precondition.
    non_mlp_names = base_keys - mlp_names
    mismatched = []
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1):
            mismatched.append(f"{name}: differs between base and ft1")
        elif not torch.equal(b, f2):
            mismatched.append(f"{name}: differs between base and ft2")
    if mismatched:
        raise RuntimeError(
            "Frozen-backbone assumption violated -- non-MLP tensors differ "
            f"between checkpoints:\n" + "\n".join(mismatched)
        )

    # --- Step 2: merge the 48 MLP tensors via task arithmetic. ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in mlp_names:
        b = base[name].to(torch.float32)
        f1 = ft1[name].to(torch.float32)
        f2 = ft2[name].to(torch.float32)
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[name] = merged.to(base[name].dtype)
        merged_count += 1

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge exactly 48 tensors, merged {merged_count}.")

    # --- Step 3: every other tensor is taken from base unchanged. ---
    for name in non_mlp_names:
        out[name] = base[name].clone()

    # --- Step 4: write output. ---
    if len(out) != 114:
        raise RuntimeError(f"Expected output to have exactly 114 tensors, got {len(out)}.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "model.safetensors"
    save_file(out, str(out_path))
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {out_path}")


if __name__ == "__main__":
    main()
