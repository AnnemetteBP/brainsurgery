import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_PATH = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_PATH = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_PATH = Path(__file__).with_name("model.safetensors")
SCALE = 0.4


def base_weight_map():
    index_path = BASE_DIR / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("Base index has no valid weight_map")
    return weight_map


def main():
    weight_map = base_weight_map()
    base_keys = set(weight_map)
    mlp_keys = {
        f"model.layers.{layer}.mlp.{projection}_proj.weight"
        for layer in range(16)
        for projection in ("gate", "up", "down")
    }
    if len(mlp_keys) != 48 or not mlp_keys <= base_keys:
        missing = sorted(mlp_keys - base_keys)
        raise RuntimeError(f"Expected exactly 48 MLP tensors in base; missing={missing}")

    shard_names = sorted(set(weight_map.values()))
    with safe_open(FT1_PATH, framework="pt", device="cpu") as ft1, \
         safe_open(FT2_PATH, framework="pt", device="cpu") as ft2:
        ft1_keys = set(ft1.keys())
        ft2_keys = set(ft2.keys())
        if base_keys != ft1_keys or base_keys != ft2_keys:
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base_only_vs_ft1={sorted(base_keys - ft1_keys)}, "
                f"ft1_only={sorted(ft1_keys - base_keys)}, "
                f"base_only_vs_ft2={sorted(base_keys - ft2_keys)}, "
                f"ft2_only={sorted(ft2_keys - base_keys)}"
            )

        # Complete the frozen-backbone verification before constructing output.
        for shard_name in shard_names:
            shard_keys = sorted(k for k, v in weight_map.items() if v == shard_name)
            with safe_open(BASE_DIR / shard_name, framework="pt", device="cpu") as base:
                for key in shard_keys:
                    if key in mlp_keys:
                        continue
                    base_tensor = base.get_tensor(key)
                    if not torch.equal(base_tensor, ft1.get_tensor(key)):
                        raise RuntimeError(f"Fine-tune 1 changed shared tensor {key}")
                    if not torch.equal(base_tensor, ft2.get_tensor(key)):
                        raise RuntimeError(f"Fine-tune 2 changed shared tensor {key}")

        output = {}
        merged_count = 0
        for shard_name in shard_names:
            shard_keys = sorted(k for k, v in weight_map.items() if v == shard_name)
            with safe_open(BASE_DIR / shard_name, framework="pt", device="cpu") as base:
                for key in shard_keys:
                    base_tensor = base.get_tensor(key)
                    if key in mlp_keys:
                        one = ft1.get_tensor(key)
                        two = ft2.get_tensor(key)
                        if not (base_tensor.dtype == one.dtype == two.dtype == torch.float32):
                            raise RuntimeError(f"MLP tensor {key} is not float32 in all checkpoints")
                        if not (base_tensor.shape == one.shape == two.shape):
                            raise RuntimeError(f"MLP tensor {key} has inconsistent shapes")
                        output[key] = (
                            base_tensor
                            + SCALE * (one - base_tensor)
                            + SCALE * (two - base_tensor)
                        )
                        merged_count += 1
                    else:
                        output[key] = base_tensor

        if merged_count != 48:
            raise RuntimeError(f"Merged {merged_count} tensors, expected 48")
        if len(output) != 114:
            raise RuntimeError(f"Output has {len(output)} tensors, expected 114")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved:
        if len(saved.keys()) != 114:
            raise RuntimeError(f"Saved output has {len(saved.keys())} tensors, expected 114")
    print(f"Wrote {OUTPUT_PATH} with 114 tensors; merged 48 MLP tensors")


if __name__ == "__main__":
    main()
