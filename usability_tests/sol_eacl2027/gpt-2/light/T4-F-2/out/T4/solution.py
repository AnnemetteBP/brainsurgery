from pathlib import Path
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "base": ROOT / "inputs/base/model.safetensors",
    "ft1": ROOT / "inputs/ft1/model.safetensors",
    "ft2": ROOT / "inputs/ft2/model.safetensors",
}
OUTPUT = Path(__file__).with_name("model.safetensors")
TEMP_OUTPUT = Path(__file__).with_name("model.safetensors.tmp")
LAMBDA = 0.4


def mlp_spec() -> dict[str, tuple[int, ...]]:
    suffix_shapes = {
        "c_fc.weight": (768, 3072),
        "c_fc.bias": (3072,),
        "c_proj.weight": (3072, 768),
        "c_proj.bias": (768,),
    }
    return {
        f"h.{layer}.mlp.{suffix}": shape
        for layer in range(12)
        for suffix, shape in suffix_shapes.items()
    }


def main() -> None:
    expected_mlp = mlp_spec()
    if len(expected_mlp) != 48:
        raise RuntimeError(f"Internal error: expected 48 MLP names, got {len(expected_mlp)}")

    # Read-only preflight: validate the complete layouts and every frozen tensor
    # before constructing or writing any output state.
    with (
        safe_open(PATHS["base"], framework="pt", device="cpu") as base,
        safe_open(PATHS["ft1"], framework="pt", device="cpu") as ft1,
        safe_open(PATHS["ft2"], framework="pt", device="cpu") as ft2,
    ):
        key_sets = {name: set(handle.keys()) for name, handle in (("base", base), ("ft1", ft1), ("ft2", ft2))}
        if key_sets["base"] != key_sets["ft1"] or key_sets["base"] != key_sets["ft2"]:
            raise RuntimeError("Checkpoint tensor-name sets differ")
        if len(key_sets["base"]) != 160:
            raise RuntimeError(f"Expected 160 input tensors, got {len(key_sets['base'])}")
        missing = set(expected_mlp) - key_sets["base"]
        if missing:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing)}")

        for key in sorted(key_sets["base"]):
            tensors = (base.get_tensor(key), ft1.get_tensor(key), ft2.get_tensor(key))
            shapes = {tuple(t.shape) for t in tensors}
            dtypes = {t.dtype for t in tensors}
            if len(shapes) != 1 or len(dtypes) != 1:
                raise RuntimeError(f"Layout mismatch for {key}: shapes={shapes}, dtypes={dtypes}")
            if tensors[0].dtype != torch.float32:
                raise RuntimeError(f"Expected float32 tensor {key}, got {tensors[0].dtype}")
            if key in expected_mlp:
                if tuple(tensors[0].shape) != expected_mlp[key]:
                    raise RuntimeError(
                        f"Unexpected shape for {key}: {tuple(tensors[0].shape)} != {expected_mlp[key]}"
                    )
            elif not (torch.equal(tensors[0], tensors[1]) and torch.equal(tensors[0], tensors[2])):
                raise RuntimeError(f"Frozen tensor differs across checkpoints: {key}")

        result: dict[str, torch.Tensor] = {}
        merged_count = 0
        for key in sorted(key_sets["base"]):
            base_tensor = base.get_tensor(key)
            if key in expected_mlp:
                ft1_tensor = ft1.get_tensor(key)
                ft2_tensor = ft2.get_tensor(key)
                result[key] = base_tensor + LAMBDA * (ft1_tensor - base_tensor) + LAMBDA * (ft2_tensor - base_tensor)
                merged_count += 1
            else:
                result[key] = base_tensor

    if merged_count != 48:
        raise RuntimeError(f"Expected exactly 48 merged tensors, got {merged_count}")
    if len(result) != 160:
        raise RuntimeError(f"Expected exactly 160 output tensors, got {len(result)}")

    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()
    save_file(result, TEMP_OUTPUT)
    with safe_open(TEMP_OUTPUT, framework="pt", device="cpu") as written:
        written_keys = set(written.keys())
        if len(written_keys) != 160 or written_keys != set(result):
            raise RuntimeError("Serialized output failed the 160-tensor key-set check")
    os.replace(TEMP_OUTPUT, OUTPUT)
    print(f"Wrote {OUTPUT} with {len(result)} tensors; merged {merged_count} MLP tensors")


if __name__ == "__main__":
    main()
