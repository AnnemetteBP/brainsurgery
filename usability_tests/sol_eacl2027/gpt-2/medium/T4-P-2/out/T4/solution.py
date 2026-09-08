from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LAMBDA = 0.4
EXPECTED_TENSOR_COUNT = 160


def expected_mlp_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(12):
        prefix = f"h.{layer}.mlp"
        shapes[f"{prefix}.c_fc.weight"] = (768, 3072)
        shapes[f"{prefix}.c_fc.bias"] = (3072,)
        shapes[f"{prefix}.c_proj.weight"] = (3072, 768)
        shapes[f"{prefix}.c_proj.bias"] = (768,)
    return shapes


def main() -> None:
    sandbox = Path(__file__).resolve().parents[2]
    paths = {
        "base": sandbox / "inputs/base/model.safetensors",
        "ft1": sandbox / "inputs/ft1/model.safetensors",
        "ft2": sandbox / "inputs/ft2/model.safetensors",
    }
    output_path = sandbox / "out/T4/model.safetensors"
    mlp_shapes = expected_mlp_shapes()
    mlp_names = set(mlp_shapes)
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected 48 distinct MLP tensor names, got {len(mlp_names)}")

    with (
        safe_open(paths["base"], framework="pt", device="cpu") as base_file,
        safe_open(paths["ft1"], framework="pt", device="cpu") as ft1_file,
        safe_open(paths["ft2"], framework="pt", device="cpu") as ft2_file,
    ):
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())

        # Validate the merge precondition completely before constructing or
        # writing any output tensors.
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        for name in sorted(base_names - mlp_names):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            if not torch.equal(base_tensor, ft1_tensor):
                raise RuntimeError(f"Fine-tune 1 changed frozen tensor {name}")
            if not torch.equal(base_tensor, ft2_tensor):
                raise RuntimeError(f"Fine-tune 2 changed frozen tensor {name}")

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            if name not in mlp_names:
                output[name] = base_tensor
                continue

            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            expected_shape = mlp_shapes[name]
            tensors = {"base": base_tensor, "ft1": ft1_tensor, "ft2": ft2_tensor}
            for checkpoint, tensor in tensors.items():
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected shape for {checkpoint}:{name}: "
                        f"{tuple(tensor.shape)} != {expected_shape}"
                    )
                if tensor.dtype != torch.float32:
                    raise RuntimeError(
                        f"Unexpected dtype for {checkpoint}:{name}: {tensor.dtype}"
                    )

            output[name] = (
                base_tensor
                + LAMBDA * (ft1_tensor - base_tensor)
                + LAMBDA * (ft2_tensor - base_tensor)
            )
            merged_count += 1

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
    if len(output) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TENSOR_COUNT} output tensors, got {len(output)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, output_path)

    with safe_open(output_path, framework="pt", device="cpu") as saved:
        saved_names = set(saved.keys())
    if len(saved_names) != EXPECTED_TENSOR_COUNT or saved_names != base_names:
        raise RuntimeError(
            "Saved checkpoint failed key/count verification: "
            f"count={len(saved_names)}, expected={EXPECTED_TENSOR_COUNT}"
        )

    print(f"Merged {merged_count} MLP tensors and saved {len(saved_names)} tensors to {output_path}")


if __name__ == "__main__":
    main()
