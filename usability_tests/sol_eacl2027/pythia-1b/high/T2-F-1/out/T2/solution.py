from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE = Path("inputs/base/model.safetensors")
DESTINATION = Path("out/T2/model.safetensors")
LAYERS = 16
EXPECTED_TENSORS = 244


def require_shape(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float16:
        raise ValueError(f"{name}: expected float16, got {tensor.dtype}")


def main() -> None:
    with safe_open(SOURCE, framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        if len(source_keys) != EXPECTED_TENSORS:
            raise ValueError(
                f"expected {EXPECTED_TENSORS} input tensors, got {len(source_keys)}"
            )

        tensors = {name: source.get_tensor(name) for name in source_keys}

        for layer in range(LAYERS):
            prefix = f"gpt_neox.layers.{layer}.attention"
            qkv_weight_name = f"{prefix}.query_key_value.weight"
            qkv_bias_name = f"{prefix}.query_key_value.bias"
            dense_weight_name = f"{prefix}.dense.weight"

            qkv_weight = tensors[qkv_weight_name]
            qkv_bias = tensors[qkv_bias_name]
            dense_weight = tensors[dense_weight_name]

            require_shape(qkv_weight, (6144, 2048), qkv_weight_name)
            require_shape(qkv_bias, (6144,), qkv_bias_name)
            require_shape(dense_weight, (2048, 2048), dense_weight_name)

            # Head 5 occupies one 768-row Q/K/V block and one 256-column
            # output block. Concatenation preserves the order of all kept heads.
            tensors[qkv_weight_name] = torch.cat(
                (qkv_weight[:3840], qkv_weight[4608:]), dim=0
            ).contiguous()
            tensors[qkv_bias_name] = torch.cat(
                (qkv_bias[:3840], qkv_bias[4608:]), dim=0
            ).contiguous()
            tensors[dense_weight_name] = torch.cat(
                (dense_weight[:, :1280], dense_weight[:, 1536:]), dim=1
            ).contiguous()

        # Required pre-write checks, plus equivalent checks for every layer.
        if len(tensors) != EXPECTED_TENSORS:
            raise ValueError(f"expected {EXPECTED_TENSORS} outputs, got {len(tensors)}")
        require_shape(
            tensors["gpt_neox.layers.0.attention.query_key_value.weight"],
            (5376, 2048),
            "gpt_neox.layers.0.attention.query_key_value.weight",
        )
        require_shape(
            tensors["gpt_neox.layers.0.attention.query_key_value.bias"],
            (5376,),
            "gpt_neox.layers.0.attention.query_key_value.bias",
        )
        require_shape(
            tensors["gpt_neox.layers.0.attention.dense.weight"],
            (2048, 1792),
            "gpt_neox.layers.0.attention.dense.weight",
        )
        for layer in range(LAYERS):
            prefix = f"gpt_neox.layers.{layer}.attention"
            require_shape(
                tensors[f"{prefix}.query_key_value.weight"],
                (5376, 2048),
                f"{prefix}.query_key_value.weight",
            )
            require_shape(
                tensors[f"{prefix}.query_key_value.bias"],
                (5376,),
                f"{prefix}.query_key_value.bias",
            )
            require_shape(
                tensors[f"{prefix}.dense.weight"],
                (2048, 1792),
                f"{prefix}.dense.weight",
            )

        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, DESTINATION, metadata=source.metadata())

    # Confirm that the serialized artifact itself has the required inventory.
    with safe_open(DESTINATION, framework="pt", device="cpu") as output:
        output_keys = list(output.keys())
        if len(output_keys) != EXPECTED_TENSORS:
            raise ValueError(
                f"written checkpoint has {len(output_keys)} tensors, "
                f"expected {EXPECTED_TENSORS}"
            )
        if set(output_keys) != set(source_keys):
            raise ValueError("written checkpoint key set differs from the input")

    print(f"Wrote {DESTINATION} with {EXPECTED_TENSORS} tensors")


if __name__ == "__main__":
    main()
