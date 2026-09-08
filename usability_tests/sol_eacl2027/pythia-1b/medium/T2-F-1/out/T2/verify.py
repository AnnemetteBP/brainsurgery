"""Independently verify the serialized T2 checkpoint against its input."""

import torch
from safetensors import safe_open


SOURCE = "inputs/base/model.safetensors"
RESULT = "out/T2/model.safetensors"


with safe_open(SOURCE, framework="pt", device="cpu") as source, safe_open(
    RESULT, framework="pt", device="cpu"
) as result:
    source_keys = set(source.keys())
    result_keys = set(result.keys())
    assert source_keys == result_keys
    assert len(result_keys) == 244

    modified = set()
    for layer in range(16):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight = f"{prefix}.query_key_value.weight"
        qkv_bias = f"{prefix}.query_key_value.bias"
        dense_weight = f"{prefix}.dense.weight"
        modified.update((qkv_weight, qkv_bias, dense_weight))

        source_qkv_weight = source.get_tensor(qkv_weight)
        source_qkv_bias = source.get_tensor(qkv_bias)
        source_dense_weight = source.get_tensor(dense_weight)
        result_qkv_weight = result.get_tensor(qkv_weight)
        result_qkv_bias = result.get_tensor(qkv_bias)
        result_dense_weight = result.get_tensor(dense_weight)

        assert result_qkv_weight.shape == (5376, 2048)
        assert result_qkv_bias.shape == (5376,)
        assert result_dense_weight.shape == (2048, 1792)
        assert result_qkv_weight.dtype == source_qkv_weight.dtype == torch.float16
        assert result_qkv_bias.dtype == source_qkv_bias.dtype == torch.float16
        assert result_dense_weight.dtype == source_dense_weight.dtype == torch.float16
        assert torch.equal(
            result_qkv_weight,
            torch.cat((source_qkv_weight[:3840], source_qkv_weight[4608:])),
        )
        assert torch.equal(
            result_qkv_bias,
            torch.cat((source_qkv_bias[:3840], source_qkv_bias[4608:])),
        )
        assert torch.equal(
            result_dense_weight,
            torch.cat((source_dense_weight[:, :1280], source_dense_weight[:, 1536:]), dim=1),
        )

    for key in source_keys - modified:
        assert torch.equal(result.get_tensor(key), source.get_tensor(key)), key

print("PASS: 244 tensors; all 48 pruned tensors and 196 untouched tensors are exact")
