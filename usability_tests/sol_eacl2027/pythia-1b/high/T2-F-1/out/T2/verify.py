import torch
from safetensors import safe_open


with safe_open("inputs/base/model.safetensors", framework="pt") as source, safe_open(
    "out/T2/model.safetensors", framework="pt"
) as output:
    source_keys = set(source.keys())
    output_keys = set(output.keys())
    assert source_keys == output_keys
    assert len(output_keys) == 244

    affected = set()
    for layer in range(16):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight = f"{prefix}.query_key_value.weight"
        qkv_bias = f"{prefix}.query_key_value.bias"
        dense_weight = f"{prefix}.dense.weight"
        affected.update((qkv_weight, qkv_bias, dense_weight))

        expected_qkv_weight = torch.cat(
            (source.get_tensor(qkv_weight)[:3840], source.get_tensor(qkv_weight)[4608:])
        )
        expected_qkv_bias = torch.cat(
            (source.get_tensor(qkv_bias)[:3840], source.get_tensor(qkv_bias)[4608:])
        )
        expected_dense_weight = torch.cat(
            (
                source.get_tensor(dense_weight)[:, :1280],
                source.get_tensor(dense_weight)[:, 1536:],
            ),
            dim=1,
        )
        assert torch.equal(output.get_tensor(qkv_weight), expected_qkv_weight)
        assert torch.equal(output.get_tensor(qkv_bias), expected_qkv_bias)
        assert torch.equal(output.get_tensor(dense_weight), expected_dense_weight)

    for name in source_keys - affected:
        assert torch.equal(output.get_tensor(name), source.get_tensor(name)), name

print("Verified exact slices for 48 pruned tensors and bit equality for all others")
