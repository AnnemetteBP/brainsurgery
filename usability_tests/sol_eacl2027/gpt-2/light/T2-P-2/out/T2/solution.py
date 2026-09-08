from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12


def main() -> None:
    tensors = load_file(INPUT)
    if len(tensors) != 160:
        raise AssertionError(f"expected 160 input tensors, got {len(tensors)}")

    # Within each 768-wide Q/K/V segment, retain all heads except head 5
    # (columns 320:384). These indices preserve the fused [q | k | v] order.
    qkv_keep = torch.cat(
        [
            torch.arange(base, base + 320)
            for base in (0, 768, 1536)
        ]
        + [
            torch.arange(base + 384, base + 768)
            for base in (0, 768, 1536)
        ]
    )
    # The construction above groups prefixes then suffixes; sort to restore
    # q-prefix, q-suffix, k-prefix, k-suffix, v-prefix, v-suffix order.
    qkv_keep = qkv_keep.sort().values
    proj_keep = torch.cat((torch.arange(0, 320), torch.arange(384, 768)))

    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        weight_key = f"{prefix}.c_attn.weight"
        bias_key = f"{prefix}.c_attn.bias"
        proj_key = f"{prefix}.c_proj.weight"

        if tuple(tensors[weight_key].shape) != (768, 2304):
            raise AssertionError(f"unexpected shape for {weight_key}: {tensors[weight_key].shape}")
        if tuple(tensors[bias_key].shape) != (2304,):
            raise AssertionError(f"unexpected shape for {bias_key}: {tensors[bias_key].shape}")
        if tuple(tensors[proj_key].shape) != (768, 768):
            raise AssertionError(f"unexpected shape for {proj_key}: {tensors[proj_key].shape}")

        tensors[weight_key] = tensors[weight_key].index_select(1, qkv_keep)
        tensors[bias_key] = tensors[bias_key].index_select(0, qkv_keep)
        tensors[proj_key] = tensors[proj_key].index_select(0, proj_keep)

        if tuple(tensors[weight_key].shape) != (768, 2112):
            raise AssertionError(f"bad output shape for {weight_key}")
        if tuple(tensors[bias_key].shape) != (2112,):
            raise AssertionError(f"bad output shape for {bias_key}")
        if tuple(tensors[proj_key].shape) != (704, 768):
            raise AssertionError(f"bad output shape for {proj_key}")

    # Required pre-write checks.
    assert tuple(tensors["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(tensors["h.0.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(tensors["h.0.attn.c_proj.weight"].shape) == (704, 768)
    assert len(tensors) == 160, f"expected 160 output tensors, got {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
