"""Prune four Pythia transformer blocks and compact the layer numbering."""

from pathlib import Path
import re

from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T1/model.safetensors")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
REMOVED_LAYERS = {2, 6, 10, 14}
SURVIVING_LAYERS = [i for i in range(16) if i not in REMOVED_LAYERS]
OLD_TO_NEW = {old: new for new, old in enumerate(SURVIVING_LAYERS)}


def main() -> None:
    source = load_file(str(INPUT_PATH), device="cpu")
    result = {}

    for old_name, tensor in source.items():
        match = LAYER_RE.match(old_name)
        if match is None:
            new_name = old_name
        else:
            old_layer = int(match.group(1))
            if old_layer in REMOVED_LAYERS:
                continue
            if old_layer not in OLD_TO_NEW:
                raise ValueError(f"unexpected input layer index: {old_layer}")
            new_name = f"gpt_neox.layers.{OLD_TO_NEW[old_layer]}.{match.group(2)}"

        if new_name in result:
            raise ValueError(f"renaming collision at {new_name}")
        result[new_name] = tensor

    output_layer_indices = {
        int(match.group(1))
        for name in result
        if (match := LAYER_RE.match(name)) is not None
    }
    forbidden = output_layer_indices & {12, 13, 14, 15}
    if forbidden:
        raise ValueError(f"forbidden output layer indices remain: {sorted(forbidden)}")
    if output_layer_indices != set(range(12)):
        raise ValueError(f"output layer indices are not exactly 0..11: {sorted(output_layer_indices)}")

    qkv_weight_re = re.compile(
        r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$"
    )
    qkv_layers = [
        int(match.group(1))
        for name in result
        if (match := qkv_weight_re.match(name)) is not None
    ]
    if len(qkv_layers) != 12 or set(qkv_layers) != set(range(12)):
        raise ValueError(f"expected one QKV weight for each of 12 blocks, got {sorted(qkv_layers)}")
    if len(result) != 184:
        raise ValueError(f"expected 184 output tensors, got {len(result)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
