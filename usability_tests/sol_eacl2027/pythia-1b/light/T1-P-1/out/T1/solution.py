import re
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
REMOVED = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in REMOVED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def main() -> None:
    source = load_file(INPUT)
    result = {}

    for name, tensor in source.items():
        match = LAYER_RE.match(name)
        if match is None:
            new_name = name
        else:
            old_index = int(match.group(1))
            if old_index in REMOVED:
                continue
            if old_index not in RENUMBER:
                raise ValueError(f"unexpected input layer index: {old_index}")
            new_name = f"gpt_neox.layers.{RENUMBER[old_index]}.{match.group(2)}"

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
        raise AssertionError(f"forbidden output layers remain: {sorted(forbidden)}")

    qkv_weights = [
        name
        for name in result
        if re.fullmatch(
            r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", name
        )
    ]
    if len(qkv_weights) != 12:
        raise AssertionError(f"expected 12 blocks, found {len(qkv_weights)}")
    if output_layer_indices != set(range(12)):
        raise AssertionError(
            f"output layer indices are not contiguous 0..11: {sorted(output_layer_indices)}"
        )
    if len(result) != 184:
        raise AssertionError(f"expected 184 tensors, found {len(result)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, OUTPUT)


if __name__ == "__main__":
    main()
