from pathlib import Path
import re

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
REMOVED = {2, 5, 8}
SURVIVORS = [i for i in range(12) if i not in REMOVED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")


def main() -> None:
    source = load_file(INPUT, device="cpu")
    result = {}

    for old_name, tensor in source.items():
        match = BLOCK_KEY.fullmatch(old_name)
        if match is None:
            new_name = old_name
        else:
            old_index = int(match.group(1))
            if old_index in REMOVED:
                continue
            if old_index not in RENUMBER:
                raise RuntimeError(f"unexpected source block index: {old_index}")
            new_name = f"h.{RENUMBER[old_index]}.{match.group(2)}"

        if new_name in result:
            raise RuntimeError(f"renaming collision at {new_name}")
        result[new_name] = tensor

    forbidden = [name for name in result if re.match(r"^h\.(?:9|10|11)\.", name)]
    if forbidden:
        raise RuntimeError(f"old block indices 9-11 remain after renumbering: {forbidden[:3]}")

    attention_weights = [
        name for name in result if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", name)
    ]
    if len(attention_weights) != 9:
        raise RuntimeError(
            f"expected 9 block attention weights, found {len(attention_weights)}"
        )

    block_indices = {
        int(match.group(1))
        for name in result
        if (match := BLOCK_KEY.fullmatch(name)) is not None
    }
    if block_indices != set(range(9)):
        raise RuntimeError(f"expected contiguous block indices 0..8, found {sorted(block_indices)}")

    if len(result) != 121:
        raise RuntimeError(f"expected 121 output tensors, found {len(result)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, OUTPUT)


if __name__ == "__main__":
    main()
