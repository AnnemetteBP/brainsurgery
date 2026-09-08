from pathlib import Path
import re

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
BLOCK_KEY = re.compile(r"h\.(\d+)\.(.+)")
REMOVED = {2, 5, 8}
SURVIVORS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def main() -> None:
    state = load_file(INPUT, device="cpu")
    output = {}

    for old_name, tensor in state.items():
        match = BLOCK_KEY.fullmatch(old_name)
        if match is None:
            new_name = old_name
        else:
            old_index = int(match.group(1))
            if old_index in REMOVED:
                continue
            if old_index not in RENUMBER:
                raise AssertionError(f"unexpected input block index: {old_index}")
            new_name = f"h.{RENUMBER[old_index]}.{match.group(2)}"

        if new_name in output:
            raise AssertionError(f"renaming collision at {new_name}")
        output[new_name] = tensor

    # Complete all required checks before creating the output file.
    remaining_indices = {
        int(match.group(1))
        for name in output
        if (match := BLOCK_KEY.fullmatch(name)) is not None
    }
    if remaining_indices & {9, 10, 11}:
        raise AssertionError("output still contains tensors from blocks 9, 10, or 11")

    attention_weights = [
        name
        for name in output
        if re.fullmatch(r"h\.(\d+)\.attn\.c_attn\.weight", name)
    ]
    if len(attention_weights) != 9:
        raise AssertionError(
            f"expected 9 block attention weights, found {len(attention_weights)}"
        )
    if remaining_indices != set(range(9)):
        raise AssertionError(f"block indices are not exactly 0..8: {remaining_indices}")
    if len(output) != 121:
        raise AssertionError(f"expected 121 tensors, found {len(output)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT)


if __name__ == "__main__":
    main()
