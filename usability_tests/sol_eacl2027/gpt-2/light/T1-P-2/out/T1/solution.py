from pathlib import Path
import re

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
REMOVED = {2, 5, 8}
SURVIVORS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")


def main() -> None:
    source = load_file(INPUT)
    assert len(source) == 160, f"expected 160 input tensors, found {len(source)}"

    block_counts = {i: 0 for i in range(12)}
    result = {}
    for name, tensor in source.items():
        match = BLOCK_KEY.fullmatch(name)
        if match is None:
            new_name = name
        else:
            old_index = int(match.group(1))
            assert old_index in block_counts, f"unexpected source block index: {old_index}"
            block_counts[old_index] += 1
            if old_index in REMOVED:
                continue
            new_name = f"h.{RENUMBER[old_index]}.{match.group(2)}"

        assert new_name not in result, f"rename collision at {new_name}"
        result[new_name] = tensor

    assert all(count == 13 for count in block_counts.values()), (
        f"expected 13 tensors per source block, found {block_counts}"
    )
    assert not any(
        re.match(r"^h\.(?:9|10|11)\.", name) for name in result
    ), "output still contains a tensor from block 9, 10, or 11"

    attention_weights = [
        name
        for name in result
        if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", name)
    ]
    assert len(attention_weights) == 9, (
        f"expected 9 block attention weights, found {len(attention_weights)}"
    )
    output_indices = {
        int(match.group(1))
        for name in result
        if (match := BLOCK_KEY.fullmatch(name)) is not None
    }
    assert output_indices == set(range(9)), (
        f"output block indices are not contiguous 0..8: {sorted(output_indices)}"
    )
    assert len(result) == 121, f"expected 121 output tensors, found {len(result)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, OUTPUT)


if __name__ == "__main__":
    main()
