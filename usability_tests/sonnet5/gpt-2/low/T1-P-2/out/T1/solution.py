import re
import sys

from safetensors.torch import load_file, save_file

INPUT_PATH = "inputs/base/model.safetensors"
OUTPUT_PATH = "out/T1/model.safetensors"

REMOVE_BLOCKS = {2, 5, 8}

BLOCK_RE = re.compile(r"^h\.(\d+)\.")


def main():
    tensors = load_file(INPUT_PATH)

    if len(tensors) != 160:
        print(f"FATAL: expected 160 input tensors, got {len(tensors)}", file=sys.stderr)
        sys.exit(1)

    surviving_old_indices = sorted(
        {int(m.group(1)) for k in tensors if (m := BLOCK_RE.match(k))} - REMOVE_BLOCKS
    )
    if surviving_old_indices != [0, 1, 3, 4, 6, 7, 9, 10, 11]:
        print(f"FATAL: unexpected surviving block indices {surviving_old_indices}", file=sys.stderr)
        sys.exit(1)

    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    output = {}
    for key, value in tensors.items():
        m = BLOCK_RE.match(key)
        if m is None:
            # non-block tensor, unchanged
            output[key] = value
            continue
        old_idx = int(m.group(1))
        if old_idx in REMOVE_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_key = f"h.{new_idx}." + key[m.end():]
        if new_key in output:
            print(f"FATAL: collision writing {new_key}", file=sys.stderr)
            sys.exit(1)
        output[new_key] = value.contiguous()

    # Required checks
    for bad in (9, 10, 11):
        if any(BLOCK_RE.match(k) and k.startswith(f"h.{bad}.") for k in output):
            print(f"FATAL: tensor of removed-index block {bad} present in output", file=sys.stderr)
            sys.exit(1)

    n_blocks = len({int(m.group(1)) for k in output if (m := BLOCK_RE.match(k))})
    if n_blocks != 9:
        print(f"FATAL: expected 9 surviving blocks, got {n_blocks}", file=sys.stderr)
        sys.exit(1)

    if len(output) != 121:
        print(f"FATAL: expected 121 output tensors, got {len(output)}", file=sys.stderr)
        sys.exit(1)

    save_file(output, OUTPUT_PATH)
    print(f"OK: wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
