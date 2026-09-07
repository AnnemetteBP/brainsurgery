import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIG_BLOCKS = 16
NUM_KEPT_BLOCKS = 12

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def main():
    tensors = load_file(str(IN_PATH))

    # Build old->new index mapping for surviving blocks, in original order.
    survivors = [i for i in range(NUM_ORIG_BLOCKS) if i not in DROP_BLOCKS]
    if len(survivors) != NUM_KEPT_BLOCKS:
        raise RuntimeError(f"expected {NUM_KEPT_BLOCKS} surviving blocks, got {len(survivors)}")
    old_to_new = {old: new for new, old in enumerate(survivors)}

    out = {}
    for name, tensor in tensors.items():
        m = LAYER_RE.match(name)
        if m is None:
            # non-block tensor, unchanged
            out[name] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_name = f"gpt_neox.layers.{new_idx}." + name[m.end():]
        if new_name in out:
            raise RuntimeError(f"collision writing {new_name}")
        out[new_name] = tensor

    # Required checks
    for bad in (12, 13, 14, 15):
        for name in out:
            m = LAYER_RE.match(name)
            if m and int(m.group(1)) == bad:
                raise RuntimeError(f"tensor of dropped/out-of-range block {bad} remains: {name}")

    qkv_count = sum(
        1
        for name in out
        if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", name)
    )
    if qkv_count != NUM_KEPT_BLOCKS:
        raise RuntimeError(f"expected {NUM_KEPT_BLOCKS} blocks, found {qkv_count}")

    if len(out) != 184:
        raise RuntimeError(f"expected 184 tensors in output, got {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
