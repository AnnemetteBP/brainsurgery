from pathlib import Path
import re

from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
LAYER_RE = re.compile(r"gpt_neox\.layers\.(\d+)\.(.+)")
DROPPED = {2, 6, 10, 14}


def main() -> None:
    tensors = {}
    source_layer_counts = {i: 0 for i in range(16)}

    with safe_open(INPUT, framework="pt", device="cpu") as checkpoint:
        source_keys = list(checkpoint.keys())
        if len(source_keys) != 244:
            raise RuntimeError(f"expected 244 input tensors, found {len(source_keys)}")

        for old_name in source_keys:
            match = LAYER_RE.fullmatch(old_name)
            if match is None:
                new_name = old_name
            else:
                old_layer = int(match.group(1))
                if old_layer not in source_layer_counts:
                    raise RuntimeError(f"unexpected source layer in {old_name}")
                source_layer_counts[old_layer] += 1
                if old_layer in DROPPED:
                    continue
                new_layer = old_layer - sum(dropped < old_layer for dropped in DROPPED)
                new_name = f"gpt_neox.layers.{new_layer}.{match.group(2)}"

            if new_name in tensors:
                raise RuntimeError(f"rename collision at {new_name}")
            tensors[new_name] = checkpoint.get_tensor(old_name)

    if any(count != 15 for count in source_layer_counts.values()):
        raise RuntimeError(f"expected 15 tensors in every source block: {source_layer_counts}")

    output_layers = []
    qkv_suffix = "attention.query_key_value.weight"
    for name in tensors:
        match = LAYER_RE.fullmatch(name)
        if match is not None and match.group(2) == qkv_suffix:
            output_layers.append(int(match.group(1)))

    forbidden = [
        name
        for name in tensors
        if (match := LAYER_RE.fullmatch(name)) is not None
        and int(match.group(1)) in {12, 13, 14, 15}
    ]
    if forbidden:
        raise RuntimeError(f"forbidden output layer tensors remain: {forbidden[:3]}")
    if len(output_layers) != 12 or sorted(output_layers) != list(range(12)):
        raise RuntimeError(f"expected exactly blocks 0..11, found QKV blocks {sorted(output_layers)}")
    if len(tensors) != 184:
        raise RuntimeError(f"expected 184 output tensors, found {len(tensors)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)
    print(f"Wrote {len(tensors)} tensors across blocks 0..11 to {OUTPUT}")


if __name__ == "__main__":
    main()
