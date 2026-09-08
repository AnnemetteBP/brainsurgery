import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
INDEX = BASE / "model.safetensors.index.json"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
REMOVED = {2, 6, 10, 14}


def main() -> None:
    with INDEX.open() as f:
        weight_map = json.load(f)["weight_map"]

    source_layers = {
        int(match.group(1))
        for key in weight_map
        if (match := LAYER_RE.match(key))
    }
    if source_layers != set(range(16)) or len(weight_map) != 114:
        raise RuntimeError(
            f"unexpected input: layers={sorted(source_layers)}, tensors={len(weight_map)}"
        )

    survivors = sorted(source_layers - REMOVED)
    renumber = {old: new for new, old in enumerate(survivors)}
    output_sources = {}
    for old_key, shard in weight_map.items():
        match = LAYER_RE.match(old_key)
        if match:
            old_layer = int(match.group(1))
            if old_layer in REMOVED:
                continue
            new_key = f"model.layers.{renumber[old_layer]}.{match.group(2)}"
        else:
            new_key = old_key
        if new_key in output_sources:
            raise RuntimeError(f"rename collision at {new_key}")
        output_sources[new_key] = (shard, old_key)

    output_layers = {
        int(match.group(1))
        for key in output_sources
        if (match := LAYER_RE.match(key))
    }
    forbidden = output_layers & {12, 13, 14, 15}
    q_proj_layers = {
        int(match.group(1))
        for key in output_sources
        if (match := re.match(
            r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", key
        ))
    }
    if forbidden:
        raise RuntimeError(f"forbidden output block indices remain: {sorted(forbidden)}")
    if output_layers != set(range(12)) or q_proj_layers != set(range(12)):
        raise RuntimeError(
            f"expected exactly blocks 0..11 and 12 q_proj tensors; "
            f"got layers={sorted(output_layers)}, q_proj={sorted(q_proj_layers)}"
        )
    if len(output_sources) != 86:
        raise RuntimeError(f"expected 86 output tensors, got {len(output_sources)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()
    try:
        with ExitStack() as stack:
            shard_files = {
                name: stack.enter_context(
                    safe_open(BASE / name, framework="pt", device="cpu")
                )
                for name in sorted(set(weight_map.values()))
            }
            tensors = {
                new_key: shard_files[shard].get_tensor(old_key)
                for new_key, (shard, old_key) in output_sources.items()
            }
            save_file(tensors, TEMP_OUTPUT)
        os.replace(TEMP_OUTPUT, OUTPUT)
    finally:
        if TEMP_OUTPUT.exists():
            TEMP_OUTPUT.unlink()

    print(f"wrote {len(output_sources)} tensors across 12 layers to {OUTPUT}")


if __name__ == "__main__":
    main()
