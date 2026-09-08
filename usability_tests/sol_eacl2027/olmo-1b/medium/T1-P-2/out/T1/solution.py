import json
import os
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
INDEX = BASE / "model.safetensors.index.json"

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
Q_PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"
)
REMOVED = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in REMOVED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def fail(message):
    raise RuntimeError(message)


def main():
    with INDEX.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    # Group by shard so each sharded input file is opened only once.
    by_shard = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    output_tensors = {}
    seen_old_layers = set()
    removed_tensor_count = 0

    for shard, expected_names in sorted(by_shard.items()):
        with safe_open(BASE / shard, framework="pt", device="cpu") as source:
            actual_names = set(source.keys())
            expected_set = set(expected_names)
            if actual_names != expected_set:
                fail(f"Shard/index disagreement in {shard}")

            for old_name in expected_names:
                match = LAYER_RE.match(old_name)
                if match is None:
                    new_name = old_name
                else:
                    old_layer = int(match.group(1))
                    seen_old_layers.add(old_layer)
                    if old_layer in REMOVED:
                        removed_tensor_count += 1
                        continue
                    if old_layer not in RENUMBER:
                        fail(f"Unexpected source layer {old_layer}: {old_name}")
                    new_name = (
                        f"model.layers.{RENUMBER[old_layer]}.{match.group(2)}"
                    )

                if new_name in output_tensors:
                    fail(f"Rename collision at {new_name}")
                output_tensors[new_name] = source.get_tensor(old_name)

    if seen_old_layers != set(range(16)):
        fail(f"Expected source layers 0..15, found {sorted(seen_old_layers)}")
    if removed_tensor_count != 28:
        fail(f"Expected to remove 28 tensors, removed {removed_tensor_count}")

    # Required post-transformation checks, performed before any output is written.
    forbidden = [
        name
        for name in output_tensors
        if (match := LAYER_RE.match(name)) and int(match.group(1)) in {12, 13, 14, 15}
    ]
    if forbidden:
        fail(f"Forbidden output layers 12..15 remain: {forbidden[:3]}")

    q_proj_layers = sorted(
        int(match.group(1))
        for name in output_tensors
        if (match := Q_PROJ_RE.match(name))
    )
    if q_proj_layers != list(range(12)):
        fail(f"Expected exactly q_proj layers 0..11, found {q_proj_layers}")
    if len(output_tensors) != 86:
        fail(f"Expected 86 output tensors, found {len(output_tensors)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP_OUTPUT.unlink(missing_ok=True)
    save_file(output_tensors, TEMP_OUTPUT)
    os.replace(TEMP_OUTPUT, OUTPUT)
    print(f"Wrote {len(output_tensors)} tensors to {OUTPUT}")


if __name__ == "__main__":
    main()
