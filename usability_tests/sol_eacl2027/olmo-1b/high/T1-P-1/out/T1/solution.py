#!/usr/bin/env python3
"""Prune four OLMo transformer blocks and compact the remaining layer names."""

import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_DIR = Path("inputs/base")
INDEX_PATH = BASE_DIR / "model.safetensors.index.json"
OUTPUT_PATH = Path("out/T1/model.safetensors")
TEMP_PATH = OUTPUT_PATH.with_suffix(".safetensors.tmp")

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
DROPPED_LAYERS = {2, 6, 10, 14}
SURVIVING_LAYERS = [i for i in range(16) if i not in DROPPED_LAYERS]
OLD_TO_NEW = {old: new for new, old in enumerate(SURVIVING_LAYERS)}
EXPECTED_SUFFIXES = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}
EXPECTED_NON_BLOCK_KEYS = {"model.embed_tokens.weight", "lm_head.weight"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    # Validate the source namespace before loading its large tensors.
    if len(weight_map) != 114:
        fail(f"expected 114 input tensors, found {len(weight_map)}")

    source_layers: dict[int, set[str]] = {}
    non_block_keys: set[str] = set()
    for key in weight_map:
        match = LAYER_RE.fullmatch(key)
        if match:
            source_layers.setdefault(int(match.group(1)), set()).add(match.group(2))
        else:
            non_block_keys.add(key)

    if set(source_layers) != set(range(16)):
        fail(f"expected source layers 0..15, found {sorted(source_layers)}")
    for layer, suffixes in source_layers.items():
        if suffixes != EXPECTED_SUFFIXES:
            fail(f"unexpected tensor set in source layer {layer}")
    if non_block_keys != EXPECTED_NON_BLOCK_KEYS:
        fail(f"unexpected non-block input keys: {sorted(non_block_keys)}")

    output_tensors: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        shard_handles = {
            shard: stack.enter_context(
                safe_open(BASE_DIR / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(weight_map.values()))
        }

        for old_key, shard in weight_map.items():
            match = LAYER_RE.fullmatch(old_key)
            if match:
                old_layer = int(match.group(1))
                if old_layer in DROPPED_LAYERS:
                    continue
                new_key = f"model.layers.{OLD_TO_NEW[old_layer]}.{match.group(2)}"
            else:
                new_key = old_key

            if new_key in output_tensors:
                fail(f"rename collision at {new_key}")
            tensor = shard_handles[shard].get_tensor(old_key)
            if tensor.dtype != torch.float32:
                fail(f"unexpected dtype for {old_key}: {tensor.dtype}")
            output_tensors[new_key] = tensor

        # Required post-transformation checks, performed before writing anything.
        output_layer_indices = {
            int(match.group(1))
            for key in output_tensors
            if (match := LAYER_RE.fullmatch(key))
        }
        forbidden = output_layer_indices & {12, 13, 14, 15}
        if forbidden:
            fail(f"forbidden output layers remain: {sorted(forbidden)}")
        if output_layer_indices != set(range(12)):
            fail(f"expected output layers 0..11, found {sorted(output_layer_indices)}")

        q_proj_keys = [
            key
            for key in output_tensors
            if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", key)
        ]
        if len(q_proj_keys) != 12:
            fail(f"expected 12 q_proj block markers, found {len(q_proj_keys)}")
        if len(output_tensors) != 86:
            fail(f"expected 86 output tensors, found {len(output_tensors)}")
        if not EXPECTED_NON_BLOCK_KEYS.issubset(output_tensors):
            fail("a non-block tensor was lost")

        # Save through a temporary path so a failed write cannot leave a partial result.
        try:
            TEMP_PATH.unlink(missing_ok=True)
            save_file(output_tensors, TEMP_PATH, metadata={"format": "pt"})
            os.replace(TEMP_PATH, OUTPUT_PATH)
        except Exception:
            TEMP_PATH.unlink(missing_ok=True)
            raise

    print(f"Wrote {len(output_tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
