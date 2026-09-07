#!/usr/bin/env python3
"""Independent direct-PyTorch checkpoint transforms for behavioral methodology v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from revision_tests.scaling.baseline import load_state, save_sharded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operation",
        choices=("structural-roundtrip", "cast-float32"),
        required=True,
    )
    parser.add_argument("--temporary-namespace", default="__structural_roundtrip")
    parser.add_argument("--shard-size-bytes", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_state(args.input.resolve())
    original_names = list(state)
    if args.operation == "structural-roundtrip":
        renamed = {f"{args.temporary_namespace}.{name}": tensor for name, tensor in state.items()}
        state = {
            name.removeprefix(f"{args.temporary_namespace}."): tensor
            for name, tensor in renamed.items()
        }
        if list(state) != original_names:
            raise RuntimeError("structural round trip did not restore the exact key set")
    else:
        state = {
            name: tensor.to(torch.float32) if tensor.is_floating_point() else tensor
            for name, tensor in state.items()
        }
    result = save_sharded(state, args.output.resolve(), args.shard_size_bytes)
    print(json.dumps(result | {"operation": args.operation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
