"""T4: task-vector merge of two OLMo-1B fine-tunes into the base checkpoint.

    out[X] = base[X] + lam*(ft1[X] - base[X]) + lam*(ft2[X] - base[X])

for the 48 MLP tensors, every other tensor copied from the base verbatim.
Both task vectors are taken against the *unmodified* base.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
INPUTS = SANDBOX / "inputs"
OUT_FILE = HERE / "model.safetensors"

LAMBDA = 0.4
N_TOTAL = 114
N_MLP = 48
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact comparison (unlike ==, this is correct for NaN and -0.0)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class Ckpt:
    """Read-only view over one checkpoint, sharded or single-file."""

    def __init__(self, name: str, path: Path, stack: contextlib.ExitStack):
        self.name = name
        if path.is_dir():
            files = sorted(path.glob("*.safetensors"))
            if not files:
                die(f"{name}: no .safetensors under {path}")
            index = path / "model.safetensors.index.json"
            if index.exists():
                # Trust the index, but only as a cross-check on what the shards hold.
                declared = set(json.loads(index.read_text())["weight_map"])
            else:
                declared = None
        else:
            files = [path]
            declared = None

        self._handles = {}
        self.key_to_handle: dict[str, object] = {}
        for f in files:
            h = stack.enter_context(safe_open(str(f), framework="pt"))
            self._handles[f.name] = h
            for k in h.keys():
                if k in self.key_to_handle:
                    die(f"{name}: tensor {k!r} appears in more than one shard")
                self.key_to_handle[k] = h

        if declared is not None and declared != set(self.key_to_handle):
            die(
                f"{name}: index.json disagrees with the shards "
                f"(only in index: {sorted(declared - set(self.key_to_handle))[:5]}, "
                f"only in shards: {sorted(set(self.key_to_handle) - declared)[:5]})"
            )

    def keys(self) -> set[str]:
        return set(self.key_to_handle)

    def get(self, key: str) -> torch.Tensor:
        return self.key_to_handle[key].get_tensor(key)


def main() -> None:
    with contextlib.ExitStack() as stack:
        base = Ckpt("base", INPUTS / "base", stack)
        ft1 = Ckpt("ft1", INPUTS / "ft1" / "model.safetensors", stack)
        ft2 = Ckpt("ft2", INPUTS / "ft2" / "model.safetensors", stack)

        # --- step 1a: identical tensor name sets ------------------------------
        kb, k1, k2 = base.keys(), ft1.keys(), ft2.keys()
        for other, ko in (("ft1", k1), ("ft2", k2)):
            if ko != kb:
                die(
                    f"tensor names differ between base and {other}: "
                    f"missing from {other}: {sorted(kb - ko)[:5]}, "
                    f"extra in {other}: {sorted(ko - kb)[:5]}"
                )
        if len(kb) != N_TOTAL:
            die(f"expected {N_TOTAL} tensors per checkpoint, found {len(kb)}")
        print(f"[1/4] name sets agree across the three checkpoints ({len(kb)} tensors)")

        # --- identify the MLP tensors ----------------------------------------
        mlp = {k for k in kb if MLP_RE.match(k)}
        expected = {
            f"model.layers.{i}.mlp.{p}.weight"
            for i in range(16)
            for p in ("gate_proj", "up_proj", "down_proj")
        }
        if mlp != expected:
            die(
                f"MLP tensor set is not the expected 48 names "
                f"(unexpected: {sorted(mlp - expected)[:5]}, missing: {sorted(expected - mlp)[:5]})"
            )
        if len(mlp) != N_MLP:
            die(f"expected {N_MLP} MLP tensors, matched {len(mlp)}")
        shared = sorted(kb - mlp)
        print(f"[2/4] {len(mlp)} MLP tensors to merge, {len(shared)} shared tensors to verify")

        # --- step 1b: every non-MLP tensor identical in all three -------------
        # Done in full before any merge is computed, and base copies are kept
        # so the output is built from the verified bytes.
        out: dict[str, torch.Tensor] = {}
        for key in shared:
            b = base.get(key)
            for other, ck in (("ft1", ft1), ("ft2", ft2)):
                o = ck.get(key)
                if b.shape != o.shape or b.dtype != o.dtype:
                    die(
                        f"{key}: base is {tuple(b.shape)}/{b.dtype} but "
                        f"{other} is {tuple(o.shape)}/{o.dtype}"
                    )
                if not bit_identical(b, o):
                    die(
                        f"{key}: non-MLP tensor differs between base and {other}; "
                        "the frozen-backbone precondition does not hold, aborting"
                    )
                del o
            out[key] = b
        print(f"[3/4] all {len(shared)} non-MLP tensors are bit-identical in base, ft1 and ft2")

        # --- step 2: the merge ------------------------------------------------
        merged = 0
        for key in sorted(mlp):
            b = base.get(key)
            t1 = ft1.get(key)
            t2 = ft2.get(key)
            for other, o in (("ft1", t1), ("ft2", t2)):
                if b.shape != o.shape or b.dtype != o.dtype:
                    die(
                        f"{key}: base is {tuple(b.shape)}/{b.dtype} but "
                        f"{other} is {tuple(o.shape)}/{o.dtype}"
                    )
            b32 = b.to(torch.float32)
            res = b32 + LAMBDA * (t1.to(torch.float32) - b32) + LAMBDA * (t2.to(torch.float32) - b32)
            out[key] = res.to(b.dtype).contiguous()
            merged += 1
            del b, t1, t2, b32, res

        if merged != N_MLP:
            die(f"merged {merged} tensors, expected exactly {N_MLP}")
        if len(out) != N_TOTAL:
            die(f"output holds {len(out)} tensors, expected exactly {N_TOTAL}")

    # --- step 4: write ---------------------------------------------------------
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE))

    with safe_open(str(OUT_FILE), framework="pt") as h:
        written = set(h.keys())
    if len(written) != N_TOTAL:
        die(f"wrote {len(written)} tensors to {OUT_FILE}, expected exactly {N_TOTAL}")
    if written != set(out):
        die("the tensor names on disk do not match the ones that were merged")
    print(f"[4/4] wrote {len(written)} tensors ({merged} merged, lambda={LAMBDA}) to {OUT_FILE}")


if __name__ == "__main__":
    sys.exit(main())
