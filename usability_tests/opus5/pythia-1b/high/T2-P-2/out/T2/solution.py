"""T2: structured attention-head pruning for Pythia-1B (GPT-NeoX).

Removes head 5 of 8 from every layer at the checkpoint level:

  * attention.query_key_value.weight/bias: the fused projection is interleaved
    per head, head h owning rows 768*h .. 768*h+767 (its q, then k, then v).
    Dropping head 5 means dropping rows 3840..4607 -> [5376, 2048] / [5376].
  * attention.dense.weight: the output projection consumes heads as 256-wide
    column blocks, so head 5 is columns 1280..1535 -> [2048, 1792].

Everything else (dense.bias, the attention buffers, the MLP and embedding
tensors) is copied through untouched, names unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T2")
OUT_PATH = OUT_DIR / "model.safetensors"

NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
PRUNE_HEAD = 5

QKV_PER_HEAD = 3 * HEAD_DIM  # 768 rows: q, k, v of one head
QKV_ROWS = NUM_HEADS * QKV_PER_HEAD  # 6144
KEPT_HEADS = NUM_HEADS - 1  # 7

EXPECTED_TENSORS = 244
EXPECTED_LAYERS = 16

QKV_RE = re.compile(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.(weight|bias)$")
DENSE_W_RE = re.compile(r"^gpt_neox\.layers\.\d+\.attention\.dense\.weight$")


def kept_qkv_rows() -> torch.Tensor:
    """Row indices of the fused qkv projection that survive, in order."""
    start = PRUNE_HEAD * QKV_PER_HEAD
    stop = start + QKV_PER_HEAD
    return torch.cat([torch.arange(0, start), torch.arange(stop, QKV_ROWS)])


def kept_dense_cols() -> torch.Tensor:
    """Column indices of the output projection that survive, in order."""
    start = PRUNE_HEAD * HEAD_DIM
    stop = start + HEAD_DIM
    return torch.cat([torch.arange(0, start), torch.arange(stop, HIDDEN)])


def main() -> None:
    if not IN_PATH.is_file():
        raise SystemExit(f"input checkpoint not found: {IN_PATH}")

    qkv_rows = kept_qkv_rows()
    dense_cols = kept_dense_cols()
    if qkv_rows.numel() != KEPT_HEADS * QKV_PER_HEAD:
        raise AssertionError(f"kept qkv rows: {qkv_rows.numel()} != {KEPT_HEADS * QKV_PER_HEAD}")
    if dense_cols.numel() != KEPT_HEADS * HEAD_DIM:
        raise AssertionError(f"kept dense cols: {dense_cols.numel()} != {KEPT_HEADS * HEAD_DIM}")

    out: dict[str, torch.Tensor] = {}
    touched_qkv = 0
    touched_dense = 0

    with safe_open(str(IN_PATH), framework="pt") as f:
        metadata = f.metadata()
        keys = list(f.keys())
        if len(keys) != EXPECTED_TENSORS:
            raise AssertionError(f"input has {len(keys)} tensors, expected {EXPECTED_TENSORS}")

        for key in keys:
            t = f.get_tensor(key)
            dtype_in = t.dtype

            if QKV_RE.match(key):
                if t.shape[0] != QKV_ROWS:
                    raise AssertionError(f"{key}: dim0 {t.shape[0]} != {QKV_ROWS}")
                t = t[qkv_rows]
                touched_qkv += 1
            elif DENSE_W_RE.match(key):
                if tuple(t.shape) != (HIDDEN, HIDDEN):
                    raise AssertionError(f"{key}: shape {tuple(t.shape)} != {(HIDDEN, HIDDEN)}")
                t = t[:, dense_cols]
                touched_dense += 1

            if t.dtype != dtype_in:
                raise AssertionError(f"{key}: dtype changed {dtype_in} -> {t.dtype}")
            out[key] = t.contiguous()

    # Every layer contributed qkv.weight + qkv.bias, and one dense.weight.
    if touched_qkv != 2 * EXPECTED_LAYERS:
        raise AssertionError(f"matched {touched_qkv} qkv tensors, expected {2 * EXPECTED_LAYERS}")
    if touched_dense != EXPECTED_LAYERS:
        raise AssertionError(f"matched {touched_dense} dense weights, expected {EXPECTED_LAYERS}")

    # Required checks from TASK.md.
    checks = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (KEPT_HEADS * QKV_PER_HEAD, HIDDEN),
        "gpt_neox.layers.0.attention.query_key_value.bias": (KEPT_HEADS * QKV_PER_HEAD,),
        "gpt_neox.layers.0.attention.dense.weight": (HIDDEN, KEPT_HEADS * HEAD_DIM),
    }
    for key, want in checks.items():
        if key not in out:
            raise AssertionError(f"missing required tensor: {key}")
        got = tuple(out[key].shape)
        if got != want:
            raise AssertionError(f"{key}: shape {got} != {want}")
    if len(out) != EXPECTED_TENSORS:
        raise AssertionError(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    # Shape sanity on every layer, not just layer 0.
    for i in range(EXPECTED_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        for name, want in (
            (p + "query_key_value.weight", (KEPT_HEADS * QKV_PER_HEAD, HIDDEN)),
            (p + "query_key_value.bias", (KEPT_HEADS * QKV_PER_HEAD,)),
            (p + "dense.weight", (HIDDEN, KEPT_HEADS * HEAD_DIM)),
            (p + "dense.bias", (HIDDEN,)),
        ):
            got = tuple(out[name].shape)
            if got != want:
                raise AssertionError(f"{name}: shape {got} != {want}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH), metadata=metadata)

    # Read back and re-verify what was actually written.
    with safe_open(str(OUT_PATH), framework="pt") as f:
        written = list(f.keys())
        if len(written) != EXPECTED_TENSORS:
            raise AssertionError(f"wrote {len(written)} tensors, expected {EXPECTED_TENSORS}")
        for key, want in checks.items():
            got = tuple(f.get_slice(key).get_shape())
            if got != want:
                raise AssertionError(f"written {key}: shape {got} != {want}")

    print(f"wrote {OUT_PATH} with {len(written)} tensors")
    print(f"  pruned head {PRUNE_HEAD} of {NUM_HEADS} in {EXPECTED_LAYERS} layers")
    print(f"  qkv tensors sliced: {touched_qkv}, dense weights sliced: {touched_dense}")


if __name__ == "__main__":
    main()
