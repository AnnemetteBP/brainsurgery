"""T2: structured attention-head pruning of Pythia-1B (remove head 5 in every layer).

Plain torch + safetensors script. Layout facts used (from TASK.md, verified
against the checkpoint below):
  - query_key_value.weight is [6144, 2048], nn.Linear [out, in], rows grouped
    per head in GPT-NeoX interleaved order: head h owns rows 768*h..768*h+767
    (q, then k, then v, 256 each). Dropping head 5 drops rows 3840..4607.
  - query_key_value.bias is [6144] with the same row layout.
  - dense.weight is [2048, 2048]; heads are 256-wide *column* blocks, so
    head 5 is columns 1280..1535.
Everything else (dense.bias, the attention buffers, MLP, embeddings, norms)
is copied through untouched, keys and dtypes unchanged.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 16
N_HEADS = 8
HEAD_DIM = 256
PRUNE_HEAD = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection

QKV_LO, QKV_HI = QKV_BLOCK * PRUNE_HEAD, QKV_BLOCK * (PRUNE_HEAD + 1)  # 3840, 4608
D_LO, D_HI = HEAD_DIM * PRUNE_HEAD, HEAD_DIM * (PRUNE_HEAD + 1)  # 1280, 1536

QKV_OUT = QKV_BLOCK * (N_HEADS - 1)  # 5376
D_IN = HEAD_DIM * (N_HEADS - 1)  # 1792
HIDDEN = HEAD_DIM * N_HEADS  # 2048


def drop_rows(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return torch.cat([t[:lo], t[hi:]], dim=0).contiguous()


def drop_cols(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return torch.cat([t[:, :lo], t[:, hi:]], dim=1).contiguous()


def main() -> None:
    sd = load_file(str(SRC))
    n_in = len(sd)

    out: dict[str, torch.Tensor] = dict(sd)
    touched = 0

    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        for k in (w, b, d):
            if k not in sd:
                raise KeyError(f"expected tensor missing from input: {k}")

        # Pre-conditions: the input must have the layout the slicing assumes.
        if tuple(sd[w].shape) != (QKV_BLOCK * N_HEADS, HIDDEN):
            raise ValueError(f"{w}: expected {(QKV_BLOCK * N_HEADS, HIDDEN)}, got {tuple(sd[w].shape)}")
        if tuple(sd[b].shape) != (QKV_BLOCK * N_HEADS,):
            raise ValueError(f"{b}: expected {(QKV_BLOCK * N_HEADS,)}, got {tuple(sd[b].shape)}")
        if tuple(sd[d].shape) != (HIDDEN, HIDDEN):
            raise ValueError(f"{d}: expected {(HIDDEN, HIDDEN)}, got {tuple(sd[d].shape)}")

        out[w] = drop_rows(sd[w], QKV_LO, QKV_HI)
        out[b] = drop_rows(sd[b], QKV_LO, QKV_HI)
        out[d] = drop_cols(sd[d], D_LO, D_HI)
        touched += 3

        # Per-layer post-conditions on every head-bearing projection.
        if tuple(out[w].shape) != (QKV_OUT, HIDDEN):
            raise AssertionError(f"{w}: got {tuple(out[w].shape)}, want {(QKV_OUT, HIDDEN)}")
        if tuple(out[b].shape) != (QKV_OUT,):
            raise AssertionError(f"{b}: got {tuple(out[b].shape)}, want {(QKV_OUT,)}")
        if tuple(out[d].shape) != (HIDDEN, D_IN):
            raise AssertionError(f"{d}: got {tuple(out[d].shape)}, want {(HIDDEN, D_IN)}")
        for k in (w, b, d):
            if out[k].dtype != sd[k].dtype:
                raise AssertionError(f"{k}: dtype changed {sd[k].dtype} -> {out[k].dtype}")

        # Value check: the kept pieces must be verbatim slices in the right order.
        if not torch.equal(out[w][:QKV_LO], sd[w][:QKV_LO]):
            raise AssertionError(f"{w}: leading block not preserved")
        if not torch.equal(out[w][QKV_LO:], sd[w][QKV_HI:]):
            raise AssertionError(f"{w}: trailing block not preserved")
        if not torch.equal(out[d][:, D_LO:], sd[d][:, D_HI:]):
            raise AssertionError(f"{d}: trailing column block not preserved")

    if touched != 3 * N_LAYERS:
        raise AssertionError(f"touched {touched} tensors, expected {3 * N_LAYERS}")

    # Untouched tensors must be bit-identical and same-keyed.
    if set(out) != set(sd):
        raise AssertionError("key set changed")
    for k in sd:
        if k in {
            f"gpt_neox.layers.{i}.attention.{s}"
            for i in range(N_LAYERS)
            for s in ("query_key_value.weight", "query_key_value.bias", "dense.weight")
        }:
            continue
        if out[k] is not sd[k] or out[k].shape != sd[k].shape:
            raise AssertionError(f"{k}: should have been left untouched")

    # Required checks from TASK.md, stated literally, before writing.
    required = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
        "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
        "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
    }
    for k, want in required.items():
        got = tuple(out[k].shape)
        if got != want:
            raise AssertionError(f"required check failed: {k} has shape {got}, want {want}")
    if len(out) != 244:
        raise AssertionError(f"required check failed: output has {len(out)} tensors, want 244")
    if n_in != 244:
        raise AssertionError(f"input had {n_in} tensors, want 244")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata={"format": "pt"})
    print(f"wrote {DST} with {len(out)} tensors ({touched} pruned, {len(out) - touched} copied)")


if __name__ == "__main__":
    main()
