"""T2: prune attention head 5 from every layer of Pythia-1B at the checkpoint level.

Plain safetensors + torch. Uses exact tensor names (no regex) so nothing else
can be touched, slices with torch.cat so every saved tensor is contiguous, and
enforces the required checks before writing.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 16
N_HEADS = 8
HEAD_DIM = 256
PRUNE_HEAD = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the interleaved fused projection

HIDDEN = N_HEADS * HEAD_DIM
QKV_OUT = N_HEADS * QKV_BLOCK
EXPECTED_QKV_ROWS = (N_HEADS - 1) * QKV_BLOCK  # 5376
EXPECTED_DENSE_COLS = (N_HEADS - 1) * HEAD_DIM  # 1792


def drop_block(t: torch.Tensor, dim: int, start: int, stop: int) -> torch.Tensor:
    """Return t with indices [start, stop) removed along dim, as a contiguous tensor."""
    before = t.narrow(dim, 0, start)
    after = t.narrow(dim, stop, t.shape[dim] - stop)
    return torch.cat([before, after], dim=dim).contiguous()


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"refusing to overwrite existing output {DST}")

    qkv_start, qkv_stop = PRUNE_HEAD * QKV_BLOCK, (PRUNE_HEAD + 1) * QKV_BLOCK  # 3840..4608
    dense_start, dense_stop = PRUNE_HEAD * HEAD_DIM, (PRUNE_HEAD + 1) * HEAD_DIM  # 1280..1536

    out: dict[str, torch.Tensor] = {}
    with safe_open(str(SRC), framework="pt") as f:
        keys = list(f.keys())
        metadata = f.metadata()
        if len(keys) != 244:
            raise AssertionError(f"input has {len(keys)} tensors, expected 244")

        touched: set[str] = set()
        for i in range(N_LAYERS):
            p = f"gpt_neox.layers.{i}.attention."
            touched |= {p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"}

        for k in keys:
            t = f.get_tensor(k)
            if k in touched:
                orig_dtype = t.dtype
                if k.endswith("query_key_value.weight"):
                    assert tuple(t.shape) == (QKV_OUT, HIDDEN), (k, t.shape)
                    t = drop_block(t, 0, qkv_start, qkv_stop)
                    assert tuple(t.shape) == (EXPECTED_QKV_ROWS, HIDDEN), (k, t.shape)
                elif k.endswith("query_key_value.bias"):
                    assert tuple(t.shape) == (QKV_OUT,), (k, t.shape)
                    t = drop_block(t, 0, qkv_start, qkv_stop)
                    assert tuple(t.shape) == (EXPECTED_QKV_ROWS,), (k, t.shape)
                elif k.endswith("dense.weight"):
                    assert tuple(t.shape) == (HIDDEN, HIDDEN), (k, t.shape)
                    t = drop_block(t, 1, dense_start, dense_stop)
                    assert tuple(t.shape) == (HIDDEN, EXPECTED_DENSE_COLS), (k, t.shape)
                else:
                    raise AssertionError(f"unexpected touched key {k}")
                assert t.dtype == orig_dtype, (k, t.dtype, orig_dtype)
            out[k] = t.contiguous()

        missing = touched - set(keys)
        if missing:
            raise AssertionError(f"head-bearing tensors missing from input: {sorted(missing)}")

    # Required checks (TASK.md) before writing.
    l0 = "gpt_neox.layers.0.attention."
    checks = {
        l0 + "query_key_value.weight": (5376, 2048),
        l0 + "query_key_value.bias": (5376,),
        l0 + "dense.weight": (2048, 1792),
    }
    for k, shape in checks.items():
        got = tuple(out[k].shape)
        if got != shape:
            raise AssertionError(f"{k}: shape {got}, expected {shape}")
    if len(out) != 244:
        raise AssertionError(f"output has {len(out)} tensors, expected 244")

    # Untouched tensors must be identical to the input (name, shape, dtype, bits).
    with safe_open(str(SRC), framework="pt") as f:
        for k in keys:
            if k in touched:
                continue
            src = f.get_tensor(k)
            if src.shape != out[k].shape or src.dtype != out[k].dtype:
                raise AssertionError(f"{k}: untouched tensor changed shape/dtype")
            if not torch.equal(src.view(torch.uint8) if src.ndim else src, out[k].view(torch.uint8) if out[k].ndim else out[k]):
                raise AssertionError(f"{k}: untouched tensor changed values")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata=metadata)

    # Post-write verification of the file on disk.
    with safe_open(str(DST), framework="pt") as f:
        wkeys = list(f.keys())
        if len(wkeys) != 244 or set(wkeys) != set(keys):
            raise AssertionError("written key set differs from input key set")
        for k, shape in checks.items():
            got = tuple(f.get_slice(k).get_shape())
            if got != shape:
                raise AssertionError(f"written {k}: shape {got}, expected {shape}")
        # Spot-check the reassembly on layer 0: kept rows/cols equal the source slices.
        with safe_open(str(SRC), framework="pt") as g:
            for i in range(N_LAYERS):
                p = f"gpt_neox.layers.{i}.attention."
                sw, dw = g.get_tensor(p + "query_key_value.weight"), f.get_tensor(p + "query_key_value.weight")
                assert torch.equal(dw[:qkv_start], sw[:qkv_start]) and torch.equal(dw[qkv_start:], sw[qkv_stop:])
                sb, db = g.get_tensor(p + "query_key_value.bias"), f.get_tensor(p + "query_key_value.bias")
                assert torch.equal(db[:qkv_start], sb[:qkv_start]) and torch.equal(db[qkv_start:], sb[qkv_stop:])
                sd, dd = g.get_tensor(p + "dense.weight"), f.get_tensor(p + "dense.weight")
                assert torch.equal(dd[:, :dense_start], sd[:, :dense_start]) and torch.equal(
                    dd[:, dense_start:], sd[:, dense_stop:]
                )
    print(f"OK: wrote {DST} with {len(out)} tensors; pruned head {PRUNE_HEAD} in {N_LAYERS} layers")


if __name__ == "__main__":
    main()
