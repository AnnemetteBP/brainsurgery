"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks and renumber contiguously."""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP = {2, 5, 8}
N_BLOCKS_IN = 12
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    with safe_open(str(SRC), framework="pt") as f:
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    if len(tensors) != 160:
        die(f"expected 160 input tensors, got {len(tensors)}")

    # old block index -> new block index, in original order, gaps closed
    keep = [i for i in range(N_BLOCKS_IN) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}

    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        m = BLOCK_RE.match(name)
        if m is None:
            out[name] = tensor  # wte / wpe / ln_f
            continue
        old = int(m.group(1))
        if old in DROP:
            continue
        new_name = f"h.{remap[old]}.{m.group(2)}"
        if new_name in out:
            die(f"renumbering collision on {new_name}")
        out[new_name] = tensor

    # --- required checks -------------------------------------------------
    surviving = sorted({int(m.group(1)) for m in map(BLOCK_RE.match, out) if m})
    stale = [i for i in surviving if i >= 9]
    if stale:
        die(f"tensors of blocks {stale} still present")
    if surviving != list(range(9)):
        die(f"block indices are not 0..8 contiguous: {surviving}")

    n_attn = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    if n_attn != 9:
        die(f"expected 9 blocks (h.<i>.attn.c_attn.weight), got {n_attn}")

    if len(out) != 121:
        die(f"expected 121 output tensors, got {len(out)}")

    # every surviving block must be complete and value-identical to its source
    for new_name, tensor in out.items():
        m = BLOCK_RE.match(new_name)
        src_name = new_name if m is None else f"h.{keep[int(m.group(1))]}.{m.group(2)}"
        src = tensors.get(src_name)
        if src is None:
            die(f"output tensor {new_name} has no source {src_name}")
        if src.shape != tensor.shape or src.dtype != tensor.dtype:
            die(f"shape/dtype drift on {new_name}")
        if not torch.equal(src, tensor):
            die(f"value drift on {new_name}")

    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if name not in out:
            die(f"non-block tensor {name} missing from output")

    # --- write -----------------------------------------------------------
    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(DST), metadata=metadata)

    with safe_open(str(DST), framework="pt") as f:
        written = list(f.keys())
    if len(written) != 121:
        DST.unlink()
        die(f"written file has {len(written)} tensors, expected 121")

    cfg_path = SANDBOX / "inputs" / "base" / "config.json"
    n_layer = json.loads(cfg_path.read_text()).get("n_layer")
    print(f"input config n_layer={n_layer} -> output has 9 blocks, {len(written)} tensors")
    print(f"dropped blocks {sorted(DROP)}; remap {remap}")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
