"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks with renumbering."""

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent / "inputs" / "base"
OUT = HERE / "model.safetensors"

DROP = {2, 6, 10, 14}
N_LAYERS_IN = 16
BLOCK_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    index_path = BASE / "model.safetensors.index.json"
    if not index_path.is_file():
        fail(f"missing index {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    # old block index -> new block index, surviving blocks in original order
    keep = [i for i in range(N_LAYERS_IN) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}
    if len(keep) != 12:
        fail(f"expected 12 surviving blocks, got {len(keep)}")

    # group by shard so each file is opened once
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    if len(weight_map) != 114:
        fail(f"expected 114 input tensors, got {len(weight_map)}")

    out: dict = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(BASE / shard), framework="pt", device="cpu") as f:
            for name in names:
                m = BLOCK_RE.match(name)
                if m is None:
                    new_name = name  # embed_tokens / lm_head
                else:
                    old = int(m.group(1))
                    if old in DROP:
                        continue
                    if old not in remap:
                        fail(f"unexpected block index {old} in {name}")
                    new_name = f"model.layers.{remap[old]}.{m.group(2)}"
                if new_name in out:
                    fail(f"destination collision on {new_name}")
                out[new_name] = f.get_tensor(name).contiguous()

    # ---- required checks ----
    stale = sorted(
        n
        for n in out
        if (m := BLOCK_RE.match(n)) and int(m.group(1)) in (12, 13, 14, 15)
    )
    if stale:
        fail(f"tensors of blocks 12..15 remain: {stale[:5]}")

    q = sorted(
        int(m.group(1))
        for n in out
        if (m := re.match(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", n))
    )
    if q != list(range(12)):
        fail(f"expected q_proj for blocks 0..11, got {q}")

    blocks = sorted({int(m.group(1)) for n in out if (m := BLOCK_RE.match(n))})
    if blocks != list(range(12)):
        fail(f"expected exactly 12 contiguous blocks, got {blocks}")

    if len(out) != 86:
        fail(f"expected 86 output tensors, got {len(out)}")

    nonblock = sorted(n for n in out if not BLOCK_RE.match(n))
    if nonblock != ["lm_head.weight", "model.embed_tokens.weight"]:
        fail(f"unexpected non-block tensors: {nonblock}")

    save_file(out, str(OUT), metadata={"format": "pt"})
    print(f"wrote {OUT} with {len(out)} tensors, {len(blocks)} blocks")


main()
