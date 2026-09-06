"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T3")
MAX_SHARD_BYTES = 256 * 1024 * 1024

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_state_dict() -> dict[str, torch.Tensor]:
    index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)
    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        with safe_open(IN_DIR / shard, framework="pt") as f:
            for name in names:
                state[name] = f.get_tensor(name)
    return {k: state[k] for k in sorted(state)}


def plan_shards(state: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedy contiguous packing; a tensor over the budget gets its own shard."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, tensor in state.items():
        nbytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
        if current_bytes >= MAX_SHARD_BYTES:
            groups.append(current)
            current, current_bytes = [], 0
    if current:
        groups.append(current)
    return groups


def main() -> None:
    state = load_state_dict()
    if len(state) != 114:
        raise SystemExit(f"expected 114 input tensors, got {len(state)}")

    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if PROJ_RE.match(name):
            out[name] = tensor.to(torch.bfloat16)
        else:
            if tensor.dtype != torch.float32:
                out[name] = tensor.to(torch.float32)
            else:
                out[name] = tensor

    # Required checks: fail loudly before writing anything.
    bf16 = [k for k, v in out.items() if v.dtype == torch.bfloat16]
    if len(bf16) != 112:
        raise SystemExit(f"expected 112 bfloat16 tensors, got {len(bf16)}")
    if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        raise SystemExit("model.layers.0.self_attn.q_proj.weight is not bfloat16")
    if out["model.embed_tokens.weight"].dtype != torch.float32:
        raise SystemExit("model.embed_tokens.weight is not float32")
    if len(out) != 114:
        raise SystemExit(f"expected 114 output tensors, got {len(out)}")
    if set(out) != set(state):
        raise SystemExit("tensor names changed")
    for k, v in out.items():
        if v.shape != state[k].shape:
            raise SystemExit(f"shape changed for {k}")
        if v.dtype not in (torch.bfloat16, torch.float32):
            raise SystemExit(f"unexpected dtype for {k}: {v.dtype}")

    groups = plan_shards(out)
    n = len(groups)
    for i, names in enumerate(groups, start=1):
        total = sum(out[k].numel() * out[k].element_size() for k in names)
        if total > MAX_SHARD_BYTES and len(names) > 1:
            raise SystemExit(f"shard {i} over budget: {total}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.safetensors"):
        stale.unlink()

    weight_map: dict[str, str] = {}
    total_size = 0
    for i, names in enumerate(groups, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {k: out[k].contiguous().clone() for k in names}
        save_file(tensors, OUT_DIR / fname, metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname
            total_size += out[k].numel() * out[k].element_size()

    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
        + "\n"
    )
    print(f"wrote {len(weight_map)} tensors into {n} shards ({total_size} bytes)")


if __name__ == "__main__":
    main()
