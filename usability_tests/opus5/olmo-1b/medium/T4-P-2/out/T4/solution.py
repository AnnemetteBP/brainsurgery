"""T4: task-vector merge of two OLMo-1B fine-tunes into the base checkpoint."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INPUTS = ROOT / "inputs"
OUT = HERE / "model.safetensors"

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
N_MLP = 48
N_TOTAL = 114


class Checkpoint:
    """Lazy reader over one checkpoint, sharded or single-file."""

    def __init__(self, path: Path):
        if path.is_dir():
            index = path / "model.safetensors.index.json"
            if index.exists():
                weight_map = json.loads(index.read_text())["weight_map"]
                self._shard_of = {k: path / v for k, v in weight_map.items()}
            else:
                shards = sorted(path.glob("*.safetensors"))
                if not shards:
                    raise FileNotFoundError(f"no safetensors under {path}")
                self._shard_of = {}
                for shard in shards:
                    with safe_open(shard, framework="pt") as f:
                        for k in f.keys():
                            self._shard_of[k] = shard
        else:
            self._shard_of = {}
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    self._shard_of[k] = path
        self.path = path
        self._handles: dict[Path, object] = {}

    def _handle(self, shard: Path):
        if shard not in self._handles:
            self._handles[shard] = safe_open(shard, framework="pt")
        return self._handles[shard]

    @property
    def names(self) -> set[str]:
        return set(self._shard_of)

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self._shard_of[name]).get_tensor(name)


def main() -> None:
    base = Checkpoint(INPUTS / "base")
    ft1 = Checkpoint(INPUTS / "ft1" / "model.safetensors")
    ft2 = Checkpoint(INPUTS / "ft2" / "model.safetensors")

    # --- step 1: same tensor names in all three checkpoints -------------------
    names = base.names
    for label, ckpt in (("ft1", ft1), ("ft2", ft2)):
        if ckpt.names != names:
            missing = sorted(names - ckpt.names)
            extra = sorted(ckpt.names - names)
            raise SystemExit(
                f"tensor names differ between base and {label}: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
    if len(names) != N_TOTAL:
        raise SystemExit(f"expected {N_TOTAL} tensors in the base, found {len(names)}")

    mlp_names = sorted(n for n in names if MLP_RE.match(n))
    if len(mlp_names) != N_MLP:
        raise SystemExit(f"expected {N_MLP} MLP tensors, matched {len(mlp_names)}: {mlp_names}")
    shared_names = sorted(names - set(mlp_names))

    # --- step 1 (cont.): every non-MLP tensor identical in all three ----------
    out: dict[str, torch.Tensor] = {}
    for name in shared_names:
        b = base.get(name)
        for label, ckpt in (("ft1", ft1), ("ft2", ft2)):
            t = ckpt.get(name)
            if t.shape != b.shape or t.dtype != b.dtype:
                raise SystemExit(
                    f"{name}: {label} has shape/dtype {tuple(t.shape)}/{t.dtype}, "
                    f"base has {tuple(b.shape)}/{b.dtype}"
                )
            if not torch.equal(t, b):
                raise SystemExit(
                    f"{name} is not identical in base and {label}, but it is outside the "
                    f"{N_MLP} MLP tensors; the frozen-backbone assumption does not hold"
                )
        out[name] = b.clone()

    # --- step 2: merge the task vectors, both taken against the pristine base -
    merged = 0
    for name in mlp_names:
        b = base.get(name)
        t1 = ft1.get(name)
        t2 = ft2.get(name)
        for label, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                raise SystemExit(
                    f"{name}: {label} has shape/dtype {tuple(t.shape)}/{t.dtype}, "
                    f"base has {tuple(b.shape)}/{b.dtype}"
                )
        if b.dtype != torch.float32:
            raise SystemExit(f"{name}: expected float32, found {b.dtype}")
        b32 = b.to(torch.float32)
        result = b32 + LAMBDA * (t1.to(torch.float32) - b32) + LAMBDA * (t2.to(torch.float32) - b32)
        out[name] = result.to(b.dtype).contiguous()
        merged += 1

    if merged != N_MLP:
        raise SystemExit(f"merged {merged} tensors, expected {N_MLP}")
    if len(out) != N_TOTAL:
        raise SystemExit(f"output has {len(out)} tensors, expected {N_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))

    # --- verify what actually landed on disk ---------------------------------
    with safe_open(OUT, framework="pt") as f:
        written = set(f.keys())
    if written != names:
        raise SystemExit("written tensor names do not match the base tensor names")
    if len(written) != N_TOTAL:
        raise SystemExit(f"written file has {len(written)} tensors, expected {N_TOTAL}")

    print(f"verified {len(shared_names)} shared tensors identical across base, ft1, ft2")
    print(f"merged {merged} MLP tensors with lambda={LAMBDA}")
    print(f"wrote {OUT} with {len(written)} tensors")


if __name__ == "__main__":
    main()
