"""T4: task-vector merge of two OLMo-1B fine-tunes onto the base.

Plain torch + safetensors script: the required precondition (every non-MLP
tensor bit-identical across the three checkpoints) is not expressible in a
mergekit config, so the merge is written directly.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_MLP = 48
N_TOTAL = 114
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"


class CheckFailed(Exception):
    pass


def load_sharded(directory: Path) -> dict[str, torch.Tensor]:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(directory / shard, framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def load_flat(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main() -> None:
    base = load_sharded(BASE_DIR)
    ft1 = load_flat(FT1)
    ft2 = load_flat(FT2)

    # --- step 1: verification, before touching anything ---------------------
    if not (set(base) == set(ft1) == set(ft2)):
        only = lambda a, b: sorted(set(a) - set(b))[:5]  # noqa: E731
        raise CheckFailed(
            f"tensor name sets differ: base-only={only(base, ft1)} "
            f"ft1-only={only(ft1, base)} ft2-only={only(ft2, base)}"
        )
    if len(base) != N_TOTAL:
        raise CheckFailed(f"expected {N_TOTAL} tensors, found {len(base)}")

    mlp_keys = sorted(k for k in base if MLP_RE.match(k))
    if len(mlp_keys) != N_MLP:
        raise CheckFailed(f"expected {N_MLP} MLP tensors, matched {len(mlp_keys)}")

    for name in sorted(base):
        for other, tag in ((ft1, "ft1"), (ft2, "ft2")):
            if base[name].shape != other[name].shape or base[name].dtype != other[name].dtype:
                raise CheckFailed(
                    f"{name}: base {tuple(base[name].shape)}/{base[name].dtype} != "
                    f"{tag} {tuple(other[name].shape)}/{other[name].dtype}"
                )
        if name in set(mlp_keys):
            continue
        for other, tag in ((ft1, "ft1"), (ft2, "ft2")):
            if not torch.equal(base[name], other[name]):
                raise CheckFailed(f"shared tensor {name} differs between base and {tag}")
    print(f"verified: {len(base)} tensors, {N_TOTAL - N_MLP} shared tensors bit-identical")

    # --- step 2/3: merge; task vectors are taken against the pristine base ---
    result: dict[str, torch.Tensor] = {}
    merged = 0
    for name, tensor in base.items():
        if name in set(mlp_keys):
            b = tensor.to(torch.float32)
            merged_t = b + LAMBDA * (ft1[name].to(torch.float32) - b) \
                         + LAMBDA * (ft2[name].to(torch.float32) - b)
            result[name] = merged_t.to(tensor.dtype).contiguous()
            merged += 1
        else:
            result[name] = tensor.clone().contiguous()

    if merged != N_MLP:
        raise CheckFailed(f"merged {merged} tensors, expected {N_MLP}")
    if len(result) != N_TOTAL:
        raise CheckFailed(f"output has {len(result)} tensors, expected {N_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, str(OUT))

    # --- post-write check on the file that was actually written -------------
    written = load_flat(OUT)
    if len(written) != N_TOTAL:
        raise CheckFailed(f"written file has {len(written)} tensors, expected {N_TOTAL}")
    if set(written) != set(base):
        raise CheckFailed("written file key set differs from the base key set")
    print(f"wrote {OUT} : {len(written)} tensors, {merged} merged, lambda={LAMBDA}")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
