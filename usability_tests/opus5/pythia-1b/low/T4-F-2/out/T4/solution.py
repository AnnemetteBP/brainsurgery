"""T4: task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X]) for the 64 MLP
tensors, in float32, cast back to the base dtype. Everything else copied from
base bit-exactly. Fails loudly if the preconditions do not hold.
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAM = 0.4
N_LAYERS = 16
N_TOTAL = 244
N_MLP = 64

MLP_KEYS = {
    f"gpt_neox.layers.{i}.mlp.{proj}.{kind}"
    for i in range(N_LAYERS)
    for proj in ("dense_h_to_4h", "dense_4h_to_h")
    for kind in ("weight", "bias")
}


class CheckFailed(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise CheckFailed(msg)


def load(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main():
    base = load("inputs/base/model.safetensors")
    ft1 = load("inputs/ft1/model.safetensors")
    ft2 = load("inputs/ft2/model.safetensors")

    # 1. same tensor names in all three
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        missing = sorted(set(base) - set(sd))
        extra = sorted(set(sd) - set(base))
        check(not missing and not extra,
              f"{name} key set differs from base: missing={missing[:5]} extra={extra[:5]}")
    check(len(base) == N_TOTAL, f"base has {len(base)} tensors, expected {N_TOTAL}")

    # the 64 MLP names must actually exist
    absent = sorted(MLP_KEYS - set(base))
    check(not absent, f"expected MLP tensors absent from base: {absent[:5]}")
    check(len(MLP_KEYS) == N_MLP, f"MLP key set has {len(MLP_KEYS)} names, expected {N_MLP}")

    # shapes and dtypes must agree everywhere
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        for k, b in base.items():
            check(sd[k].shape == b.shape,
                  f"{name}[{k}] shape {tuple(sd[k].shape)} != base {tuple(b.shape)}")
            check(sd[k].dtype == b.dtype,
                  f"{name}[{k}] dtype {sd[k].dtype} != base {b.dtype}")

    # 1b. every tensor outside the 64 MLP tensors is identical in all three
    for k, b in base.items():
        if k in MLP_KEYS:
            continue
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            check(torch.equal(sd[k], b),
                  f"non-MLP tensor {k} differs between base and {name}")

    # 2/3. merge; task vectors are both taken against the untouched base
    out = {}
    merged = 0
    for k, b in base.items():
        if k in MLP_KEYS:
            b32 = b.to(torch.float32)
            m = b32 + LAM * (ft1[k].to(torch.float32) - b32) \
                    + LAM * (ft2[k].to(torch.float32) - b32)
            out[k] = m.to(b.dtype).contiguous()
            merged += 1
        else:
            out[k] = b.clone().contiguous()

    check(merged == N_MLP, f"merged {merged} tensors, expected {N_MLP}")
    check(len(out) == N_TOTAL, f"output has {len(out)} tensors, expected {N_TOTAL}")

    save_file(out, "out/T4/model.safetensors")

    # 4. re-read what was written and re-assert the counts
    back = load("out/T4/model.safetensors")
    check(len(back) == N_TOTAL, f"written file has {len(back)} tensors, expected {N_TOTAL}")
    check(set(back) == set(base), "written file key set differs from base")
    differing = {k for k in back if not torch.equal(back[k], base[k])}
    check(differing <= MLP_KEYS, f"tensors changed outside the MLP set: {sorted(differing - MLP_KEYS)[:5]}")
    print(f"OK: 244 tensors written, {merged} merged, {len(differing)} differ from base")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
