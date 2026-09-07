"""T3: mixed-precision export of GPT-2 (124M) with sharding.

Casts the 48 projection matrices to bfloat16, keeps everything else float32,
drops the 12 causal-mask buffers, and writes a sharded safetensors checkpoint
with an index file.
"""

import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
DST_DIR = os.path.join(SANDBOX, "out", "T3")

MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67,108,864

# The projection matrices: exactly these four per layer, nothing else.
PROJ_RE = re.compile(r"h\.\d+\.(?:attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight\Z")
# The non-parameter causal-mask buffer. Note this is `attn.bias`, NOT the
# projection biases `attn.c_attn.bias` / `attn.c_proj.bias`, which are kept.
BUFFER_RE = re.compile(r"h\.\d+\.attn\.bias\Z")

EXPECTED_PROJ_SHAPES = {
    "attn.c_attn": (768, 2304),
    "attn.c_proj": (768, 768),
    "mlp.c_fc": (768, 3072),
    "mlp.c_proj": (3072, 768),
}


def fail(msg):
    raise SystemExit(f"CHECK FAILED: {msg}")


def main():
    src = load_file(SRC)
    print(f"loaded {len(src)} tensors from {SRC}")
    if len(src) != 160:
        fail(f"expected 160 input tensors, got {len(src)}")

    # --- build the expected name sets explicitly, and cross-check the regexes --
    layers = sorted({int(m.group(1)) for k in src for m in [re.match(r"h\.(\d+)\.", k)] if m})
    if layers != list(range(12)):
        fail(f"expected layers 0..11, got {layers}")

    want_proj = {
        f"h.{i}.{stem}.weight" for i in layers for stem in EXPECTED_PROJ_SHAPES
    }
    want_buffers = {f"h.{i}.attn.bias" for i in layers}

    got_proj = {k for k in src if PROJ_RE.fullmatch(k)}
    got_buffers = {k for k in src if BUFFER_RE.fullmatch(k)}

    if got_proj != want_proj:
        fail(
            "projection match mismatch; "
            f"unexpected={sorted(got_proj - want_proj)} missing={sorted(want_proj - got_proj)}"
        )
    if got_buffers != want_buffers:
        fail(
            "buffer match mismatch; "
            f"unexpected={sorted(got_buffers - want_buffers)} missing={sorted(want_buffers - got_buffers)}"
        )
    if got_proj & got_buffers:
        fail(f"a tensor is both a projection and a buffer: {sorted(got_proj & got_buffers)}")
    if len(got_proj) != 48:
        fail(f"expected 48 projection matrices, got {len(got_proj)}")
    if len(got_buffers) != 12:
        fail(f"expected 12 buffers, got {len(got_buffers)}")

    # Shapes of the projections must be what the task documents (guards against
    # a name matching the pattern but being something else).
    for name in sorted(got_proj):
        stem = name[len("h."):].split(".", 1)[1].rsplit(".weight", 1)[0]
        want_shape = EXPECTED_PROJ_SHAPES[stem]
        if tuple(src[name].shape) != want_shape:
            fail(f"{name}: expected shape {want_shape}, got {tuple(src[name].shape)}")

    # Sanity: the things that must stay float32 are not in the cast set.
    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if name not in src:
            fail(f"missing expected tensor {name}")
        if name in got_proj:
            fail(f"{name} was selected for casting but must stay float32")
    for name in src:
        if name.endswith(".bias") and name in got_proj:
            fail(f"bias {name} was selected for casting")
        if ".ln_" in name and name in got_proj:
            fail(f"layer norm {name} was selected for casting")

    # --- build the output state dict ------------------------------------------
    out = {}
    for name in sorted(src):  # deterministic, alphabetical order
        if name in got_buffers:
            continue
        t = src[name]
        if t.dtype != torch.float32:
            fail(f"{name}: input dtype is {t.dtype}, expected float32")
        if name in got_proj:
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            out[name] = t.clone().contiguous()  # unchanged float32 values

    # --- required checks, before writing --------------------------------------
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 48:
        fail(f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        fail(f"h.0.attn.c_attn.weight is {out['h.0.attn.c_attn.weight'].dtype}, expected bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        fail(f"wte.weight is {out['wte.weight'].dtype}, expected float32")
    if len(out) != 148:
        fail(f"expected 148 output tensors, got {len(out)}")

    # extra guards: no parameter lost, no name changed, non-cast dtypes all f32
    if set(out) != set(src) - want_buffers:
        fail("output key set is not the input key set minus the 12 buffers")
    for name, t in out.items():
        if name not in got_proj and t.dtype != torch.float32:
            fail(f"{name}: expected float32 in output, got {t.dtype}")
        if tuple(t.shape) != tuple(src[name].shape):
            fail(f"{name}: shape changed {tuple(src[name].shape)} -> {tuple(t.shape)}")
    for name in out:
        if name in got_proj:
            continue
        if not torch.equal(out[name], src[name]):
            fail(f"{name}: values changed but must be unchanged")

    # --- greedy sharding -------------------------------------------------------
    def nbytes(t):
        return t.numel() * t.element_size()

    shards = []  # list of list-of-names
    cur, cur_size = [], 0
    for name in out:  # insertion order == sorted order
        size = nbytes(out[name])
        if cur and cur_size + size > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += size
    if cur:
        shards.append(cur)

    n = len(shards)
    filenames = [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]

    for fn, names in zip(filenames, shards):
        size = sum(nbytes(out[k]) for k in names)
        if size > MAX_SHARD_BYTES and len(names) != 1:
            fail(f"{fn}: {size} bytes over the {MAX_SHARD_BYTES} budget with {len(names)} tensors")
        print(f"{fn}: {len(names):3d} tensors, {size:,} bytes")

    weight_map = {k: fn for fn, names in zip(filenames, shards) for k in names}
    if len(weight_map) != 148:
        fail(f"weight_map covers {len(weight_map)} tensors, expected 148")

    total_size = sum(nbytes(t) for t in out.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}

    # --- write -----------------------------------------------------------------
    os.makedirs(DST_DIR, exist_ok=True)
    for stale in os.listdir(DST_DIR):
        if stale.endswith(".safetensors") or stale == "model.safetensors.index.json":
            os.remove(os.path.join(DST_DIR, stale))

    for fn, names in zip(filenames, shards):
        save_file(
            {k: out[k] for k in names},
            os.path.join(DST_DIR, fn),
            metadata={"format": "pt"},
        )
    with open(os.path.join(DST_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=False)
        f.write("\n")

    # --- read back and re-verify ----------------------------------------------
    back = {}
    for fn in filenames:
        part = load_file(os.path.join(DST_DIR, fn))
        dup = set(part) & set(back)
        if dup:
            fail(f"tensor(s) present in more than one shard: {sorted(dup)}")
        back.update(part)

    if len(back) != 148:
        fail(f"read back {len(back)} tensors, expected 148")
    if set(back) != set(weight_map):
        fail("read-back key set differs from the index weight_map")
    if sum(1 for t in back.values() if t.dtype == torch.bfloat16) != 48:
        fail("read-back bfloat16 count is not 48")
    if back["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        fail("read-back h.0.attn.c_attn.weight is not bfloat16")
    if back["wte.weight"].dtype != torch.float32:
        fail("read-back wte.weight is not float32")
    for name, t in back.items():
        ref = src[name].to(torch.bfloat16) if name in got_proj else src[name]
        if t.dtype != ref.dtype:
            fail(f"read-back {name}: dtype {t.dtype} != {ref.dtype}")
        if not torch.equal(t, ref):
            fail(f"read-back {name}: values are not bit-exact")

    print(
        f"OK: {len(back)} tensors, {n_bf16} bfloat16, {len(want_buffers)} buffers dropped, "
        f"{n} shards, total_size={total_size:,} bytes -> {DST_DIR}"
    )


if __name__ == "__main__":
    main()
