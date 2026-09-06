"""T5: fold a PEFT LoRA adapter into the Pythia-1B base weights and write a
sharded safetensors checkpoint.

Approach: operate directly on the checkpoint files with `safetensors` +
`torch`, and shard with `huggingface_hub.split_torch_state_dict_into_shards`
(the same splitter `transformers.save_pretrained` uses), so the shard layout
and index follow the standard convention.

`peft.merge_and_unload` was deliberately not used: it requires instantiating a
`GPTNeoXForCausalLM`, and `save_pretrained` would then drop the non-persistent
buffers that this base checkpoint stores (`attention.bias`,
`attention.masked_bias`, `rotary_emb.inv_freq`), breaking the required
244-tensor key set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = Path("inputs/base/model.safetensors")
LORA = Path("inputs/lora/adapter_model.safetensors")
LORA_CFG = Path("inputs/lora/adapter_config.json")
OUT_DIR = Path("out/T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 512 MiB of tensor data
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244
PROBE = "gpt_neox.layers.0.attention.query_key_value.weight"
PROBE_SHAPE = (6144, 2048)
INDEX_NAME = "model.safetensors.index.json"

# base_model.model.<base name>.lora_{A,B}.weight  (PEFT wrapper prefix)
ADAPTER_RE = re.compile(
    r"^base_model\.model\.(?P<base>.+)\.lora_(?P<side>[AB])\.(?:weight|default\.weight)$"
)


class CheckFailed(RuntimeError):
    """A required check did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def load_adapter_pairs() -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    cfg = json.loads(LORA_CFG.read_text())
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    check(isinstance(r, int) and r > 0, f"adapter_config.json: bad r={r!r}")
    scale = alpha / r

    # The base weights use the nn.Linear [out, in] layout, matching the adapter
    # factors, so B @ A is added untransposed. Anything else is unhandled.
    check(
        fan_in_fan_out is False,
        f"adapter_config.json: fan_in_fan_out={fan_in_fan_out!r}, "
        "only the fan_in_fan_out=false (nn.Linear [out, in]) layout is handled",
    )

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(LORA, framework="pt") as f:
        for key in f.keys():
            m = ADAPTER_RE.match(key)
            check(m is not None, f"unrecognised adapter tensor name: {key}")
            assert m is not None
            pairs.setdefault(m["base"], {})[m["side"]] = f.get_tensor(key)

    for base_name, sides in sorted(pairs.items()):
        check(
            set(sides) == {"A", "B"},
            f"{base_name}: incomplete adapter pair, sides={sorted(sides)}",
        )
        a, b = sides["A"], sides["B"]
        check(a.ndim == 2 and b.ndim == 2, f"{base_name}: lora factors must be 2-D")
        check(
            a.shape[0] == r and b.shape[1] == r,
            f"{base_name}: rank mismatch, A{tuple(a.shape)} B{tuple(b.shape)}, r={r}",
        )

    # Required check: exactly 16 adapter pairs.
    check(
        len(pairs) == EXPECTED_PAIRS,
        f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}",
    )
    return pairs, scale


def merge(state: dict[str, torch.Tensor], pairs, scale: float) -> int:
    merged = 0
    for base_name, sides in sorted(pairs.items()):
        target = f"{base_name}.weight"
        check(target in state, f"adapter targets {target}, which is not in the base")
        w = state[target]
        a, b = sides["A"], sides["B"]
        delta_shape = (b.shape[0], a.shape[1])
        check(
            tuple(w.shape) == delta_shape,
            f"{target}: base shape {tuple(w.shape)} != B @ A shape {delta_shape}",
        )
        dtype = w.dtype
        # compute in float32, cast back to the base dtype
        updated = w.to(torch.float32) + scale * (b.to(torch.float32) @ a.to(torch.float32))
        state[target] = updated.to(dtype)
        check(
            state[target].dtype == dtype and tuple(state[target].shape) == delta_shape,
            f"{target}: merge changed dtype or shape",
        )
        merged += 1
    check(merged == EXPECTED_PAIRS, f"merged {merged} pairs, expected {EXPECTED_PAIRS}")
    return merged


def run_required_checks(state: dict[str, torch.Tensor], merged: int) -> None:
    check(merged == EXPECTED_PAIRS, f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")

    leaked = sorted(k for k in state if "lora_" in k)
    check(not leaked, f"adapter/intermediate tensors leaked into the output: {leaked}")

    check(PROBE in state, f"{PROBE} missing from the output")
    check(
        tuple(state[PROBE].shape) == PROBE_SHAPE,
        f"{PROBE} has shape {tuple(state[PROBE].shape)}, expected {list(PROBE_SHAPE)}",
    )

    check(
        len(state) == EXPECTED_TENSORS,
        f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}",
    )


def write_sharded(state: dict[str, torch.Tensor]) -> None:
    split = split_torch_state_dict_into_shards(
        state,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD_BYTES,
    )

    # Sharding invariants, checked before anything is written.
    for filename, keys in split.filename_to_tensors.items():
        total = sum(nbytes(state[k]) for k in keys)
        if len(keys) == 1:
            continue  # an oversized tensor is allowed to sit alone in its own shard
        check(
            total <= MAX_SHARD_BYTES,
            f"{filename}: {total} bytes of tensor data exceeds the "
            f"{MAX_SHARD_BYTES}-byte shard budget",
        )
        check(
            all(nbytes(state[k]) <= MAX_SHARD_BYTES for k in keys),
            f"{filename}: an over-budget tensor shares a shard with others",
        )
    mapped = {k for keys in split.filename_to_tensors.values() for k in keys}
    check(mapped == set(state), "shard split does not cover exactly the output tensors")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove only checkpoint files left by an earlier run; this directory also
    # holds this script, so it must never be cleared wholesale.
    for stale in (*OUT_DIR.glob("*.safetensors"), OUT_DIR / INDEX_NAME):
        stale.unlink(missing_ok=True)

    for filename, keys in split.filename_to_tensors.items():
        shard = {k: state[k].contiguous() for k in keys}
        save_file(shard, OUT_DIR / filename, metadata={"format": "pt"})

    check(split.is_sharded, "the output did not need sharding, expected several shards")
    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    (OUT_DIR / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def verify_written(expected: dict[str, torch.Tensor]) -> None:
    index_path = OUT_DIR / INDEX_NAME
    check(index_path.is_file(), f"{INDEX_NAME} was not written")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    check(
        len(weight_map) == EXPECTED_TENSORS,
        f"weight_map has {len(weight_map)} entries, expected {EXPECTED_TENSORS}",
    )
    check(set(weight_map) == set(expected), "weight_map keys differ from the merged state dict")

    listed = set(weight_map.values())
    on_disk = {p.name for p in OUT_DIR.glob("*.safetensors")}
    check(listed == on_disk, f"shard files on disk {sorted(on_disk)} != indexed {sorted(listed)}")

    seen: dict[str, torch.Tensor] = {}
    for filename in sorted(listed):
        path = OUT_DIR / filename
        check(path.is_file(), f"{filename} listed in the index but not written")
        shard = load_file(path)
        payload = sum(nbytes(t) for t in shard.values())
        check(
            payload <= MAX_SHARD_BYTES or len(shard) == 1,
            f"{filename}: {payload} bytes of tensor data over the shard budget",
        )
        for k, t in shard.items():
            check(k not in seen, f"{k} appears in more than one shard")
            check(weight_map[k] == filename, f"{k} is in {filename} but mapped elsewhere")
            seen[k] = t

    check(len(seen) == EXPECTED_TENSORS, f"read back {len(seen)} tensors, expected 244")
    check(not [k for k in seen if "lora_" in k], "adapter tensor found in the written output")
    check(tuple(seen[PROBE].shape) == PROBE_SHAPE, f"{PROBE} shape wrong on read-back")
    for k, t in seen.items():
        check(t.dtype == expected[k].dtype, f"{k}: dtype {t.dtype} != {expected[k].dtype}")
        check(tuple(t.shape) == tuple(expected[k].shape), f"{k}: shape changed on read-back")
        check(torch.equal(t, expected[k]), f"{k}: values changed on read-back")


def main() -> None:
    state = load_file(BASE)
    base_keys = set(state)
    check(
        len(state) == EXPECTED_TENSORS,
        f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}",
    )

    pairs, scale = load_adapter_pairs()
    print(f"adapter: {len(pairs)} pairs, scale = alpha/r = {scale}")
    merged = merge(state, pairs, scale)

    run_required_checks(state, merged)

    # Every non-adapted tensor must be bit-identical to the base.
    targets = {f"{n}.weight" for n in pairs}
    check(set(state) == base_keys, "output key set differs from the base key set")
    with safe_open(BASE, framework="pt") as f:
        for k in state:
            if k in targets:
                continue
            check(torch.equal(state[k], f.get_tensor(k)), f"{k} was modified but should be unchanged")
    print(f"merged {merged} weights; {len(state) - len(targets)} tensors untouched")

    write_sharded(state)
    verify_written(state)

    shards = sorted(p.name for p in OUT_DIR.glob("*.safetensors"))
    print(f"wrote {len(state)} tensors into {len(shards)} shards")
    for name in shards:
        n = sum(nbytes(t) for t in load_file(OUT_DIR / name).values())
        print(f"  {name}: {n} bytes ({n / 2**20:.1f} MiB)")
    print("OK")


if __name__ == "__main__":
    main()
