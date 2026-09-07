"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint."""

import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
ADAPTER = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
ADAPTER_CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB of tensor data per shard
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160

# base_model.model.<base name minus ".weight">.lora_{A,B}.weight
ADAPTER_RE = re.compile(r"^base_model\.model\.(?P<mod>.+)\.lora_(?P<ab>[AB])\.weight$")


def main() -> None:
    cfg = json.load(open(ADAPTER_CFG))
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    fan_in_fan_out = bool(cfg["fan_in_fan_out"])
    scale = alpha / r
    if scale != 2.0:
        raise SystemExit(f"unexpected LoRA scale alpha/r = {scale}, expected 2.0")

    base = load_file(BASE)
    adapter = load_file(ADAPTER)

    # ---- group adapter tensors into (A, B) pairs per adapted module --------
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in adapter.items():
        m = ADAPTER_RE.match(name)
        if m is None:
            raise SystemExit(f"adapter tensor with unrecognised name: {name}")
        pairs.setdefault(m.group("mod"), {})[m.group("ab")] = tensor

    incomplete = sorted(k for k, v in pairs.items() if set(v) != {"A", "B"})
    if incomplete:
        raise SystemExit(f"incomplete LoRA pairs (missing A or B): {incomplete}")
    if len(pairs) != EXPECTED_PAIRS:
        raise SystemExit(f"found {len(pairs)} LoRA pairs, expected {EXPECTED_PAIRS}")

    # ---- merge -------------------------------------------------------------
    for mod, ab in sorted(pairs.items()):
        target = f"{mod}.weight"
        if target not in base:
            raise SystemExit(f"adapter targets {mod!r} but base has no tensor {target!r}")
        a, b = ab["A"], ab["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != r or b.shape[1] != r:
            raise SystemExit(f"{mod}: unexpected factor shapes A={tuple(a.shape)} B={tuple(b.shape)}")

        w = base[target].to(torch.float32)
        delta = scale * (b.to(torch.float32) @ a.to(torch.float32))  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # base uses Conv1D [in, out]
        if delta.shape != w.shape:
            raise SystemExit(
                f"{mod}: delta {tuple(delta.shape)} does not match base {tuple(w.shape)}"
            )
        base[target] = (w + delta).contiguous().to(torch.float32)

    merged = sorted(f"{m}.weight" for m in pairs)
    print(f"merged {len(merged)} LoRA pairs, scale={scale}, fan_in_fan_out={fan_in_fan_out}")

    # ---- required checks ---------------------------------------------------
    if len(merged) != EXPECTED_PAIRS:
        raise SystemExit(f"merged {len(merged)} pairs, expected {EXPECTED_PAIRS}")
    leaked = sorted(k for k in base if "lora_" in k)
    if leaked:
        raise SystemExit(f"adapter tensors leaked into the output: {leaked}")
    probe = "h.0.attn.c_attn.weight"
    if tuple(base[probe].shape) != (768, 2304):
        raise SystemExit(f"{probe} has shape {tuple(base[probe].shape)}, expected (768, 2304)")
    if base[probe].dtype != torch.float32:
        raise SystemExit(f"{probe} has dtype {base[probe].dtype}, expected float32")
    if len(base) != EXPECTED_TENSORS:
        raise SystemExit(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    # ---- shard -------------------------------------------------------------
    names = list(base.keys())
    nbytes = {k: base[k].numel() * base[k].element_size() for k in names}

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for k in names:
        size = nbytes[k]
        if size > MAX_SHARD_BYTES:  # oversized tensor gets a shard of its own
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([k])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(k)
        current_bytes += size
    if current:
        shards.append(current)

    total = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale in os.listdir(OUT_DIR):
        if stale.endswith(".safetensors") or stale == "model.safetensors.index.json":
            os.remove(os.path.join(OUT_DIR, stale))

    weight_map: dict[str, str] = {}
    for i, keys in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{total:05d}.safetensors"
        shard_bytes = sum(nbytes[k] for k in keys)
        if shard_bytes > MAX_SHARD_BYTES and len(keys) != 1:
            raise SystemExit(f"{filename}: {shard_bytes} bytes over budget with {len(keys)} tensors")
        save_file(
            {k: base[k].contiguous() for k in keys},
            os.path.join(OUT_DIR, filename),
            metadata={"format": "pt"},
        )
        for k in keys:
            weight_map[k] = filename
        print(f"{filename}: {len(keys):3d} tensors, {shard_bytes / 1048576:.2f} MiB")

    if len(weight_map) != EXPECTED_TENSORS:
        raise SystemExit(f"weight_map has {len(weight_map)} entries, expected {EXPECTED_TENSORS}")
    index = {
        "metadata": {"total_size": sum(nbytes.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # ---- verify what was written ------------------------------------------
    seen: dict[str, torch.Tensor] = {}
    for filename in sorted(set(weight_map.values())):
        for k, v in load_file(os.path.join(OUT_DIR, filename)).items():
            if k in seen:
                raise SystemExit(f"tensor {k} written to more than one shard")
            seen[k] = v
    if set(seen) != set(base):
        raise SystemExit("readback key set differs from the merged state dict")
    if len(seen) != EXPECTED_TENSORS:
        raise SystemExit(f"readback has {len(seen)} tensors, expected {EXPECTED_TENSORS}")
    if any("lora_" in k for k in seen):
        raise SystemExit("readback contains adapter tensors")
    for k, v in seen.items():
        if v.shape != base[k].shape or v.dtype != base[k].dtype:
            raise SystemExit(f"readback mismatch for {k}")
    print(f"OK: {len(seen)} tensors in {total} shards -> {OUT_DIR}")


if __name__ == "__main__":
    main()
