"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint."""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
ADAPTER = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CONFIG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")
MAX_SHARD_BYTES = 100 * 1024 * 1024


def load(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def main():
    cfg = json.load(open(CONFIG))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg["fan_in_fan_out"]
    scale = alpha / r

    base = load(BASE)
    adapter = load(ADAPTER)
    n_base = len(base)

    # Collect adapter pairs: <prefix>.lora_A.weight / <prefix>.lora_B.weight
    pairs = {}
    for name in adapter:
        if name.endswith(".lora_A.weight"):
            pairs.setdefault(name[: -len(".lora_A.weight")], {})["A"] = name
        elif name.endswith(".lora_B.weight"):
            pairs.setdefault(name[: -len(".lora_B.weight")], {})["B"] = name
        else:
            raise SystemExit(f"unexpected adapter tensor: {name}")

    merged = 0
    for prefix, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            raise SystemExit(f"incomplete LoRA pair for {prefix}")
        # strip the PEFT wrapper prefix to get the base tensor name
        target = prefix
        for p in ("base_model.model.", "base_model.", "model."):
            if target.startswith(p):
                target = target[len(p) :]
                break
        target = target + ".weight"
        if target not in base:
            raise SystemExit(f"no base tensor for adapter {prefix} (looked for {target})")

        A = adapter[ab["A"]].to(torch.float32)  # [r, in]
        B = adapter[ab["B"]].to(torch.float32)  # [out, r]
        if A.shape[0] != r or B.shape[1] != r:
            raise SystemExit(f"rank mismatch for {prefix}: A={tuple(A.shape)} B={tuple(B.shape)}")

        delta = scale * (B @ A)  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # base uses Conv1D [in, out]

        W = base[target]
        if W.shape != delta.shape:
            raise SystemExit(
                f"shape mismatch for {target}: base={tuple(W.shape)} delta={tuple(delta.shape)}"
            )
        if W.dtype != torch.float32:
            raise SystemExit(f"unexpected base dtype for {target}: {W.dtype}")
        base[target] = (W.to(torch.float32) + delta).contiguous()
        merged += 1

    # ---- required checks -------------------------------------------------
    if merged != 12:
        raise SystemExit(f"expected 12 adapter pairs merged, got {merged}")
    bad = [k for k in base if "lora_" in k]
    if bad:
        raise SystemExit(f"adapter tensors leaked into the output: {bad}")
    probe = "h.0.attn.c_attn.weight"
    if tuple(base[probe].shape) != (768, 2304):
        raise SystemExit(f"{probe} has shape {tuple(base[probe].shape)}, expected (768, 2304)")
    if len(base) != 160:
        raise SystemExit(f"output has {len(base)} tensors, expected 160")
    if len(base) != n_base:
        raise SystemExit(f"tensor count changed: {n_base} -> {len(base)}")

    # ---- shard -----------------------------------------------------------
    def nbytes(t):
        return t.numel() * t.element_size()

    names = list(base.keys())
    shards = []
    cur, cur_size = [], 0
    for name in names:
        size = nbytes(base[name])
        if cur and cur_size + size > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += size
    if cur:
        shards.append(cur)

    total = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        payload = {n: base[n].contiguous().clone() for n in shard}
        shard_bytes = sum(nbytes(t) for t in payload.values())
        if shard_bytes > MAX_SHARD_BYTES and len(payload) > 1:
            raise SystemExit(f"shard {fname} is {shard_bytes} bytes over the budget")
        save_file(payload, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        total_size += shard_bytes
        for n in shard:
            weight_map[n] = fname

    if len(weight_map) != 160:
        raise SystemExit(f"weight_map has {len(weight_map)} entries, expected 160")
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2
        )

    print(f"merged {merged} LoRA pairs, wrote {total} shards, {len(weight_map)} tensors")
    for i, shard in enumerate(shards, start=1):
        s = sum(nbytes(base[n]) for n in shard)
        print(f"  shard {i}: {len(shard)} tensors, {s} bytes")


if __name__ == "__main__":
    main()
