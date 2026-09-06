"""T5: fold a LoRA adapter into the Pythia-1B base weights ("merge and
unload") and write the result as a sharded safetensors checkpoint.

Approach: plain script on top of `safetensors` + `torch` (both in
F-allowed.md, which explicitly permits "plain scripts on top of them").
This gives exact control over the sharding rule in TASK.md (>=512MiB
bin-packing, oversized tensors alone in their own shard), which is safer
here than routing through peft.merge_and_unload + transformers
save_pretrained sharding (that path also requires instantiating the full
GPTNeoX model just to rewrite a state dict).
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
INPUTS = REPO_ROOT / "inputs"
OUT_DIR = HERE  # out/T5/

BASE_CKPT = INPUTS / "base" / "model.safetensors"
LORA_CKPT = INPUTS / "lora" / "adapter_model.safetensors"
LORA_CFG = INPUTS / "lora" / "adapter_config.json"

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes, tensor data only

ADAPTER_PREFIX = "base_model.model."


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def main() -> None:
    cfg = json.loads(LORA_CFG.read_text())
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg["fan_in_fan_out"]
    if fan_in_fan_out:
        raise ValueError(
            f"fan_in_fan_out=True is not handled by this script (got {fan_in_fan_out})"
        )
    scale = alpha / r

    base = load_all(BASE_CKPT)
    lora = load_all(LORA_CKPT)

    # Map module name -> {"A": tensor, "B": tensor}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in lora.items():
        if not key.startswith(ADAPTER_PREFIX):
            raise ValueError(f"unexpected adapter key prefix: {key}")
        rest = key[len(ADAPTER_PREFIX) :]
        if rest.endswith(".lora_A.weight"):
            module = rest[: -len(".lora_A.weight")]
            pairs.setdefault(module, {})["A"] = tensor
        elif rest.endswith(".lora_B.weight"):
            module = rest[: -len(".lora_B.weight")]
            pairs.setdefault(module, {})["B"] = tensor
        else:
            raise ValueError(f"unexpected adapter tensor name: {key}")

    # --- Required checks (fail loudly before writing anything) ---
    if len(pairs) != 16:
        raise AssertionError(f"expected exactly 16 adapter pairs, found {len(pairs)}")
    for module, parts in pairs.items():
        if set(parts) != {"A", "B"}:
            raise AssertionError(f"incomplete adapter pair for {module}: {sorted(parts)}")

    merged = dict(base)  # unchanged tensors keep the exact same object -> bit-exact
    for module, parts in sorted(pairs.items()):
        base_key = f"{module}.weight"
        if base_key not in base:
            raise KeyError(f"adapter targets missing base tensor {base_key}")
        A = parts["A"].to(torch.float32)
        B = parts["B"].to(torch.float32)
        base_weight = base[base_key]
        delta = scale * (B @ A)  # [out, r] @ [r, in] -> [out, in]
        if delta.shape != base_weight.shape:
            raise AssertionError(
                f"{base_key}: delta shape {tuple(delta.shape)} != "
                f"base shape {tuple(base_weight.shape)}"
            )
        merged_weight = (base_weight.to(torch.float32) + delta).to(base_weight.dtype)
        merged[base_key] = merged_weight

    if any("lora_" in k for k in merged):
        raise AssertionError("adapter tensor leaked into merged state dict")

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(merged[qkv0].shape) != (6144, 2048):
        raise AssertionError(f"{qkv0} has shape {tuple(merged[qkv0].shape)}, expected (6144, 2048)")

    if len(merged) != 244:
        raise AssertionError(f"expected 244 tensors in output, got {len(merged)}")

    # --- Shard: greedy bin-pack in a fixed (sorted) key order, oversized
    # tensors get their own shard. ---
    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names = sorted(merged.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_bytes(merged[name])
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any stale shard/index files from a previous attempt.
    for stale in OUT_DIR.glob("model-*-of-*.safetensors"):
        stale.unlink()
    index_path = OUT_DIR / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_file = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: merged[name].contiguous() for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_file), metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_file
            total_size += tensor_bytes(merged[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path.write_text(json.dumps(index, indent=2))

    print(f"merged {len(pairs)} adapter pairs into {len(merged)} tensors")
    print(f"wrote {n_shards} shard(s) + index to {OUT_DIR}")


if __name__ == "__main__":
    main()
