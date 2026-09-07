"""T5: merge a LoRA adapter into GPT-2's c_attn weights and write a sharded
safetensors checkpoint, using plain torch + safetensors (condition F).

Why not peft.merge_and_unload(): that path requires instantiating the full
HF model, merging in-place on the module tree, then re-exporting via
transformers' sharded save. It works, but it hides the two facts that
actually decide correctness here (fan_in_fan_out transpose, scale = alpha/r)
inside library code, and it's harder to enforce the "no lora_/exactly 12
pairs/160 tensors" checks *before* writing. Doing the merge directly on the
state dicts keeps every one of those decisions explicit and checkable.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root (out/T5 -> out -> sandbox)
BASE_PATH = ROOT / "inputs" / "base" / "model.safetensors"
LORA_PATH = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CONFIG_PATH = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = ROOT / "out" / "T5"
MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB, tensor data only

LORA_A_RE = re.compile(r"^base_model\.model\.h\.(\d+)\.(.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.h\.(\d+)\.(.+)\.lora_B\.weight$")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base = load_state_dict(BASE_PATH)
    lora = load_state_dict(LORA_PATH)
    lora_config = json.loads(LORA_CONFIG_PATH.read_text())

    r = lora_config["r"]
    alpha = lora_config["lora_alpha"]
    fan_in_fan_out = lora_config["fan_in_fan_out"]
    if not fan_in_fan_out:
        raise AssertionError(
            "expected fan_in_fan_out=true (Conv1D base layout); got False, "
            "merge math would need to drop the transpose"
        )
    scale = alpha / r

    # Pair up lora_A / lora_B by (layer, module).
    pairs: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in lora.items():
        m = LORA_A_RE.match(key)
        if m:
            pairs.setdefault((m.group(1), m.group(2)), {})["A"] = tensor
            continue
        m = LORA_B_RE.match(key)
        if m:
            pairs.setdefault((m.group(1), m.group(2)), {})["B"] = tensor
            continue
        raise AssertionError(f"unrecognized adapter tensor name: {key!r}")

    if len(pairs) != 12:
        raise AssertionError(f"expected exactly 12 adapter pairs, found {len(pairs)}")
    for (layer, module), parts in pairs.items():
        if set(parts) != {"A", "B"}:
            raise AssertionError(
                f"layer {layer} module {module!r} is missing lora_A or lora_B"
            )

    merged = dict(base)
    merged_count = 0
    for (layer, module), parts in pairs.items():
        base_key = f"h.{layer}.{module}.weight"
        if base_key not in merged:
            raise AssertionError(f"base tensor {base_key!r} not found for adapter pair")

        base_tensor = merged[base_key]
        if base_tensor.dtype != torch.float32:
            raise AssertionError(f"{base_key} is not float32: {base_tensor.dtype}")

        A = parts["A"].to(torch.float32)  # [r, in]
        B = parts["B"].to(torch.float32)  # [out, r]
        delta = scale * (B @ A).T  # Linear layout [out, in] -> transpose to Conv1D [in, out]

        if delta.shape != base_tensor.shape:
            raise AssertionError(
                f"shape mismatch merging {base_key}: base {tuple(base_tensor.shape)} "
                f"vs delta {tuple(delta.shape)}"
            )

        merged[base_key] = (base_tensor.to(torch.float32) + delta).contiguous()
        merged_count += 1

    if merged_count != 12:
        raise AssertionError(f"expected to merge 12 tensors, merged {merged_count}")

    if any("lora_" in key for key in merged):
        raise AssertionError("adapter tensor leaked into merged state dict")

    c_attn0_shape = tuple(merged["h.0.attn.c_attn.weight"].shape)
    if c_attn0_shape != (768, 2304):
        raise AssertionError(f"h.0.attn.c_attn.weight has shape {c_attn0_shape}, expected (768, 2304)")

    if len(merged) != 160:
        raise AssertionError(f"expected 160 tensors in output, got {len(merged)}")

    # Shard: greedily pack tensors (in original key order) into shards capped
    # at MAX_SHARD_BYTES of tensor data; an oversized tensor gets its own shard.
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key, tensor in merged.items():
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([key])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(key)
        current_bytes += nbytes
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, keys in enumerate(shards, start=1):
        shard_name = f"model-{i:05d}-of-{n:05d}.safetensors"
        shard_tensors = {k: merged[k] for k in keys}
        save_file(shard_tensors, str(OUT_DIR / shard_name), metadata={"format": "pt"})
        for k, t in shard_tensors.items():
            weight_map[k] = shard_name
            total_size += t.numel() * t.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"Merged {merged_count} adapter pairs into {len(merged)} tensors, {n} shards.")


if __name__ == "__main__":
    main()
