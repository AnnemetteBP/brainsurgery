"""
Merge a PEFT-style LoRA adapter into the GPT-2 base checkpoint ("merge and
unload") and write the result as a sharded safetensors checkpoint.

- base tensors are Conv1D layout [in, out]
- adapter factors (lora_A, lora_B) follow the nn.Linear convention, so the
  low-rank update B @ A (shape [out, in]) must be transposed before being
  added to the Conv1D weight, per fan_in_fan_out=true in adapter_config.json.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
LORA_PATH = HERE.parent.parent / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CONFIG_PATH = HERE.parent.parent / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = HERE
SHARD_BYTE_LIMIT = 100 * 1024 * 1024  # 104,857,600 bytes, tensor data only

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

    with open(LORA_CONFIG_PATH) as f:
        lora_config = json.load(f)

    r = lora_config["r"]
    alpha = lora_config["lora_alpha"]
    fan_in_fan_out = lora_config["fan_in_fan_out"]
    scale = alpha / r

    if not fan_in_fan_out:
        raise RuntimeError("expected fan_in_fan_out=true; solution assumes Linear-layout factors")

    # Collect (layer, module) -> {"A": tensor, "B": tensor}
    pairs: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for name, tensor in lora.items():
        m = LORA_A_RE.match(name)
        if m:
            layer, module = m.group(1), m.group(2)
            pairs.setdefault((layer, module), {})["A"] = tensor
            continue
        m = LORA_B_RE.match(name)
        if m:
            layer, module = m.group(1), m.group(2)
            pairs.setdefault((layer, module), {})["B"] = tensor
            continue
        raise RuntimeError(f"unrecognized adapter tensor name: {name}")

    if len(pairs) != 12:
        raise RuntimeError(f"expected exactly 12 adapter pairs, found {len(pairs)}")

    for (layer, module), factors in pairs.items():
        if "A" not in factors or "B" not in factors:
            raise RuntimeError(f"incomplete adapter pair for layer {layer} module {module}")

        base_name = f"h.{layer}.{module}.weight"
        if base_name not in base:
            raise RuntimeError(f"base tensor {base_name} not found for adapter pair")

        A = factors["A"].to(torch.float32)  # [r, in]
        B = factors["B"].to(torch.float32)  # [out, r]

        base_weight = base[base_name]
        if base_weight.dtype != torch.float32:
            raise RuntimeError(f"expected float32 base tensor for {base_name}")

        delta = scale * (B @ A).T  # [in, out], matches Conv1D layout
        if delta.shape != base_weight.shape:
            raise RuntimeError(
                f"shape mismatch merging {base_name}: delta {tuple(delta.shape)} "
                f"vs base {tuple(base_weight.shape)}"
            )

        base[base_name] = (base_weight + delta).contiguous()

    # --- Required checks ---
    if len(pairs) != 12:
        raise RuntimeError(f"expected 12 merged adapter pairs, got {len(pairs)}")
    if any("lora_" in name for name in base):
        raise RuntimeError("adapter tensor leaked into output state dict")
    if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        raise RuntimeError(
            f"h.0.attn.c_attn.weight has wrong shape: {tuple(base['h.0.attn.c_attn.weight'].shape)}"
        )
    if len(base) != 160:
        raise RuntimeError(f"expected 160 tensors in output, got {len(base)}")

    # --- Shard and write ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Bin-pack tensors into shards, each capped at SHARD_BYTE_LIMIT bytes of
    # tensor data (a single oversized tensor gets its own shard).
    items = list(base.items())  # preserves insertion (base file) order
    shards: list[list[str]] = []
    shard_bytes: list[int] = []
    for name, tensor in items:
        nbytes = tensor.numel() * tensor.element_size()
        placed = False
        if nbytes <= SHARD_BYTE_LIMIT:
            for i in range(len(shards)):
                if shard_bytes[i] + nbytes <= SHARD_BYTE_LIMIT:
                    shards[i].append(name)
                    shard_bytes[i] += nbytes
                    placed = True
                    break
        if not placed:
            shards.append([name])
            shard_bytes.append(nbytes)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: base[name] for name in names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename), metadata={"format": "pt"})
        for name, tensor in shard_tensors.items():
            weight_map[name] = shard_filename
            total_size += tensor.numel() * tensor.element_size()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {len(pairs)} adapter pairs into {len(base)} tensors across {num_shards} shards.")


if __name__ == "__main__":
    main()
