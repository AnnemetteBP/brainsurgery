"""T5: Merge a LoRA adapter into a GPT-2 base checkpoint and write it sharded.

Standalone script: torch + safetensors only, no brainsurgery.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
BASE_PATH = INPUTS / "base" / "model.safetensors"
ADAPTER_PATH = INPUTS / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = INPUTS / "lora" / "adapter_config.json"
OUT_DIR = HERE
MAX_SHARD_BYTES = 100 * 1024 * 1024  # 104,857,600 bytes

LORA_KEY_RE = re.compile(
    r"^base_model\.model\.(h\.\d+\.attn\.c_attn)\.lora_([AB])\.weight$"
)


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base = load_all(BASE_PATH)
    adapter = load_all(ADAPTER_PATH)
    config = json.loads(ADAPTER_CONFIG_PATH.read_text())

    r = config["r"]
    lora_alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    target_modules = config["target_modules"]
    assert target_modules == ["c_attn"], f"unexpected target_modules: {target_modules}"
    assert fan_in_fan_out is True, "this script assumes fan_in_fan_out=true (Conv1D layout)"
    scale = lora_alpha / r

    # Group adapter tensors by base module name.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter.items():
        m = LORA_KEY_RE.match(key)
        if m is None:
            raise ValueError(f"unrecognized adapter tensor name: {key!r}")
        module_name, ab = m.group(1), m.group(2)
        pairs.setdefault(module_name, {})[ab] = tensor

    if len(pairs) != 12:
        raise AssertionError(f"expected exactly 12 adapter pairs, found {len(pairs)}")
    for module_name, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise AssertionError(f"module {module_name} is missing lora_A or lora_B")

    merged = dict(base)  # shallow copy; unchanged tensors are shared references
    merged_count = 0
    for module_name, ab in pairs.items():
        base_key = f"{module_name}.weight"
        if base_key not in base:
            raise KeyError(f"base checkpoint has no tensor {base_key!r}")

        A = ab["A"].to(torch.float32)  # [r, in]   = [16, 768]
        B = ab["B"].to(torch.float32)  # [out, r]  = [2304, 16]
        base_weight = base[base_key].to(torch.float32)  # Conv1D layout [in, out]

        if A.shape != (r, base_weight.shape[0]):
            raise AssertionError(f"{module_name}: unexpected lora_A shape {tuple(A.shape)}")
        if B.shape != (base_weight.shape[1], r):
            raise AssertionError(f"{module_name}: unexpected lora_B shape {tuple(B.shape)}")

        delta = scale * (B @ A).T  # nn.Linear-style [out, in] -> transpose to Conv1D [in, out]
        if delta.shape != base_weight.shape:
            raise AssertionError(
                f"{module_name}: delta shape {tuple(delta.shape)} != base shape "
                f"{tuple(base_weight.shape)}"
            )

        merged[base_key] = (base_weight + delta).contiguous()
        merged_count += 1

    # --- Required checks ---
    if merged_count != 12:
        raise AssertionError(f"merged {merged_count} adapter pairs, expected 12")
    if any("lora_" in name for name in merged):
        raise AssertionError("an adapter tensor leaked into the merged output")
    if tuple(merged["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        raise AssertionError(
            f"h.0.attn.c_attn.weight has shape {tuple(merged['h.0.attn.c_attn.weight'].shape)}, "
            "expected (768, 2304)"
        )
    if len(merged) != 160:
        raise AssertionError(f"output has {len(merged)} tensors, expected 160")

    # --- Shard and write ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name, tensor in merged.items():
        size = tensor_bytes(tensor)
        if size > MAX_SHARD_BYTES:
            # Doesn't fit with anything else; goes alone in its own shard.
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
    if current:
        shards.append(current)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    digits = max(5, len(str(num_shards)))
    for i, shard in enumerate(shards, start=1):
        shard_name = f"model-{i:0{digits}d}-of-{num_shards:0{digits}d}.safetensors"
        shard_path = OUT_DIR / shard_name
        save_file(shard, str(shard_path), metadata={"format": "pt"})
        for name in shard:
            weight_map[name] = shard_name
            total_size += tensor_bytes(shard[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"Merged {merged_count} LoRA pairs into {len(merged)} tensors, "
          f"wrote {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
