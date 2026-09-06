"""
T5: LoRA adapter merge with sharded export (Pythia-1B).

Approach: plain script on top of `safetensors` + `torch`. We deliberately
avoid `peft.merge_and_unload`, which requires instantiating the full HF model
(extra dependency surface and memory) just to do a rank-16 matmul-and-add on
32 tensors whose naming convention is already known from adapter_config.json.
Reading/writing safetensors directly also gives direct control over the
sharding rule required by the task (per-shard tensor-data budget, with
oversized tensors alone in their own shard), which mirrors what
`transformers` sharded `save_pretrained` does internally.

Steps:
1. Load base state dict and LoRA adapter state dict.
2. Validate adapter_config.json matches expectations (r, alpha, fan_in_fan_out).
3. Map each lora_A/lora_B pair to its base tensor name, merge in float32,
   cast back to the base dtype (float16).
4. Run required checks (exactly 16 pairs merged, no lora_ tensor leaks,
   shape preserved, tensor count preserved) before writing anything.
5. Shard the resulting state dict under a 512 MiB per-shard tensor-data
   budget (oversized tensors get their own shard) and write
   model.safetensors.index.json + shard files.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # out/T5/.. -> out -> sandbox root
INPUTS = ROOT / "inputs"
OUT_DIR = ROOT / "out" / "T5"

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes, tensor data only

BASE_PATH = INPUTS / "base" / "model.safetensors"
LORA_PATH = INPUTS / "lora" / "adapter_model.safetensors"
LORA_CONFIG_PATH = INPUTS / "lora" / "adapter_config.json"

LORA_A_RE = re.compile(
    r"^base_model\.model\.(?P<base_name>.+)\.lora_A\.weight$"
)
LORA_B_RE = re.compile(
    r"^base_model\.model\.(?P<base_name>.+)\.lora_B\.weight$"
)


def main() -> None:
    config = json.loads(LORA_CONFIG_PATH.read_text())
    r = config["r"]
    alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    assert not fan_in_fan_out, (
        "fan_in_fan_out=True is not handled by this script "
        "(would require transposing B @ A before adding)"
    )
    scale = alpha / r

    base_sd = load_file(str(BASE_PATH))
    lora_sd = load_file(str(LORA_PATH))

    # Group lora_A / lora_B pairs by their target base tensor name.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora_sd.items():
        m = LORA_A_RE.match(name)
        if m:
            pairs.setdefault(m.group("base_name"), {})["A"] = tensor
            continue
        m = LORA_B_RE.match(name)
        if m:
            pairs.setdefault(m.group("base_name"), {})["B"] = tensor
            continue
        raise ValueError(f"unrecognized adapter tensor name: {name}")

    for base_name, ab in pairs.items():
        assert "A" in ab and "B" in ab, f"incomplete lora pair for {base_name}: {ab.keys()}"
        weight_name = f"{base_name}.weight"
        assert weight_name in base_sd, f"no base tensor for adapter target {base_name}"

    # --- Required check: exactly 16 adapter pairs found. ---
    assert len(pairs) == 16, f"expected 16 adapter pairs, found {len(pairs)}"

    merged_count = 0
    for base_name, ab in pairs.items():
        weight_name = f"{base_name}.weight"
        base_weight = base_sd[weight_name]
        base_dtype = base_weight.dtype

        A = ab["A"].to(torch.float32)  # [r, in]
        B = ab["B"].to(torch.float32)  # [out, r]
        delta = scale * (B @ A)  # [out, in], matches base layout (fan_in_fan_out=False)

        assert delta.shape == base_weight.shape, (
            f"shape mismatch for {weight_name}: delta {delta.shape} vs base {base_weight.shape}"
        )

        merged = base_weight.to(torch.float32) + delta
        base_sd[weight_name] = merged.to(base_dtype)
        merged_count += 1

    assert merged_count == 16, f"expected to merge 16 tensors, merged {merged_count}"

    # --- Required checks on the resulting state dict, before writing. ---
    assert not any("lora_" in name for name in base_sd), "adapter tensor leaked into output"

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert base_sd[qkv0].shape == (6144, 2048), (
        f"{qkv0} has shape {tuple(base_sd[qkv0].shape)}, expected (6144, 2048)"
    )

    assert len(base_sd) == 244, f"expected 244 tensors in output, got {len(base_sd)}"

    write_sharded(base_sd)


def write_sharded(state_dict: dict[str, torch.Tensor]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.element_size() * t.nelement()

    # Preserve base file's tensor order for stable, deterministic sharding.
    names = list(state_dict.keys())

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_bytes(state_dict[name])
        if size > MAX_SHARD_BYTES:
            # Oversized tensor: own shard, flush whatever was pending first.
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for i, shard_names in enumerate(shards, start=1):
        shard_file = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: state_dict[name] for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_file), metadata={"format": "pt"})
        for name, t in shard_tensors.items():
            weight_map[name] = shard_file
            total_size += tensor_bytes(t)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"wrote {len(state_dict)} tensors across {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
