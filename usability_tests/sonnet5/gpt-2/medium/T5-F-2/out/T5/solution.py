"""T5: LoRA adapter merge with sharded export (GPT-2, condition F).

Plain script on top of `safetensors` + `torch`. No model is instantiated:
the adapter is folded directly into the base checkpoint's tensors, then the
result is re-sharded to a 100 MiB (104,857,600 byte) per-shard budget.

We use raw safetensors/torch rather than `peft.merge_and_unload` because
that API requires instantiating the full GPT2LMHeadModel and its PEFT
wrapper just to get back a state dict we then have to re-shard ourselves
anyway; operating on the checkpoint directly is simpler, avoids extra
dependencies at merge time, and makes the sharding budget and the required
checks explicit and easy to verify.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_PATH = REPO_ROOT / "inputs" / "base" / "model.safetensors"
LORA_PATH = REPO_ROOT / "inputs" / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = REPO_ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = REPO_ROOT / "out" / "T5"
SHARD_BUDGET_BYTES = 100 * 1024 * 1024  # 104,857,600 bytes, tensor data only

LORA_A_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_B\.weight$")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def main() -> None:
    adapter_config = json.loads(ADAPTER_CONFIG_PATH.read_text())
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    if not fan_in_fan_out:
        raise AssertionError(
            "expected fan_in_fan_out=True (Conv1D base layout); adapter_config.json says otherwise"
        )
    scale = lora_alpha / r

    base = load_state_dict(BASE_PATH)
    lora = load_state_dict(LORA_PATH)

    # Pair up lora_A / lora_B tensors by their base module name.
    a_by_base: dict[str, torch.Tensor] = {}
    b_by_base: dict[str, torch.Tensor] = {}
    for key, tensor in lora.items():
        m = LORA_A_RE.match(key)
        if m:
            a_by_base[m.group("base")] = tensor
            continue
        m = LORA_B_RE.match(key)
        if m:
            b_by_base[m.group("base")] = tensor
            continue
        raise AssertionError(f"unrecognized adapter tensor name: {key}")

    if a_by_base.keys() != b_by_base.keys():
        raise AssertionError(
            f"lora_A/lora_B key mismatch: A only={a_by_base.keys() - b_by_base.keys()}, "
            f"B only={b_by_base.keys() - a_by_base.keys()}"
        )

    pairs = sorted(a_by_base.keys())
    if len(pairs) != 12:
        raise AssertionError(f"expected exactly 12 adapter pairs, found {len(pairs)}: {pairs}")

    merged_count = 0
    for module_name in pairs:
        base_name = f"{module_name}.weight"
        if base_name not in base:
            raise AssertionError(f"adapter targets base tensor {base_name!r} not present in base checkpoint")
        A = a_by_base[module_name].to(torch.float32)
        B = b_by_base[module_name].to(torch.float32)
        base_weight = base[base_name]
        if base_weight.dtype != torch.float32:
            raise AssertionError(f"{base_name}: expected float32 base tensor, got {base_weight.dtype}")

        delta = scale * (B @ A).T  # Conv1D layout: [in, out]
        if delta.shape != base_weight.shape:
            raise AssertionError(
                f"{base_name}: delta shape {tuple(delta.shape)} != base shape {tuple(base_weight.shape)}"
            )
        base[base_name] = (base_weight + delta).contiguous()
        merged_count += 1

    if merged_count != 12:
        raise AssertionError(f"expected to merge exactly 12 tensors, merged {merged_count}")

    # --- Required checks -------------------------------------------------
    if any("lora_" in name for name in base):
        raise AssertionError("adapter tensor leaked into the merged state dict")
    if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        raise AssertionError(
            f"h.0.attn.c_attn.weight has shape {tuple(base['h.0.attn.c_attn.weight'].shape)}, expected (768, 2304)"
        )
    if len(base) != 160:
        raise AssertionError(f"expected exactly 160 tensors in the output, found {len(base)}")

    # --- Sharding ----------------------------------------------------------
    # Greedy bin-packing in checkpoint order, budget = 100 MiB of tensor data
    # per shard (header bytes excluded). A tensor that alone exceeds the
    # budget (wte.weight, ~154 MB) gets its own shard.
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names = list(base.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_nbytes(base[name])
        if current and current_bytes + size > SHARD_BUDGET_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    for shard_names in shards:
        total = sum(tensor_nbytes(base[n]) for n in shard_names)
        if total > SHARD_BUDGET_BYTES and len(shard_names) > 1:
            raise AssertionError(f"shard exceeds budget with more than one tensor: {shard_names}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_shards = len(shards)
    width = max(5, len(str(num_shards)))
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:0{width}d}-of-{num_shards:0{width}d}.safetensors"
        shard_tensors = {name: base[name] for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename), metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(base[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"Merged {merged_count} adapter pairs into {len(base)} base tensors.")
    print(f"Wrote {num_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
