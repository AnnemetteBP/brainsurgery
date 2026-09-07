"""T5: fold a PEFT LoRA adapter into Pythia-1B base weights and export sharded.

Plain script on safetensors + torch + huggingface_hub's sharding helper.
No model instantiation: the merge is done directly on the checkpoint tensors.
"""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

BASE = Path("inputs/base/model.safetensors")
LORA_DIR = Path("inputs/lora")
OUT = Path("out/T5")
MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912

LORA_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_(?P<ab>A|B)\.weight$")


def main() -> None:
    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    scale = alpha / r

    base_sd = load_file(BASE)
    lora_sd = load_file(LORA_DIR / "adapter_model.safetensors")

    # Group adapter factors by the base tensor name they adapt.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora_sd.items():
        m = LORA_RE.match(name)
        if m is None:
            raise ValueError(f"unrecognized adapter tensor name: {name}")
        pairs.setdefault(m.group("base") + ".weight", {})[m.group("ab")] = tensor

    n_pairs = 0
    for target, factors in sorted(pairs.items()):
        if set(factors) != {"A", "B"}:
            raise ValueError(f"{target}: incomplete adapter pair, got {sorted(factors)}")
        if target not in base_sd:
            raise KeyError(f"adapter targets {target}, which is not in the base checkpoint")
        A, B = factors["A"].float(), factors["B"].float()
        if A.shape[0] != r or B.shape[1] != r:
            raise ValueError(f"{target}: factor ranks {A.shape}/{B.shape} disagree with r={r}")
        delta = scale * (B @ A)  # [out, in], float32
        if fan_in_fan_out:
            delta = delta.T
        w = base_sd[target]
        if delta.shape != w.shape:
            raise ValueError(f"{target}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        base_sd[target] = (w.float() + delta).to(w.dtype)
        n_pairs += 1

    # --- required checks, all before writing ---
    if n_pairs != 16:
        raise AssertionError(f"expected 16 adapter pairs merged, got {n_pairs}")
    offenders = [k for k in base_sd if "lora_" in k]
    if offenders:
        raise AssertionError(f"adapter tensors leaked into the output: {offenders[:5]}")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base_sd[probe].shape) != (6144, 2048):
        raise AssertionError(f"{probe} has shape {tuple(base_sd[probe].shape)}, expected (6144, 2048)")
    if len(base_sd) != 244:
        raise AssertionError(f"expected 244 output tensors, got {len(base_sd)}")

    OUT.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(
        base_sd,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD_BYTES,
    )
    for filename, keys in split.filename_to_tensors.items():
        shard = {k: base_sd[k].contiguous() for k in keys}
        nbytes = sum(t.numel() * t.element_size() for t in shard.values())
        if nbytes > MAX_SHARD_BYTES and len(shard) > 1:
            raise AssertionError(f"{filename}: {nbytes} bytes over budget with {len(shard)} tensors")
        save_file(shard, OUT / filename, metadata={"format": "pt"})
    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"merged {n_pairs} adapter pairs (scale={scale}), wrote {len(base_sd)} tensors "
          f"into {len(split.filename_to_tensors)} shards under {OUT}")


if __name__ == "__main__":
    main()
