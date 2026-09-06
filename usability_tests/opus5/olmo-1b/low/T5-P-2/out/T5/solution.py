"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = Path("inputs/base")
LORA_FILE = Path("inputs/lora/adapter_model.safetensors")
LORA_CONFIG = Path("inputs/lora/adapter_config.json")
OUT_DIR = Path("out/T5")
MAX_SHARD_BYTES = 512 * 1024 * 1024


def load_dir(path: Path) -> dict[str, torch.Tensor]:
    index = json.loads((path / "model.safetensors.index.json").read_text())
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(path / shard, framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    return tensors


def load_file_st(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    return tensors


def main() -> None:
    cfg = json.loads(LORA_CONFIG.read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r
    assert not cfg.get("fan_in_fan_out", False), "fan_in_fan_out=True is not handled"

    base = load_dir(BASE_DIR)
    lora = load_file_st(LORA_FILE)
    assert len(base) == 114, f"expected 114 base tensors, got {len(base)}"

    # Pair up lora_A / lora_B by their common prefix and map to base names.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora.items():
        if ".lora_A.weight" in name:
            key, side = name.split(".lora_A.weight")[0], "A"
        elif ".lora_B.weight" in name:
            key, side = name.split(".lora_B.weight")[0], "B"
        else:
            raise AssertionError(f"unexpected adapter tensor: {name}")
        pairs.setdefault(key, {})[side] = tensor

    merged = 0
    for key, sides in sorted(pairs.items()):
        assert set(sides) == {"A", "B"}, f"incomplete adapter pair for {key}"
        A, B = sides["A"].float(), sides["B"].float()
        assert A.shape[0] == r and B.shape[1] == r, f"rank mismatch for {key}: {A.shape} {B.shape}"

        base_name = key
        for prefix in ("base_model.model.",):
            assert base_name.startswith(prefix), f"unexpected adapter prefix: {key}"
            base_name = base_name[len(prefix) :]
        base_name += ".weight"
        assert base_name in base, f"no base tensor for adapter {key} (looked for {base_name})"

        w = base[base_name]
        delta = scale * (B @ A)  # fan_in_fan_out=false -> [out, in], same layout as w
        assert delta.shape == w.shape, f"delta {delta.shape} != base {w.shape} for {base_name}"
        assert w.dtype == torch.float32, f"{base_name} is {w.dtype}, expected float32"
        base[base_name] = (w.float() + delta).to(torch.float32)
        merged += 1

    # Required checks.
    assert merged == 32, f"expected 32 merged adapter pairs, got {merged}"
    assert not [n for n in base if "lora_" in n], "adapter tensors leaked into the output"
    q0 = base["model.layers.0.self_attn.q_proj.weight"]
    assert tuple(q0.shape) == (2048, 2048), f"q_proj shape changed: {tuple(q0.shape)}"
    assert len(base) == 114, f"expected 114 output tensors, got {len(base)}"

    # Greedy sharding in sorted name order; an oversized tensor lands alone.
    names = sorted(base)
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        nbytes = base[name].numel() * base[name].element_size()
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
        if current_bytes > MAX_SHARD_BYTES:  # single oversized tensor, flush it alone
            shards.append(current)
            current, current_bytes = [], 0
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.safetensors"):
        stale.unlink()

    total = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        payload = {n: base[n].contiguous() for n in shard}
        shard_bytes = sum(t.numel() * t.element_size() for t in payload.values())
        assert shard_bytes <= MAX_SHARD_BYTES or len(payload) == 1, (
            f"{fname} holds {shard_bytes} bytes over the budget with {len(payload)} tensors"
        )
        total_size += shard_bytes
        save_file(payload, os.fspath(OUT_DIR / fname), metadata={"format": "pt"})
        for n in shard:
            weight_map[n] = fname

    assert len(weight_map) == 114, f"weight_map covers {len(weight_map)} tensors"
    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
        + "\n"
    )
    print(f"merged {merged} adapter pairs (scale={scale})")
    print(f"wrote {len(weight_map)} tensors into {total} shards, {total_size} bytes")


if __name__ == "__main__":
    main()
