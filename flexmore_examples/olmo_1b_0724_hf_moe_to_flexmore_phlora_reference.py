from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from brainsurgery.algorithms.phlora import PhloraSvdCache, compute_phlora_factors
from brainsurgery.engine.checkpoint_io import _load_state_dict_from_path, persist_state_dict


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    _load_state_dict_from_path(path, state_dict, max_io_workers=4)
    return state_dict


def upcycle_moe_state_dict_to_flexmore_phlora(
    source: dict[str, torch.Tensor],
    *,
    rank: int,
) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in source.items()}
    cache = PhloraSvdCache()

    for layer in range(16):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expert0_key = f"model.layers.{layer}.mlp.experts.0.{proj}.weight"
            expert1_key = f"model.layers.{layer}.mlp.experts.1.{proj}.weight"
            factor_a_key = f"model.layers.{layer}.mlp.experts.1.{proj}.phlora_a.weight"
            factor_b_key = f"model.layers.{layer}.mlp.experts.1.{proj}.phlora_b.weight"
            delta = source[expert1_key] - source[expert0_key]
            factor_a, factor_b = compute_phlora_factors(
                delta,
                rank,
                cache=cache,
                cache_key=expert1_key,
                error_type=ValueError,
                op_name="phlora",
                tensor_name=expert1_key,
            )
            out[factor_a_key] = factor_a.to(dtype=source[expert1_key].dtype, device=source[expert1_key].device)
            out[factor_b_key] = factor_b.to(dtype=source[expert1_key].dtype, device=source[expert1_key].device)
            del out[expert1_key]

    return out


def _copy_non_weight_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name.startswith("model"):
            continue
        if item.is_file():
            shutil.copy2(item, target_dir / item.name)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_example_metadata(
    *,
    model_dir: Path,
    target_dir: Path,
    rank: int,
) -> None:
    dense_config = _load_json(model_dir / "config.json")
    conversion_manifest = {
        "conversion": "hf_moe_to_flexmore_phlora_example",
        "source_model": str(model_dir),
        "output_dir": str(target_dir),
        "rank": rank,
        "dense_anchor_expert": 0,
        "factorized_expert": 1,
        "reference_script": "flexmore_examples/olmo_1b_0724_hf_moe_to_flexmore_phlora_reference.py",
        "brainsurgery_plan": "flexmore_examples/olmo_1b_0724_hf_moe_to_flexmore_phlora.yaml",
        "validation_plan": "flexmore_examples/olmo_1b_0724_hf_moe_to_flexmore_phlora_validate.yaml",
        "note": (
            "Expert 1 is stored as PHLoRA delta factors relative to dense expert 0. "
            "The output is a factorized checkpoint layout."
        ),
    }
    dense_config["brainsurgery_flexmore_phlora_example"] = {
        "rank": rank,
        "dense_anchor_expert": 0,
        "factorized_expert": 1,
    }
    _write_json(target_dir / "config.flexmore_phlora_example.json", dense_config)
    _write_json(target_dir / "brainsurgery_conversion.json", conversion_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference MoE -> factorized FlexMoRE PHLoRA conversion for the OLMo-1B 0724 example",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--copy-metadata", action="store_true")
    parser.add_argument("--write-example-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = _load_state_dict(args.model)
    rewritten = upcycle_moe_state_dict_to_flexmore_phlora(source, rank=args.rank)

    written_path = persist_state_dict(
        rewritten,
        output_path=args.target,
        output_format="safetensors",
        shard_size=5 * 1024 * 1024 * 1024,
        sharded_output_root=args.target,
        max_io_workers=4,
    )

    target_dir = written_path.parent if written_path.name == "model.safetensors.index.json" else args.target

    if args.copy_metadata:
        _copy_non_weight_files(args.model, target_dir)
    if args.write_example_config:
        _write_example_metadata(
            model_dir=args.model,
            target_dir=target_dir,
            rank=args.rank,
        )


if __name__ == "__main__":
    main()
