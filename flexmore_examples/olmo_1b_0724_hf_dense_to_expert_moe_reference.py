from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from brainsurgery.engine.checkpoint_io import _load_state_dict_from_path, persist_state_dict


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    _load_state_dict_from_path(path, state_dict, max_io_workers=4)
    return state_dict


def _assert_shared_tensors_match(
    dense_a: dict[str, torch.Tensor],
    dense_b: dict[str, torch.Tensor],
) -> None:
    shared_keys = {
        "model.embed_tokens.weight",
        "lm_head.weight",
    }
    shared_keys.update(
        {
            f"model.layers.{layer}.self_attn.{proj}.weight"
            for layer in range(16)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
    )
    shared_keys.update(
        {
            f"model.layers.{layer}.mlp.{proj}.weight"
            for layer in range(16)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
    )

    for key in sorted(shared_keys):
        if key not in dense_a:
            raise KeyError(f"missing required key in dense_a: {key}")
        if key not in dense_b:
            raise KeyError(f"missing required key in dense_b: {key}")
        if not torch.equal(dense_a[key], dense_b[key]):
            raise ValueError(f"shared tensor mismatch: {key}")


def upcycle_hf_dense_state_dict_to_expert_moe(
    dense_a: dict[str, torch.Tensor],
    dense_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _assert_shared_tensors_match(dense_a, dense_b)

    out = {key: value.clone() for key, value in dense_a.items()}

    for layer in range(16):
        gate_proj = f"model.layers.{layer}.mlp.gate_proj.weight"
        up_proj = f"model.layers.{layer}.mlp.up_proj.weight"
        down_proj = f"model.layers.{layer}.mlp.down_proj.weight"
        q_proj = f"model.layers.{layer}.self_attn.q_proj.weight"

        out[f"model.layers.{layer}.mlp.experts.0.gate_proj.weight"] = dense_a[gate_proj].clone()
        out[f"model.layers.{layer}.mlp.experts.0.up_proj.weight"] = dense_a[up_proj].clone()
        out[f"model.layers.{layer}.mlp.experts.0.down_proj.weight"] = dense_a[down_proj].clone()

        out[f"model.layers.{layer}.mlp.experts.1.gate_proj.weight"] = dense_b[gate_proj].clone()
        out[f"model.layers.{layer}.mlp.experts.1.up_proj.weight"] = dense_b[up_proj].clone()
        out[f"model.layers.{layer}.mlp.experts.1.down_proj.weight"] = dense_b[down_proj].clone()

        hidden_size = dense_a[q_proj].shape[1]
        out[f"model.layers.{layer}.mlp.gate.weight"] = torch.zeros(
            (2, hidden_size),
            dtype=dense_a[q_proj].dtype,
        )

        del out[gate_proj]
        del out[up_proj]
        del out[down_proj]

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


def _example_moe_companion_config(
    dense_config: dict,
    *,
    num_experts: int,
    experts_per_token: int,
) -> dict:
    hidden_size = int(dense_config["hidden_size"])
    intermediate_size = int(dense_config["intermediate_size"])
    num_hidden_layers = int(dense_config["num_hidden_layers"])

    companion = dict(dense_config)
    companion.update(
        {
            "architectures": ["OlmoeForCausalLM"],
            "model_type": "olmoe_example",
            "num_local_experts": num_experts,
            "num_experts_per_tok": experts_per_token,
            "moe_intermediate_size": intermediate_size,
            "moe_router_hidden_size": hidden_size,
            "moe_num_hidden_layers": num_hidden_layers,
            "moe_weight_layout": {
                "router": "model.layers.{layer}.mlp.gate.weight",
                "expert_0": "model.layers.{layer}.mlp.experts.0.{proj}.weight",
                "expert_1": "model.layers.{layer}.mlp.experts.1.{proj}.weight",
                "removed_dense": [
                    "model.layers.{layer}.mlp.gate_proj.weight",
                    "model.layers.{layer}.mlp.up_proj.weight",
                    "model.layers.{layer}.mlp.down_proj.weight",
                ],
            },
            "brainsurgery_demo_only": True,
            "brainsurgery_note": (
                "Companion metadata for the BrainSurgery dense-to-expert-MoE example. "
                "This file documents the converted checkpoint layout but does not by itself "
                "guarantee compatibility with a specific external HF MoE runtime."
            ),
        }
    )
    return companion


def _write_example_metadata(
    *,
    model_a_dir: Path,
    model_b_dir: Path,
    target_dir: Path,
    num_experts: int,
    experts_per_token: int,
) -> None:
    dense_config = _load_json(model_a_dir / "config.json")
    companion_config = _example_moe_companion_config(
        dense_config,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
    )
    conversion_manifest = {
        "conversion": "hf_dense_to_expert_moe_example",
        "source_model_a": str(model_a_dir),
        "source_model_b": str(model_b_dir),
        "output_dir": str(target_dir),
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "shared_tensor_policy": "preserve m0; assert m1 equality",
        "router_initialization": "zeros",
        "removed_dense_mlp_tensors": True,
        "reference_script": "flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe_reference.py",
        "brainsurgery_plan": "flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe.yaml",
        "validation_plan": "flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe_validate.yaml",
    }
    _write_json(target_dir / "config.moe_example.json", companion_config)
    _write_json(target_dir / "brainsurgery_conversion.json", conversion_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference HF dense -> expert-MoE upcycling for the OLMo-1B 0724 example",
    )
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--copy-metadata", action="store_true")
    parser.add_argument("--write-example-config", action="store_true")
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--experts-per-token", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dense_a = _load_state_dict(args.model_a)
    dense_b = _load_state_dict(args.model_b)
    upcycled = upcycle_hf_dense_state_dict_to_expert_moe(dense_a, dense_b)

    written_path = persist_state_dict(
        upcycled,
        output_path=args.target,
        output_format="safetensors",
        shard_size=5 * 1024 * 1024 * 1024,
        sharded_output_root=args.target,
        max_io_workers=4,
    )

    if args.copy_metadata:
        target_dir = written_path.parent if written_path.name == "model.safetensors.index.json" else args.target
        _copy_non_weight_files(args.model_a, target_dir)
    else:
        target_dir = written_path.parent if written_path.name == "model.safetensors.index.json" else args.target

    if args.write_example_config:
        _write_example_metadata(
            model_a_dir=args.model_a,
            model_b_dir=args.model_b,
            target_dir=target_dir,
            num_experts=args.num_experts,
            experts_per_token=args.experts_per_token,
        )


if __name__ == "__main__":
    main()
