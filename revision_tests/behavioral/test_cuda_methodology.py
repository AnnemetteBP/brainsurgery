from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from revision_tests.behavioral.run_cuda_methodology import (
    PROTOCOL_ID,
    load_protocol,
    write_bs_plan,
)


def test_protocol_separates_primary_and_ablation() -> None:
    protocol = load_protocol()
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["structural_losslessness"]["arithmetic_allowed"] is False
    assert protocol["structural_losslessness"]["model_ids"] == protocol["expected_model_ids"]
    assert protocol["pythia_precision_ablation"]["model_ids"] == ["P01", "P02", "P03", "P04"]
    fp32 = protocol["pythia_precision_ablation"]["arms"][1]
    assert fp32 == {
        "id": "fp32_throughout",
        "storage_dtype": "float32",
        "inference_dtype": "float32",
        "cast_back_to_float16": False,
    }


def test_structural_plan_contains_moves_without_arithmetic(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    transforms = [
        {"move": {"from": r"^(.*)$", "to": r"__structural_roundtrip.\1"}},
        {"move": {"from": r"^__structural_roundtrip\.(.*)$", "to": r"\1"}},
    ]
    write_bs_plan(tmp_path / "source", tmp_path / "output", transforms, plan)
    payload = yaml.safe_load(plan.read_text())
    assert payload["transforms"] == transforms
    assert all("scale_" not in transform for transform in payload["transforms"])


def test_independent_baseline_roundtrip_and_fp32_cast(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {
            "layer.weight": torch.tensor([[1.0, -2.0]], dtype=torch.float16),
            "layer.bias": torch.tensor([3.0], dtype=torch.float16),
        },
        str(source / "model.safetensors"),
    )
    script = Path("revision_tests/behavioral/methodology_baseline.py")
    structural = tmp_path / "structural"
    subprocess.run(
        [
            sys.executable, str(script), "--input", str(source), "--output", str(structural),
            "--operation", "structural-roundtrip", "--shard-size-bytes", "1024",
        ],
        check=True,
    )
    cast = tmp_path / "fp32"
    subprocess.run(
        [
            sys.executable, str(script), "--input", str(source), "--output", str(cast),
            "--operation", "cast-float32", "--shard-size-bytes", "1024",
        ],
        check=True,
    )
    original = load_file(str(source / "model.safetensors"))
    structural_state = load_file(str(next(structural.glob("*.safetensors"))))
    cast_state = load_file(str(next(cast.glob("*.safetensors"))))
    assert original.keys() == structural_state.keys()
    assert all(torch.equal(original[key], structural_state[key]) for key in original)
    assert all(tensor.dtype == torch.float32 for tensor in cast_state.values())
