from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from revision_tests.behavioral.run_cuda_paper_matrix import (
    load_protocol,
    paper_totals,
    render_latex,
    render_markdown,
    render_paper_text,
    validate_result,
    write_plan,
)
from revision_tests.behavioral.test_paper_analysis import complete_row


def result_fixture():
    rows = [complete_row()]
    aggregate = {
        "prompt_count": 1,
        "reference_mean_perplexity": 2.0,
        "transformed_mean_perplexity": 2.0,
        "mean_perplexity_ratio": 1.0,
        "maximum_perplexity_ratio": 1.0,
        "mean_last_token_logit_cosine": 1.0,
        "minimum_last_token_logit_cosine": 1.0,
        "mean_last_token_absolute_difference": 0.0,
        "maximum_last_token_absolute_difference": 0.0,
        "mean_full_sequence_logit_cosine": 1.0,
        "minimum_full_sequence_logit_cosine": 1.0,
        "full_sequence_positions_compared": 12,
        "mean_full_sequence_absolute_difference": 0.0,
        "maximum_full_sequence_absolute_difference": 0.0,
        "top1_matches": 1,
        "output_exact_matches": 1,
        "mean_output_char_similarity": 1.0,
        "mean_output_token_sequence_similarity": 1.0,
        "mean_output_token_bag_cosine": 1.0,
        "mcq_count": 1,
        "mcq_prediction_matches": 1,
    }
    return {
        "protocol_id": "eacl2027_behavioral_paper_v4",
        "thresholds_passed": True,
        "reported_eligible": False,
        "aggregate": aggregate,
        "by_manifest_dimension": {"source": {}, "task_category": {}, "language_code": {}},
        "prompts": rows,
    }


def test_protocol_retains_all_original_metrics():
    protocol = load_protocol()
    assert protocol["reporting"]["forbid_nonfinite_metrics"] is True
    assert protocol["reporting"]["restored_tensor_reference"] == (
        "independent_pytorch_forward_backward"
    )


def test_result_gate_accepts_complete_result():
    validate_result(result_fixture(), 1)


def test_result_gate_rejects_missing_aggregate():
    result = deepcopy(result_fixture())
    del result["aggregate"]["mean_perplexity_ratio"]
    with pytest.raises(ValueError, match="aggregate evidence is incomplete"):
        validate_result(result, 1)


def test_tables_are_generated_only_from_complete_evidence():
    result = result_fixture()
    case = {
        "display": "Synthetic",
        "tensor_oracle": {
            "brainsurgery_scaled": {"tensors_checked": 2},
            "brainsurgery_restored": {"tensors_checked": 2},
        },
        "comparisons": {
            "imperative_equivalence": {"aggregate": result["aggregate"]},
            "regression_preservation": {"aggregate": result["aggregate"]},
        },
    }
    evidence = {
        "protocol_id": result["protocol_id"],
        "run_id": "synthetic",
        "git_commit": "abc",
        "gpu": "synthetic",
        "results": [case],
    }
    assert "1.00000000" in render_markdown(evidence)
    assert "1.00000000" in render_latex(evidence)
    assert paper_totals(evidence)["prompts"] == 1
    assert "1/1 top-1 agreement" in render_paper_text(evidence)
    assert "independent direct-PyTorch forward--backward outputs" in render_paper_text(evidence)
    assert "\\paragraph{Behavioral" in render_paper_text(evidence, latex=True)


def test_plans_encode_meaningful_forward_and_round_trip(tmp_path: Path):
    protocol = load_protocol()
    scaled = tmp_path / "scaled.yaml"
    restored = tmp_path / "restored.yaml"
    source = tmp_path / "source"
    write_plan(scaled, source, tmp_path / "scaled", protocol, restore=False)
    write_plan(restored, source, tmp_path / "restored", protocol, restore=True)

    scaled_text = scaled.read_text(encoding="utf-8")
    restored_text = restored.read_text(encoding="utf-8")
    assert "by: 0.5" in scaled_text
    assert "by: 0.5" in restored_text
    assert "by: 2.0" in restored_text
    assert restored_text.count("move:") == 2
