from __future__ import annotations

import pytest
import torch

from revision_tests.behavioral.run_paper_analysis import (
    REQUIRED_PROMPT_METRICS,
    prompt_forward,
    require_complete_rows,
    summarize_rows,
)


def complete_row(**changes):
    row = {
        "prompt_id": "p1",
        "expected_kind": "multiple_choice",
        "reference_perplexity": 2.0,
        "transformed_perplexity": 2.0,
        "perplexity_ratio": 1.0,
        "last_token_logit_cosine": 1.0,
        "last_token_mean_absolute_difference": 0.0,
        "last_token_max_absolute_difference": 0.0,
        "top1_match": True,
        "full_sequence_positions_compared": 12,
        "full_sequence_mean_logit_cosine": 1.0,
        "full_sequence_min_logit_cosine": 1.0,
        "full_sequence_mean_abs_logit_diff": 0.0,
        "full_sequence_max_abs_logit_diff": 0.0,
        "output_exact_match": True,
        "output_char_similarity": 1.0,
        "output_token_sequence_similarity": 1.0,
        "output_token_bag_cosine": 1.0,
        "mcq_prediction_match": True,
    }
    row.update(changes)
    return row


def test_summary_preserves_every_old_paper_metric():
    summary = summarize_rows([complete_row(), complete_row(prompt_id="p2")])
    assert summary["reference_mean_perplexity"] == 2.0
    assert summary["transformed_mean_perplexity"] == 2.0
    assert summary["mean_perplexity_ratio"] == 1.0
    assert summary["mean_last_token_logit_cosine"] == 1.0
    assert summary["mean_full_sequence_logit_cosine"] == 1.0
    assert summary["output_exact_matches"] == 2
    assert summary["mcq_prediction_matches"] == 2


@pytest.mark.parametrize("missing", sorted(REQUIRED_PROMPT_METRICS))
def test_completeness_gate_rejects_every_missing_metric(missing):
    row = complete_row()
    del row[missing]
    with pytest.raises(ValueError, match="missing"):
        require_complete_rows([row], 1)


def test_completeness_gate_rejects_nonfinite_metric():
    with pytest.raises(ValueError, match="not finite"):
        require_complete_rows([complete_row(reference_perplexity=float("nan"))], 1)


class NonfiniteModel(torch.nn.Module):
    def __init__(self, *, loss: float, logit: float):
        super().__init__()
        self.loss = loss
        self.logit = logit

    def forward(self, **_kwargs):
        return type(
            "Output",
            (),
            {
                "loss": torch.tensor(self.loss),
                "logits": torch.tensor([[[self.logit, 0.0]]]),
            },
        )()


@pytest.mark.parametrize(
    ("loss", "logit", "message"),
    [(float("nan"), 0.0, "loss"), (0.0, float("inf"), "logits")],
)
def test_prompt_forward_rejects_nonfinite_model_outputs(loss, logit, message):
    with pytest.raises(FloatingPointError, match=message):
        prompt_forward(
            NonfiniteModel(loss=loss, logit=logit),
            torch.tensor([[1]]),
            torch.tensor([[1]]),
        )
