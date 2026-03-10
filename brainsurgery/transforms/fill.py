from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    parse_slice,
    register_transform,
    require_numeric,
    select_tensor,
    validate_payload_keys,
)

FillMode = Literal["constant", "rand", "tensor"]
RandDistribution = Literal["uniform", "normal"]


class FillTransformError(TransformError):
    pass


@dataclass(frozen=True)
class FillConfig:
    mode: FillMode
    constant_value: float | None
    values_payload: object | None
    distribution: RandDistribution
    low: float
    high: float
    mean: float
    std: float
    seed: int | None


@dataclass(frozen=True)
class FillSpec(BinaryMappingSpec):
    config: FillConfig


class FillTransform(BinaryMappingTransform[FillSpec]):
    name = "fill"
    error_type = FillTransformError
    spec_type = FillSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {
        "from",
        "to",
        "mode",
        "value",
        "values",
        "distribution",
        "low",
        "high",
        "mean",
        "std",
        "seed",
    }
    required_keys = {"from", "to", "mode"}
    progress_desc = "Applying fill transforms"
    help_text = (
        "Fills new destination tensors using the source tensor shape.\n"
        "\n"
        "Modes:\n"
        "  - constant: uses scalar 'value'\n"
        "  - rand: random fill ('distribution': uniform|normal)\n"
        "  - tensor: uses concrete payload 'values' (broadcasted if needed)\n"
        "\n"
        "Source references may be sliced. Destination tensors must not exist.\n"
        "\n"
        "Example:\n"
        "  fill: { from: x, to: x_zeros, mode: constant, value: 0 }"
    )

    def compile(self, payload: dict, default_model: str | None) -> FillSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )
        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)
        config = parse_fill_config(payload, op_name=self.name, error_type=FillTransformError)
        return FillSpec(from_ref=from_ref, to_ref=to_ref, config=config)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise FillTransformError("fill destination must not be sliced")

    def apply_item(self, spec: FillSpec, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)
        template = select_tensor(src_sd[item.src_name], item.src_slice)
        dst_sd[item.dst_name] = build_filled_tensor_like(template, spec.config, FillTransformError)


def parse_fill_config(
    payload: dict,
    *,
    op_name: str,
    error_type: type[TransformError],
) -> FillConfig:
    raw_mode = payload.get("mode")
    if not isinstance(raw_mode, str) or not raw_mode:
        raise error_type(f"{op_name}.mode must be one of: constant, rand, tensor")
    mode = raw_mode.strip().lower()
    if mode not in {"constant", "rand", "tensor"}:
        raise error_type(f"{op_name}.mode must be one of: constant, rand, tensor")

    constant_value: float | None = None
    values_payload: object | None = None
    distribution: RandDistribution = "uniform"
    low = 0.0
    high = 1.0
    mean = 0.0
    std = 1.0
    seed: int | None = None

    if "seed" in payload:
        raw_seed = payload.get("seed")
        if not isinstance(raw_seed, int):
            raise error_type(f"{op_name}.seed must be an integer when provided")
        seed = raw_seed

    if mode == "constant":
        if "value" not in payload:
            raise error_type(f"{op_name}.value is required when mode=constant")
        constant_value = require_numeric(payload, op_name=op_name, key="value")
    elif mode == "tensor":
        if "values" not in payload:
            raise error_type(f"{op_name}.values is required when mode=tensor")
        values_payload = payload.get("values")
    else:
        raw_dist = payload.get("distribution", "uniform")
        if not isinstance(raw_dist, str) or raw_dist.strip().lower() not in {"uniform", "normal"}:
            raise error_type(f"{op_name}.distribution must be one of: uniform, normal")
        distribution = raw_dist.strip().lower()  # type: ignore[assignment]
        if distribution == "uniform":
            if "low" in payload:
                low = require_numeric(payload, op_name=op_name, key="low")
            if "high" in payload:
                high = require_numeric(payload, op_name=op_name, key="high")
            if low >= high:
                raise error_type(f"{op_name} requires low < high for uniform distribution")
        else:
            if "mean" in payload:
                mean = require_numeric(payload, op_name=op_name, key="mean")
            if "std" in payload:
                std = require_numeric(payload, op_name=op_name, key="std")
            if std <= 0:
                raise error_type(f"{op_name}.std must be > 0 for normal distribution")

    return FillConfig(
        mode=mode,  # type: ignore[arg-type]
        constant_value=constant_value,
        values_payload=values_payload,
        distribution=distribution,
        low=low,
        high=high,
        mean=mean,
        std=std,
        seed=seed,
    )


def build_filled_tensor_like(
    template: torch.Tensor,
    config: FillConfig,
    error_type: type[TransformError],
) -> torch.Tensor:
    if config.mode == "constant":
        assert config.constant_value is not None
        return torch.full_like(template, config.constant_value)

    if config.mode == "rand":
        out = torch.empty_like(template)
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=out.device)
            generator.manual_seed(config.seed)
        if config.distribution == "uniform":
            out.uniform_(config.low, config.high, generator=generator)
        else:
            out.normal_(config.mean, config.std, generator=generator)
        return out

    assert config.mode == "tensor"
    value = torch.as_tensor(config.values_payload, dtype=template.dtype, device=template.device)
    if value.shape == template.shape:
        return value.clone()
    try:
        expanded = value.expand_as(template)
    except RuntimeError as exc:
        raise error_type(
            f"fill.values cannot broadcast to target shape {tuple(template.shape)} from {tuple(value.shape)}"
        ) from exc
    return expanded.clone()


def _unit_test_fill_compile_requires_mode_specific_payload() -> None:
    try:
        FillTransform().compile(
            {"from": "x", "to": "y", "mode": "constant"},
            default_model="m",
        )
    except FillTransformError as exc:
        assert "fill.value is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected fill.value validation error")


def _unit_test_fill_tensor_mode_broadcasts_values() -> None:
    template = torch.zeros((2, 2), dtype=torch.float32)
    config = parse_fill_config(
        {"mode": "tensor", "values": [1.0, 2.0]},
        op_name="fill",
        error_type=FillTransformError,
    )
    out = build_filled_tensor_like(template, config, FillTransformError)
    assert out.tolist() == [[1.0, 2.0], [1.0, 2.0]]


def _unit_test_fill_rand_mode_is_seeded() -> None:
    template = torch.zeros((3,), dtype=torch.float32)
    config = parse_fill_config(
        {"mode": "rand", "seed": 7},
        op_name="fill",
        error_type=FillTransformError,
    )
    a = build_filled_tensor_like(template, config, FillTransformError)
    b = build_filled_tensor_like(template, config, FillTransformError)
    assert torch.equal(a, b)


__unit_tests__ = [
    _unit_test_fill_compile_requires_mode_specific_payload,
    _unit_test_fill_tensor_mode_broadcasts_values,
    _unit_test_fill_rand_mode_is_seeded,
]


register_transform(FillTransform())
