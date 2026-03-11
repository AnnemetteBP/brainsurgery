from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from .binary import BinaryMappingSpec, DestinationPolicy
from ..core import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    register_transform,
    require_numeric,
    select_tensor,
)
from ..utils import BinaryRefs, DeclarativeBinaryTransform, Docs

FillMode = Literal["constant", "rand", "tensor"]
RandDistribution = Literal["uniform", "normal"]


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


def _build_fill_spec(from_ref: TensorRef, to_ref: TensorRef, payload: dict) -> FillSpec:
    config = parse_fill_config(payload, op_name="fill", error_type=TransformError)
    return FillSpec(from_ref=from_ref, to_ref=to_ref, config=config)


def _fill_apply(
    spec: FillSpec, item: ResolvedMapping, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)
    template = select_tensor(src_sd[item.src_name], item.src_slice)
    dst_sd[item.dst_name] = build_filled_tensor_like(
        template, spec.config, TransformError
    )


class FillTransform(DeclarativeBinaryTransform[FillSpec]):
    name = "fill"
    error_type = TransformError
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
    docs = Docs(
        "Fills new destination tensors using the source tensor shape.",
        notes=(
            "Modes:",
            "  - constant: uses scalar 'value'",
            "  - rand: random fill ('distribution': uniform|normal)",
            "  - tensor: uses concrete payload 'values' (broadcasted if needed)",
        ),
        examples=("fill: { from: x, to: x_zeros, mode: constant, value: 0 }",),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_fill_spec)
    apply_fn = staticmethod(_fill_apply)


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
        if not isinstance(raw_dist, str) or raw_dist.strip().lower() not in {
            "uniform",
            "normal",
        }:
            raise error_type(f"{op_name}.distribution must be one of: uniform, normal")
        distribution = raw_dist.strip().lower()  # type: ignore[assignment]
        if distribution == "uniform":
            if "low" in payload:
                low = require_numeric(payload, op_name=op_name, key="low")
            if "high" in payload:
                high = require_numeric(payload, op_name=op_name, key="high")
            if low >= high:
                raise error_type(
                    f"{op_name} requires low < high for uniform distribution"
                )
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
    value = torch.as_tensor(
        config.values_payload, dtype=template.dtype, device=template.device
    )
    if value.shape == template.shape:
        return value.clone()
    try:
        expanded = value.expand_as(template)
    except RuntimeError as exc:
        raise error_type(
            f"fill.values cannot broadcast to target shape {tuple(template.shape)} from {tuple(value.shape)}"
        ) from exc
    return expanded.clone()


register_transform(FillTransform())
