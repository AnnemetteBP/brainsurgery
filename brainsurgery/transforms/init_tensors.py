from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..core import TensorRef, UnarySpec, UnaryTransform
from ..core import must_model
from ..core import register_transform
from ..core import require_numeric
from ..core import StateDictProvider, TransformError

RandDistribution = Literal["uniform", "normal"]


@dataclass(frozen=True)
class ShapeCreateSpec(UnarySpec):
    target_name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RandSpec(ShapeCreateSpec):
    distribution: RandDistribution
    low: float
    high: float
    mean: float
    std: float
    seed: int | None


def _parse_shape(raw: object, *, op_name: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise TransformError(f"{op_name}.shape must be a non-empty list of positive integers")
    if not all(isinstance(x, int) for x in raw):
        raise TransformError(f"{op_name}.shape must be a non-empty list of positive integers")
    if any(x <= 0 for x in raw):
        raise TransformError(f"{op_name}.shape must be a non-empty list of positive integers")
    return tuple(raw)


def _require_literal_target_name(target_ref: TensorRef, *, op_name: str) -> str:
    if not isinstance(target_ref.expr, str):
        raise TransformError(f"{op_name}.target must resolve to a single tensor name")
    return target_ref.expr


class _ShapeCreateUnaryTransform(UnaryTransform[ShapeCreateSpec]):
    progress_desc = None

    def resolve_items(
        self,
        spec: ShapeCreateSpec,
        provider: StateDictProvider,
    ) -> list[str]:
        del provider
        return [spec.target_name]

    def validate_resolved_items(
        self,
        spec: ShapeCreateSpec,
        items: list[str],
        provider: StateDictProvider,
    ) -> None:
        del items
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        if spec.target_name in sd:
            raise TransformError(
                f"{self.name} destination already exists: {model}::{spec.target_name}"
            )


class ZeroesTransform(_ShapeCreateUnaryTransform):
    name = "zeroes"
    error_type = TransformError
    spec_type = ShapeCreateSpec
    allowed_keys = {"target", "shape"}
    required_keys = {"target", "shape"}
    help_text = (
        "Creates a new tensor filled with zeros.\n"
        "\n"
        "Requires a non-existing destination tensor name and an explicit shape.\n"
        "\n"
        "Examples:\n"
        "  zeroes: { target: work::delta, shape: [1024] }\n"
        "  zeroes: { target: model::matrix, shape: [64, 64] }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ShapeCreateSpec:
        name = _require_literal_target_name(target_ref, op_name=self.name)
        shape = _parse_shape(payload.get("shape"), op_name=self.name)
        return ShapeCreateSpec(target_ref=target_ref, target_name=name, shape=shape)

    def apply_to_target(
        self, spec: ShapeCreateSpec, name: str, provider: StateDictProvider
    ) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        sd[name] = torch.zeros(spec.shape, dtype=torch.float32)


class OnesTransform(_ShapeCreateUnaryTransform):
    name = "ones"
    error_type = TransformError
    spec_type = ShapeCreateSpec
    allowed_keys = {"target", "shape"}
    required_keys = {"target", "shape"}
    help_text = (
        "Creates a new tensor filled with ones.\n"
        "\n"
        "Requires a non-existing destination tensor name and an explicit shape.\n"
        "\n"
        "Examples:\n"
        "  ones: { target: work::mask, shape: [1024] }\n"
        "  ones: { target: model::matrix, shape: [64, 64] }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ShapeCreateSpec:
        name = _require_literal_target_name(target_ref, op_name=self.name)
        shape = _parse_shape(payload.get("shape"), op_name=self.name)
        return ShapeCreateSpec(target_ref=target_ref, target_name=name, shape=shape)

    def apply_to_target(
        self, spec: ShapeCreateSpec, name: str, provider: StateDictProvider
    ) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        sd[name] = torch.ones(spec.shape, dtype=torch.float32)


class RandTransform(_ShapeCreateUnaryTransform):
    name = "rand"
    error_type = TransformError
    spec_type = RandSpec
    allowed_keys = {"target", "shape", "distribution", "low", "high", "mean", "std", "seed"}
    required_keys = {"target", "shape"}
    help_text = (
        "Creates a new random tensor.\n"
        "\n"
        "Requires a non-existing destination tensor name and an explicit shape.\n"
        "Default distribution is uniform in [0, 1).\n"
        "\n"
        "Examples:\n"
        "  rand: { target: work::noise, shape: [1024], seed: 7 }\n"
        "  rand: { target: work::noise, shape: [256], distribution: normal, mean: 0, std: 0.1 }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> RandSpec:
        name = _require_literal_target_name(target_ref, op_name=self.name)
        shape = _parse_shape(payload.get("shape"), op_name=self.name)
        distribution, low, high, mean, std, seed = _parse_rand_options(payload)
        return RandSpec(
            target_ref=target_ref,
            target_name=name,
            shape=shape,
            distribution=distribution,
            low=low,
            high=high,
            mean=mean,
            std=std,
            seed=seed,
        )

    def apply_to_target(self, spec: RandSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        out = torch.empty(spec.shape, dtype=torch.float32)
        generator = None
        if spec.seed is not None:
            generator = torch.Generator(device=out.device)
            generator.manual_seed(spec.seed)
        if spec.distribution == "uniform":
            out.uniform_(spec.low, spec.high, generator=generator)
        else:
            out.normal_(spec.mean, spec.std, generator=generator)
        sd[name] = out

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del model_aliases
        if value_key == "distribution":
            return [
                candidate
                for candidate in ("uniform", "normal")
                if candidate.startswith(prefix_text)
            ]
        return None


def _parse_rand_options(
    payload: dict,
) -> tuple[RandDistribution, float, float, float, float, int | None]:
    raw_dist = payload.get("distribution", "uniform")
    if not isinstance(raw_dist, str) or raw_dist.strip().lower() not in {"uniform", "normal"}:
        raise TransformError("rand.distribution must be one of: uniform, normal")
    distribution: RandDistribution = raw_dist.strip().lower()  # type: ignore[assignment]

    seed: int | None = None
    if "seed" in payload:
        raw_seed = payload.get("seed")
        if not isinstance(raw_seed, int):
            raise TransformError("rand.seed must be an integer when provided")
        seed = raw_seed

    low = 0.0
    high = 1.0
    mean = 0.0
    std = 1.0
    if distribution == "uniform":
        if "low" in payload:
            low = require_numeric(payload, op_name="rand", key="low")
        if "high" in payload:
            high = require_numeric(payload, op_name="rand", key="high")
        if low >= high:
            raise TransformError("rand requires low < high for uniform distribution")
    else:
        if "mean" in payload:
            mean = require_numeric(payload, op_name="rand", key="mean")
        if "std" in payload:
            std = require_numeric(payload, op_name="rand", key="std")
        if std <= 0:
            raise TransformError("rand.std must be > 0 for normal distribution")
    return distribution, low, high, mean, std, seed


register_transform(ZeroesTransform())
register_transform(OnesTransform())
register_transform(RandTransform())
