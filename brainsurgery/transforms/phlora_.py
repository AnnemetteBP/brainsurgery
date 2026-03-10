from __future__ import annotations

from dataclasses import dataclass

from ..phlora import PhloraSvdCache, reconstruct_phlora_rank, require_positive_rank
from .unary import UnarySpec, UnaryTransform
from ..refs import TensorRef, must_model
from ..transform import register_transform, require_numeric
from ..transform_types import StateDictProvider, TransformError


class PhloraInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class PhloraInPlaceSpec(UnarySpec):
    rank: int


class PhloraInPlaceTransform(UnaryTransform[PhloraInPlaceSpec]):
    name = "phlora_"
    error_type = PhloraInPlaceTransformError
    spec_type = PhloraInPlaceSpec
    allowed_keys = {"target", "rank"}
    required_keys = {"target", "rank"}
    progress_desc = "Applying phlora_ transforms"
    help_text = (
        "Applies in-place PHLoRA low-rank reconstruction.\n"
        "\n"
        "For each matched 2D tensor W:\n"
        "  u, s, vh = svd(W)\n"
        "  W <- (u[:, :r] * s[:r]) @ vh[:r, :]\n"
        "\n"
        "Examples:\n"
        "  phlora_: { target: '.*weight', rank: 64 }\n"
        "  phlora_: { target: h.0.attn.c_proj.weight, rank: 16 }"
    )

    def __init__(self) -> None:
        super().__init__()
        self._svd_cache = PhloraSvdCache()

    def build_spec(self, target_ref: TensorRef, payload: dict) -> PhloraInPlaceSpec:
        rank = require_positive_rank(
            require_numeric(payload, op_name=self.name, key="rank"),
            error_type=PhloraInPlaceTransformError,
            op_name=self.name,
            key="rank",
        )
        return PhloraInPlaceSpec(target_ref=target_ref, rank=rank)

    def apply_to_target(
        self,
        spec: PhloraInPlaceSpec,
        name: str,
        provider: StateDictProvider,
    ) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        source = sd[name]
        new_tensor = reconstruct_phlora_rank(
            source,
            spec.rank,
            cache=self._svd_cache,
            cache_key=f"{model}::{name}",
            error_type=PhloraInPlaceTransformError,
            op_name="phlora_",
            tensor_name=f"{model}::{name}",
        )
        sd[name] = new_tensor.to(dtype=source.dtype, device=source.device)




register_transform(PhloraInPlaceTransform())
