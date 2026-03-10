from __future__ import annotations

from dataclasses import dataclass

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..transform import StateDictProvider, TransformError, ensure_mapping_payload, register_transform, validate_payload_keys


class ReshapeTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ReshapeSpec(BinaryMappingSpec):
    shape: tuple[int, ...]


class ReshapeTransform(BinaryMappingTransform[ReshapeSpec]):
    name = "reshape"
    error_type = ReshapeTransformError
    spec_type = ReshapeSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "shape"}
    required_keys = {"from", "to", "shape"}
    progress_desc = "Applying reshape transforms"
    help_text = (
        "Reshapes source tensors into new destination tensors.\n"
        "\n"
        "Source references may be sliced. Destination tensors must not exist and "
        "must not be sliced. Shape may include one '-1'.\n"
        "\n"
        "Example:\n"
        "  reshape: { from: x, to: x2d, shape: [1024, -1] }"
    )

    def compile(self, payload: dict, default_model: str | None) -> ReshapeSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )
        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)
        shape = _parse_shape(payload.get("shape"), op_name=self.name, error_type=ReshapeTransformError)
        return ReshapeSpec(from_ref=from_ref, to_ref=to_ref, shape=shape)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise ReshapeTransformError("reshape destination must not be sliced")

    def apply_item(self, spec: ReshapeSpec, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)
        src_view = select_tensor(src_sd[item.src_name], item.src_slice)
        try:
            dst_sd[item.dst_name] = src_view.reshape(spec.shape).clone()
        except RuntimeError as exc:
            raise ReshapeTransformError(
                f"reshape failed for {item.src_name} -> {item.dst_name}: {exc}"
            ) from exc


def _parse_shape(raw: object, *, op_name: str, error_type: type[TransformError]) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise error_type(f"{op_name}.shape must be a non-empty list of integers")
    if not all(isinstance(x, int) for x in raw):
        raise error_type(f"{op_name}.shape must be a non-empty list of integers")
    if sum(1 for x in raw if x == -1) > 1:
        raise error_type(f"{op_name}.shape may include at most one '-1'")
    if any(x == 0 or x < -1 for x in raw):
        raise error_type(f"{op_name}.shape dimensions must be positive integers or -1")
    return tuple(raw)








register_transform(ReshapeTransform())
