from __future__ import annotations

from .unary import UnarySpec, UnaryTransform, resolve_target_names
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    register_transform,
)


class DeleteTransformError(TransformError):
    pass


class DeleteTransform(UnaryTransform[UnarySpec]):
    name = "delete"
    error_type = DeleteTransformError
    spec_type = UnarySpec
    progress_desc = "Applying delete transforms"
    help_text = (
        "Deletes one or more tensors selected by 'target'.\n"
        "\n"
        "Targets may be specified by name or pattern. The entire tensor is removed; "
        "slicing is not supported.\n"
        "\n"
        "Examples:\n"
        "  delete: { target: ln_f_copy.weight }\n"
        "  delete: { target: '.*_backup' }"
    )

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is not None:
            raise DeleteTransformError("delete target must not be sliced")

    def resolve_targets(self, spec: UnarySpec, provider: StateDictProvider) -> list[str]:
        return resolve_target_names(
            target_ref=spec.target_ref,
            provider=provider,
            op_name=self.name,
            error_type=DeleteTransformError,
        )

    def apply_to_target(self, spec: UnarySpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        if name not in sd:
            raise DeleteTransformError(f"delete target disappeared during apply: {model}::{name}")

        del sd[name]


register_transform(DeleteTransform())
