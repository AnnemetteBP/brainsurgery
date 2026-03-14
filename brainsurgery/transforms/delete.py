from ..core import UnarySpec, UnaryTransform
from ..core import TensorRef, must_model
from ..core import register_transform
from ..core import StateDictProvider, TransformError
from ..engine import emit_verbose_unary_activity


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

    def apply_to_target(self, spec: UnarySpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        if name not in sd:
            raise DeleteTransformError(f"delete target disappeared during apply: {model}::{name}")

        del sd[name]
        emit_verbose_unary_activity(self.name, name)










register_transform(DeleteTransform())
