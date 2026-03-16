from ..core import StateDictProvider, TransformError
from .plan import SurgeryPlan


def _infer_output_model(
    plan: SurgeryPlan,
    provider: StateDictProvider | None = None,
) -> str:
    destination_models = set()

    for compiled in plan.transforms:
        if not compiled.transform.contributes_output_model(compiled.spec):
            continue
        inferred_model = _infer_transform_output_model(compiled, provider)
        if provider is not None and not _has_any_tensor(provider, inferred_model):
            continue
        destination_models.add(inferred_model)

    if len(destination_models) != 1:
        raise TransformError(
            "cannot infer output model uniquely; expected exactly one destination model across all transforms"
        )

    return next(iter(destination_models))


def _infer_transform_output_model(
    compiled,
    provider: StateDictProvider | None,
) -> str:
    try:
        return compiled.transform._infer_output_model(compiled.spec)
    except TransformError:
        if provider is None:
            raise

        collect_models = getattr(compiled.spec, "collect_models", None)
        if not callable(collect_models):
            raise

        models = collect_models()
        if not isinstance(models, set):
            raise

        non_empty_models = [model for model in models if _has_any_tensor(provider, model)]
        if len(non_empty_models) == 1:
            return non_empty_models[0]
        raise


def _has_any_tensor(provider: StateDictProvider, model: str) -> bool:
    try:
        return len(provider.get_state_dict(model)) > 0
    except Exception:
        return False
