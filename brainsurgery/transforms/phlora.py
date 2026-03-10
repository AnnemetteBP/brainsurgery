from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ternary import (
    ResolvedTernaryMapping,
    TernaryMappingSpec,
    TernaryMappingTransform,
)
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    parse_model_expr,
    register_transform,
    require_expr,
    require_numeric,
    resolve_name_mappings,
    validate_payload_keys,
)


class PhloraTransformError(TransformError):
    pass


@dataclass(frozen=True)
class PhloraSpec(TernaryMappingSpec):
    rank: int
    delete_original: bool
    require_missing_dest: bool

    @property
    def target_ref(self) -> TensorRef:
        return self.from_a_ref

    @property
    def target_b_ref(self) -> TensorRef:
        return self.from_b_ref

    @property
    def target_a_ref(self) -> TensorRef:
        return self.to_ref


class PhloraTransform(TernaryMappingTransform[PhloraSpec]):
    name = "phlora"
    error_type = PhloraTransformError
    spec_type = PhloraSpec
    allowed_keys = {
        "target",
        "target_a",
        "target_b",
        "rank",
        "delete_original",
        "require_missing_dest",
    }
    required_keys = {"target", "target_a", "target_b", "rank"}
    progress_desc = "Applying phlora transforms"
    help_text = (
        "Splits each target tensor into PHLoRA low-rank factors B and A.\n"
        "\n"
        "Inputs:\n"
        "  - target: source tensor expression\n"
        "  - target_a: destination expression for A = sqrt(S) * Vh\n"
        "  - target_b: destination expression for B = U * sqrt(S)\n"
        "  - rank: low-rank dimension r\n"
        "\n"
        "target/target_a/target_b support regex or structured expressions like other transforms.\n"
        "Mappings are resolved from target -> target_a and target -> target_b; both must cover "
        "the same source set.\n"
        "\n"
        "Options:\n"
        "  - delete_original (default: true)\n"
        "  - require_missing_dest (default: true)\n"
        "\n"
        "Example:\n"
        "  phlora: { target: '(.*)\\\\.weight', rank: 16, target_a: '\\\\1.a', target_b: '\\\\1.b' }"
    )

    def __init__(self) -> None:
        super().__init__()
        self._svd_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def completion_reference_keys(self) -> list[str]:
        return ["target", "target_a", "target_b"]

    def compile(self, payload: dict, default_model: str | None) -> PhloraSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )

        target_ref = parse_model_expr(
            require_expr(payload, op_name=self.name, key="target"),
            default_model=default_model,
        )
        target_model_default = target_ref.model if target_ref.model is not None else default_model
        target_a_ref = parse_model_expr(
            require_expr(payload, op_name=self.name, key="target_a"),
            default_model=target_model_default,
        )
        target_b_ref = parse_model_expr(
            require_expr(payload, op_name=self.name, key="target_b"),
            default_model=target_model_default,
        )

        self.validate_refs(target_ref, target_b_ref, target_a_ref)

        rank = _require_positive_int(payload, op_name=self.name, key="rank")
        delete_original = _require_boolean(
            payload,
            op_name=self.name,
            key="delete_original",
            default=True,
        )
        require_missing_dest = _require_boolean(
            payload,
            op_name=self.name,
            key="require_missing_dest",
            default=True,
        )
        return PhloraSpec(
            from_a_ref=target_ref,
            from_b_ref=target_b_ref,
            to_ref=target_a_ref,
            rank=rank,
            delete_original=delete_original,
            require_missing_dest=require_missing_dest,
        )

    def validate_refs(self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_a_ref.slice_spec is not None:
            raise PhloraTransformError("phlora target must not be sliced")
        if from_b_ref.slice_spec is not None:
            raise PhloraTransformError("phlora target_b must not be sliced")
        if to_ref.slice_spec is not None:
            raise PhloraTransformError("phlora target_a must not be sliced")

    def resolve_mappings(
        self,
        spec: PhloraSpec,
        provider: StateDictProvider,
    ) -> list[ResolvedTernaryMapping]:
        a_mappings = resolve_name_mappings(
            from_ref=spec.target_ref,
            to_ref=spec.target_a_ref,
            provider=provider,
            op_name=self.name,
        )
        b_mappings = resolve_name_mappings(
            from_ref=spec.target_ref,
            to_ref=spec.target_b_ref,
            provider=provider,
            op_name=self.name,
        )
        pairs = _pair_mappings(a_mappings, b_mappings, op_name=self.name)

        resolved: list[ResolvedTernaryMapping] = []
        for src, map_a, map_b in pairs:
            resolved.append(
                ResolvedTernaryMapping(
                    src_a_model=src.src_model,
                    src_a_name=src.src_name,
                    src_a_slice=None,
                    # src_b_* is repurposed here as destination B.
                    src_b_model=map_b.dst_model,
                    src_b_name=map_b.dst_name,
                    src_b_slice=None,
                    # dst_* is destination A.
                    dst_model=map_a.dst_model,
                    dst_name=map_a.dst_name,
                    dst_slice=None,
                )
            )
        return resolved

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        mappings = self.resolve_mappings(typed, provider)

        for item in self.iter_mappings(mappings):
            src_sd = provider.get_state_dict(item.src_a_model)
            a_sd = provider.get_state_dict(item.dst_model)
            b_sd = provider.get_state_dict(item.src_b_model)

            if item.src_a_name not in src_sd:
                raise PhloraTransformError(
                    f"phlora source disappeared during apply: {item.src_a_model}::{item.src_a_name}"
                )

            source = src_sd[item.src_a_name]
            if source.ndim != 2:
                raise PhloraTransformError(
                    f"phlora target must be 2D (got shape {tuple(source.shape)}): "
                    f"{item.src_a_model}::{item.src_a_name}"
                )

            rank = min(typed.rank, min(source.shape))
            if rank <= 0:
                raise PhloraTransformError(
                    f"phlora rank became zero for {item.src_a_model}::{item.src_a_name} "
                    f"with shape {tuple(source.shape)}"
                )

            if typed.require_missing_dest and item.dst_name in a_sd:
                raise PhloraTransformError(
                    f"phlora destination already exists: {item.dst_model}::{item.dst_name}"
                )
            if typed.require_missing_dest and item.src_b_name in b_sd:
                raise PhloraTransformError(
                    f"phlora destination already exists: {item.src_b_model}::{item.src_b_name}"
                )
            if item.dst_model == item.src_b_model and item.dst_name == item.src_b_name:
                raise PhloraTransformError(
                    f"phlora destination collision for source {item.src_a_model}::{item.src_a_name}: "
                    f"{item.dst_model}::{item.dst_name}"
                )

            u, s, vh = self._get_svd(source, cache_key=f"{item.src_a_model}::{item.src_a_name}")
            lora_u = u[:, :rank]
            lora_s = s[:rank]
            lora_vh = vh[:rank, :]
            sqrt_s = lora_s.sqrt()
            lora_a = sqrt_s[:, None] * lora_vh
            lora_b = lora_u * sqrt_s

            a_sd[item.dst_name] = lora_a.to(dtype=source.dtype, device=source.device)
            b_sd[item.src_b_name] = lora_b.to(dtype=source.dtype, device=source.device)

            if typed.delete_original:
                del src_sd[item.src_a_name]

        return TransformResult(name=self.name, count=len(mappings))

    def apply_mapping(self, item: ResolvedTernaryMapping, provider: StateDictProvider) -> None:
        del item, provider
        raise PhloraTransformError(
            "phlora internal error: apply_mapping should not be called (custom apply is used)"
        )

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        model = typed.target_ref.model
        if model is None:
            raise PhloraTransformError("phlora output model missing")
        return model

    def _get_svd(
        self,
        source: torch.Tensor,
        *,
        cache_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_key = f"{cache_key}|{tuple(source.shape)}|{source.dtype}|{source.device}"
        if full_key not in self._svd_cache:
            self._svd_cache[full_key] = torch.linalg.svd(source, full_matrices=False)
        return self._svd_cache[full_key]


def _pair_mappings(
    a_mappings: list[ResolvedMapping],
    b_mappings: list[ResolvedMapping],
    *,
    op_name: str,
) -> list[tuple[ResolvedMapping, ResolvedMapping, ResolvedMapping]]:
    a_by_src = {(m.src_model, m.src_name): m for m in a_mappings}
    b_by_src = {(m.src_model, m.src_name): m for m in b_mappings}
    if set(a_by_src.keys()) != set(b_by_src.keys()):
        raise PhloraTransformError(f"{op_name} target_a/target_b mappings do not match source set")
    pairs: list[tuple[ResolvedMapping, ResolvedMapping, ResolvedMapping]] = []
    for key in sorted(a_by_src.keys()):
        src = a_by_src[key]
        pairs.append((src, a_by_src[key], b_by_src[key]))
    return pairs


def _require_positive_int(payload: dict, *, op_name: str, key: str) -> int:
    numeric = require_numeric(payload, op_name=op_name, key=key)
    integer = int(numeric)
    if float(integer) != float(numeric) or integer <= 0:
        raise PhloraTransformError(f"{op_name}.{key} must be a positive integer")
    return integer


def _require_boolean(payload: dict, *, op_name: str, key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise PhloraTransformError(f"{op_name}.{key} must be a boolean when provided")
    return value


def _unit_test_phlora_compile_rejects_non_integral_rank() -> None:
    try:
        PhloraTransform().compile(
            {"target": "w", "target_a": "w.a", "target_b": "w.b", "rank": 3.5},
            default_model="model",
        )
    except PhloraTransformError as exc:
        assert "positive integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected rank validation error")


def _unit_test_phlora_split_mode_writes_a_b_and_deletes_original_by_default() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "proj.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraTransform()
    spec = transform.compile(
        {
            "target": "(.*)\\.weight",
            "rank": 1,
            "target_a": "\\1.a",
            "target_b": "\\1.b",
        },
        default_model="model",
    )
    transform.apply(spec, provider)

    sd = provider._state_dict
    assert "proj.weight" not in sd
    assert "proj.a" in sd
    assert "proj.b" in sd


def _unit_test_phlora_split_mode_can_keep_original_when_configured() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "proj.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraTransform()
    spec = transform.compile(
        {
            "target": "(.*)\\.weight",
            "rank": 1,
            "target_a": "\\1.a",
            "target_b": "\\1.b",
            "delete_original": False,
        },
        default_model="model",
    )
    transform.apply(spec, provider)

    sd = provider._state_dict
    assert "proj.weight" in sd
    assert "proj.a" in sd
    assert "proj.b" in sd


__unit_tests__ = [
    _unit_test_phlora_compile_rejects_non_integral_rank,
    _unit_test_phlora_split_mode_writes_a_b_and_deletes_original_by_default,
    _unit_test_phlora_split_mode_can_keep_original_when_configured,
]


register_transform(PhloraTransform())
