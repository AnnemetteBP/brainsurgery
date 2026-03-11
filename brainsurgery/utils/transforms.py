from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from ..core import ResolvedMapping
from ..core import TensorRef, parse_slice
from ..transforms.ternary import (
    ResolvedTernaryMapping,
    TernaryMappingSpec,
    TernaryMappingTransform,
)
from ..core import TransformError, ensure_mapping_payload, validate_payload_keys
from ..core import StateDictProvider
from ..transforms.binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..transforms.unary import UnarySpec, UnaryTransform

UnarySpecT = TypeVar("UnarySpecT", bound=UnarySpec)
BinarySpecT = TypeVar("BinarySpecT", bound=BinaryMappingSpec)
TernarySpecT = TypeVar("TernarySpecT", bound=TernaryMappingSpec)


@dataclass(frozen=True)
class Docs:
    summary: str
    notes: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnaryRefs:
    target_slice: bool = False


@dataclass(frozen=True)
class BinaryRefs:
    from_slice: bool = False
    to_slice: bool = False


@dataclass(frozen=True)
class TernaryRefs:
    from_a_slice: bool = False
    from_b_slice: bool = False
    to_slice: bool = False


def _lines(text: Docs, *rules: str) -> str:
    lines = [text.summary, "", *rules]
    if text.notes:
        lines.extend(["", *text.notes])
    if text.examples:
        lines.extend(["", "Examples:", *[f"  {example}" for example in text.examples]])
    return "\n".join(lines)


def _slice_rule(label: str, allowed: bool) -> str:
    return (
        f"{label} may include slicing." if allowed else f"{label} must not be sliced."
    )


def _destination_rule(policy: DestinationPolicy) -> str:
    if policy is DestinationPolicy.MUST_EXIST:
        return "Destination tensors must already exist."
    if policy is DestinationPolicy.MUST_NOT_EXIST:
        return "Destination tensors must not already exist."
    return "Destination tensors may be created or overwritten."


def _validate_slice(
    ref: TensorRef,
    *,
    allowed: bool,
    op_name: str,
    label: str,
    error_type: type[TransformError],
) -> None:
    if ref.slice_spec is None:
        return
    if not allowed:
        raise error_type(f"{op_name} {label} must not be sliced")
    parse_slice(ref.slice_spec)


class DeclarativeUnaryTransform(UnaryTransform[UnarySpecT]):
    spec_type = UnarySpec
    allowed_keys = {"target"}
    required_keys = {"target"}
    docs: Docs
    refs = UnaryRefs()
    spec_builder: Callable[[TensorRef, dict], UnarySpecT] | None = None
    progress_desc: str | None = None
    apply_fn: Callable[[UnarySpecT, str, StateDictProvider], None]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "docs", None) is None:
            return
        if cls.progress_desc is None:
            cls.progress_desc = f"Applying {cls.name} transforms"
        cls.slice_policy = "allow" if cls.refs.target_slice else "forbid"
        cls.help_text = _lines(
            cls.docs, _slice_rule("Target tensors", cls.refs.target_slice)
        )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> UnarySpecT:
        if self.spec_builder is not None:
            return self.spec_builder(target_ref, payload)
        return self.spec_type(target_ref=target_ref)

    def apply_to_target(
        self, spec: UnarySpecT, name: str, provider: StateDictProvider
    ) -> None:
        self.apply_fn(spec, name, provider)


class DeclarativeBinaryTransform(BinaryMappingTransform[BinarySpecT]):
    spec_type = BinaryMappingSpec
    allowed_keys = {"from", "to"}
    required_keys = {"from", "to"}
    docs: Docs
    refs = BinaryRefs()
    spec_builder: Callable[[TensorRef, TensorRef, dict], BinarySpecT] | None = None
    progress_desc: str | None = None
    apply_fn: Callable[[BinarySpecT, ResolvedMapping, StateDictProvider], None]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "docs", None) is None:
            return
        if cls.progress_desc is None:
            cls.progress_desc = f"Applying {cls.name} transforms"
        cls.help_text = _lines(
            cls.docs,
            _slice_rule("Source references", cls.refs.from_slice),
            _slice_rule("Destination references", cls.refs.to_slice),
            _destination_rule(cls.destination_policy),
        )

    def compile(self, payload: dict, default_model: str | None) -> BinarySpecT:
        if self.spec_builder is None:
            return super().compile(payload, default_model)
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )
        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)
        assert from_ref.model is not None
        assert to_ref.model is not None
        return self.spec_builder(from_ref, to_ref, payload)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        _validate_slice(
            from_ref,
            allowed=self.refs.from_slice,
            op_name=self.name,
            label="source",
            error_type=self.error_type,
        )
        _validate_slice(
            to_ref,
            allowed=self.refs.to_slice,
            op_name=self.name,
            label="destination",
            error_type=self.error_type,
        )

    def apply_item(
        self, spec: BinarySpecT, item: ResolvedMapping, provider: StateDictProvider
    ) -> None:
        self.apply_fn(spec, item, provider)

    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        raise NotImplementedError


class DeclarativeTernaryTransform(TernaryMappingTransform[TernarySpecT]):
    spec_type = TernaryMappingSpec
    allowed_keys = {"from_a", "from_b", "to"}
    required_keys = {"from_a", "from_b", "to"}
    docs: Docs
    refs = TernaryRefs()
    progress_desc: str | None = None
    apply_fn: Callable[[TernarySpecT, ResolvedTernaryMapping, StateDictProvider], None]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "docs", None) is None:
            return
        if cls.progress_desc is None:
            cls.progress_desc = f"Applying {cls.name} transforms"
        cls.help_text = _lines(
            cls.docs,
            "References may be regex or structured mappings.",
            _slice_rule("'from_a' references", cls.refs.from_a_slice),
            _slice_rule("'from_b' references", cls.refs.from_b_slice),
            _slice_rule("'to' references", cls.refs.to_slice),
            _destination_rule(cls.destination_policy),
        )

    def validate_refs(
        self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef
    ) -> None:
        _validate_slice(
            from_a_ref,
            allowed=self.refs.from_a_slice,
            op_name=self.name,
            label="from_a",
            error_type=self.error_type,
        )
        _validate_slice(
            from_b_ref,
            allowed=self.refs.from_b_slice,
            op_name=self.name,
            label="from_b",
            error_type=self.error_type,
        )
        _validate_slice(
            to_ref,
            allowed=self.refs.to_slice,
            op_name=self.name,
            label="destination",
            error_type=self.error_type,
        )

    def apply_item(
        self,
        spec: TernarySpecT,
        item: ResolvedTernaryMapping,
        provider: StateDictProvider,
    ) -> None:
        self.apply_fn(spec, item, provider)

    def apply_mapping(
        self, item: ResolvedTernaryMapping, provider: StateDictProvider
    ) -> None:
        raise NotImplementedError


__all__ = [
    "Docs",
    "UnaryRefs",
    "BinaryRefs",
    "TernaryRefs",
    "DeclarativeUnaryTransform",
    "DeclarativeBinaryTransform",
    "DeclarativeTernaryTransform",
]
