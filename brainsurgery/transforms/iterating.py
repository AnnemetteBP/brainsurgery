from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Iterable, TypeVar

from ..model import tqdm
from ..transform import BaseTransform, StateDictProvider, TransformError, TransformResult


class DestinationPolicy(Enum):
    ANY = "any"
    MUST_EXIST = "must_exist"
    MUST_NOT_EXIST = "must_not_exist"


SpecT = TypeVar("SpecT")
ItemT = TypeVar("ItemT")


class IteratingTransform(BaseTransform, ABC, Generic[SpecT, ItemT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[SpecT]
    progress_desc: str | None = None
    progress_unit: str = "tensor"

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        items = self.resolve_items(typed, provider)
        self.validate_resolved_items(typed, items, provider)

        for item in self.iter_items(items):
            self.apply_item(typed, item, provider)

        return TransformResult(name=self.name, count=len(items))

    def require_spec(self, spec: object) -> SpecT:
        if not isinstance(spec, self.spec_type):
            raise self.error_type(
                f"{self.name} received wrong spec type: {type(spec).__name__}"
            )
        return spec

    def iter_items(self, items: list[ItemT]) -> Iterable[ItemT]:
        if self.progress_desc is None:
            return items
        return tqdm(items, desc=self.progress_desc, unit=self.progress_unit)

    @abstractmethod
    def resolve_items(
        self,
        spec: SpecT,
        provider: StateDictProvider,
    ) -> list[ItemT]:
        ...

    def validate_resolved_items(
        self,
        spec: SpecT,
        items: list[ItemT],
        provider: StateDictProvider,
    ) -> None:
        del spec, items, provider

    @abstractmethod
    def apply_item(
        self,
        spec: SpecT,
        item: ItemT,
        provider: StateDictProvider,
    ) -> None:
        ...
