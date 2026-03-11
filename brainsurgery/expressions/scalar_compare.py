from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core import AssertTransformError


COMPARISON_KEYS = ("is", "ge", "gt", "le", "lt")


@dataclass(frozen=True)
class ScalarComparison:
    exact: int | None
    ge: int | None
    gt: int | None
    le: int | None
    lt: int | None

    def describe(self) -> str:
        parts: list[str] = []
        if self.exact is not None:
            parts.append(f"exactly {self.exact}")
        if self.ge is not None:
            parts.append(f">= {self.ge}")
        if self.gt is not None:
            parts.append(f"> {self.gt}")
        if self.le is not None:
            parts.append(f"<= {self.le}")
        if self.lt is not None:
            parts.append(f"< {self.lt}")
        return " and ".join(parts)

    def matches(self, value: int) -> bool:
        if self.exact is not None and value != self.exact:
            return False
        if self.ge is not None and value < self.ge:
            return False
        if self.gt is not None and value <= self.gt:
            return False
        if self.le is not None and value > self.le:
            return False
        if self.lt is not None and value >= self.lt:
            return False
        return True


def parse_scalar_comparison(
    payload: dict[str, Any],
    *,
    op_name: str,
    aliases: dict[str, str] | None = None,
) -> ScalarComparison:
    values: dict[str, int | None] = {key: None for key in COMPARISON_KEYS}

    def parse_value(field_name: str, public_name: str) -> int | None:
        value = payload.get(public_name)
        if value is None:
            return None
        if not isinstance(value, int) or value < 0:
            raise AssertTransformError(f"{op_name}.{public_name} must be a non-negative integer")
        return value

    for key in COMPARISON_KEYS:
        values[key] = parse_value(key, key)

    if aliases is not None:
        for alias_name, canonical_name in aliases.items():
            alias_value = parse_value(canonical_name, alias_name)
            if alias_value is None:
                continue
            existing = values[canonical_name]
            if existing is not None and existing != alias_value:
                raise AssertTransformError(
                    f"{op_name}.{alias_name} conflicts with {op_name}.{canonical_name}"
                )
            values[canonical_name] = alias_value

    comparison = ScalarComparison(
        exact=values["is"],
        ge=values["ge"],
        gt=values["gt"],
        le=values["le"],
        lt=values["lt"],
    )

    if all(getattr(comparison, attr) is None for attr in ("exact", "ge", "gt", "le", "lt")):
        keys = list(COMPARISON_KEYS)
        if aliases:
            keys.extend(sorted(aliases))
        raise AssertTransformError(f"{op_name} must include at least one of: {', '.join(keys)}")

    _validate_scalar_comparison(comparison, op_name=op_name)
    return comparison


def _validate_scalar_comparison(comparison: ScalarComparison, *, op_name: str) -> None:
    lower_bound = comparison.ge
    lower_inclusive = True
    if comparison.gt is not None and (
        lower_bound is None
        or comparison.gt > lower_bound
        or (comparison.gt == lower_bound and lower_inclusive)
    ):
        lower_bound = comparison.gt
        lower_inclusive = False

    upper_bound = comparison.le
    upper_inclusive = True
    if comparison.lt is not None and (
        upper_bound is None
        or comparison.lt < upper_bound
        or (comparison.lt == upper_bound and upper_inclusive)
    ):
        upper_bound = comparison.lt
        upper_inclusive = False

    if lower_bound is not None and upper_bound is not None:
        if lower_bound > upper_bound:
            raise AssertTransformError(f"{op_name} has contradictory bounds")
        if lower_bound == upper_bound and (not lower_inclusive or not upper_inclusive):
            raise AssertTransformError(f"{op_name} has contradictory bounds")

    if comparison.exact is None:
        return

    if comparison.ge is not None and comparison.exact < comparison.ge:
        raise AssertTransformError(f"{op_name}.is cannot be smaller than {op_name}.ge")
    if comparison.gt is not None and comparison.exact <= comparison.gt:
        raise AssertTransformError(f"{op_name}.is must be greater than {op_name}.gt")
    if comparison.le is not None and comparison.exact > comparison.le:
        raise AssertTransformError(f"{op_name}.is cannot be larger than {op_name}.le")
    if comparison.lt is not None and comparison.exact >= comparison.lt:
        raise AssertTransformError(f"{op_name}.is must be less than {op_name}.lt")
