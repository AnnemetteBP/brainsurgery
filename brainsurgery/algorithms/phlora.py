import torch

from ..core import TransformError


class PhloraSvdCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get(
        self,
        source: torch.Tensor,
        *,
        cache_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_key = f"{cache_key}|{tuple(source.shape)}|{source.dtype}|{source.device}"
        if full_key not in self._cache:
            self._cache[full_key] = torch.linalg.svd(source, full_matrices=False)
        return self._cache[full_key]


def require_positive_rank(
    value: float,
    *,
    error_type: type[TransformError],
    op_name: str,
    key: str,
) -> int:
    integer = int(value)
    if float(integer) != float(value) or integer <= 0:
        raise error_type(f"{op_name}.{key} must be a positive integer")
    return integer


def _resolve_effective_rank(
    source: torch.Tensor,
    requested_rank: int,
    *,
    error_type: type[TransformError],
    op_name: str,
    tensor_name: str,
) -> int:
    _require_matrix(
        source,
        error_type=error_type,
        op_name=op_name,
        tensor_name=tensor_name,
    )
    rank = min(requested_rank, min(source.shape))
    if rank <= 0:
        raise error_type(
            f"{op_name} rank became zero for {tensor_name} with shape {tuple(source.shape)}"
        )
    return rank


def _require_matrix(
    source: torch.Tensor,
    *,
    error_type: type[TransformError],
    op_name: str,
    tensor_name: str,
) -> None:
    if source.ndim != 2:
        raise error_type(
            f"{op_name} target must be 2D (got shape {tuple(source.shape)}): {tensor_name}"
        )


def compute_phlora_factors(
    source: torch.Tensor,
    requested_rank: int,
    *,
    cache: PhloraSvdCache,
    cache_key: str,
    error_type: type[TransformError],
    op_name: str,
    tensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = _resolve_effective_rank(
        source,
        requested_rank,
        error_type=error_type,
        op_name=op_name,
        tensor_name=tensor_name,
    )
    u, s, vh = cache.get(source, cache_key=cache_key)
    sqrt_s = s[:rank].sqrt()
    lora_a = sqrt_s[:, None] * vh[:rank, :]
    lora_b = u[:, :rank] * sqrt_s
    return lora_a, lora_b


def reconstruct_phlora_rank(
    source: torch.Tensor,
    requested_rank: int,
    *,
    cache: PhloraSvdCache,
    cache_key: str,
    error_type: type[TransformError],
    op_name: str,
    tensor_name: str,
) -> torch.Tensor:
    rank = _resolve_effective_rank(
        source,
        requested_rank,
        error_type=error_type,
        op_name=op_name,
        tensor_name=tensor_name,
    )
    u, s, vh = cache.get(source, cache_key=cache_key)
    return (u[:, :rank] * s[:rank]) @ vh[:rank, :]


__all__ = [
    "PhloraSvdCache",
    "_require_matrix",
    "_resolve_effective_rank",
    "compute_phlora_factors",
    "reconstruct_phlora_rank",
    "require_positive_rank",
]
