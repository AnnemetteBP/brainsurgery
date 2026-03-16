from typing import Any

from .specs import StateDictProvider, TensorRef, must_model, parse_slice, select_tensor


def state_dict_for_ref(provider: StateDictProvider, ref: TensorRef) -> Any:
    return provider.get_state_dict(must_model(ref))


def parsed_slice(ref: TensorRef) -> Any:
    if ref.slice_spec is None:
        return None
    return parse_slice(ref.slice_spec)


def view_for_ref_name(
    provider: StateDictProvider,
    ref: TensorRef,
    tensor_name: str,
) -> tuple[Any, Any]:
    state_dict = state_dict_for_ref(provider, ref)
    tensor_view = select_tensor(state_dict[tensor_name], parsed_slice(ref))
    return state_dict, tensor_view


def unary_view_for_ref_name(
    provider: StateDictProvider,
    ref: TensorRef,
    tensor_name: str,
) -> tuple[Any, Any]:
    return view_for_ref_name(provider, ref, tensor_name)


def binary_mapping_views(
    provider: StateDictProvider,
    *,
    from_ref: TensorRef,
    to_ref: TensorRef,
    src_name: str,
    dst_name: str,
) -> tuple[Any, Any, Any, Any]:
    src_sd, src_view = view_for_ref_name(provider, from_ref, src_name)
    dst_sd, dst_view = view_for_ref_name(provider, to_ref, dst_name)
    return src_sd, dst_sd, src_view, dst_view


def ternary_mapping_views(
    provider: StateDictProvider,
    *,
    from_a_ref: TensorRef,
    from_b_ref: TensorRef,
    to_ref: TensorRef,
    src_a_name: str,
    src_b_name: str,
    dst_name: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    src_a_sd, src_a_view = view_for_ref_name(provider, from_a_ref, src_a_name)
    src_b_sd, src_b_view = view_for_ref_name(provider, from_b_ref, src_b_name)
    dst_sd, dst_view = view_for_ref_name(provider, to_ref, dst_name)
    return src_a_sd, src_b_sd, dst_sd, src_a_view, src_b_view, dst_view
