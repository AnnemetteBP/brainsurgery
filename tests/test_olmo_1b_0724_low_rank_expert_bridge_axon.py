from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules, main_module="flexmore_low_rank_bridge")


def _node_specs(graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in graph:
        assert isinstance(item, dict) and len(item) == 1
        _, node_spec = next(iter(item.items()))
        assert isinstance(node_spec, dict)
        out.append(node_spec)
    return out


def test_low_rank_expert_bridge_axon_lowers_to_moe_synapse_spec(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "flexmore_examples" / "olmo_1b_0724_low_rank_expert_bridge.axon")

    model = spec["model"]
    assert model["symbols"]["E"] == 2
    assert model["symbols"]["EPT"] == 2
    assert "flexmore_decoder_block" in model["blocks"]
    assert "expert_ffn" in model["blocks"]

    top_graph_ops = [node["_op"] for node in _node_specs(model["graph"])]
    assert "embedding" in top_graph_ops
    assert "call" in top_graph_ops

    block_nodes = _node_specs(model["blocks"]["flexmore_decoder_block"]["graph"])
    block_ops = [node["_op"] for node in block_nodes]
    assert "linear" in block_ops
    assert "attention" in block_ops
    loop_node = next(node for node in block_nodes if node["_op"] == "for")
    loop_body = _node_specs(loop_node["_body"])
    loop_ops = [node["_op"] for node in loop_body]
    assert "moe_select" in loop_ops
    assert "moe_scatter_add" in loop_ops

    expert_nodes = _node_specs(model["blocks"]["expert_ffn"]["graph"])
    expert_ops = [node["_op"] for node in expert_nodes]
    assert expert_ops.count("linear") >= 2
    assert "mul" in expert_ops
