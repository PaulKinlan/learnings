#!/usr/bin/env python3
"""Inspect and compare ONNX graph structures for Kev-0.6B: q4 vs q4f16.

Verifies operator counts, pointer head topologies, scaling factors,
and absence of baked-in temperature division by tracing the full ancestor
subgraph from logits back to backbone projections.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    import onnx
except ImportError:
    print("onnx package not installed; run with 'uv run --with onnx python3 scripts/inspect-kev-graphs.py'", file=sys.stderr)
    sys.exit(1)


def inspect_graph(onnx_path: str):
    p = Path(onnx_path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = onnx.load(str(p), load_external_data=False)
    graph = model.graph

    op_counts = Counter(n.op_type for n in graph.node)

    # Build producer index for graph traversal
    producer = {out: n for n in graph.node for out in n.output}

    # Trace complete ancestor subgraph from 'logits' backward to backbone projections
    visited_head_nodes = {}
    frontier = ["logits"]
    while frontier:
        out = frontier.pop(0)
        n = producer.get(out)
        if not n or n.name in visited_head_nodes:
            continue
        visited_head_nodes[n.name] = n
        # Boundary: stop traversing once we cross the backbone projection heads
        if "/head_k/MatMul" in n.name or "/head_q/MatMul" in n.name:
            continue
        for inp in n.input:
            frontier.append(inp)

    # Audit for any division operation in the ancestor lineage of logits
    div_nodes_in_head = [n for n in visited_head_nodes.values() if n.op_type == "Div"]
    has_temperature_division = len(div_nodes_in_head) > 0

    # Inspect pointer head projection nodes
    head_nodes = [n for n in graph.node if "head" in n.name.lower()]

    # Inspect constant scaling factor
    scale_value = None
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    val = onnx.numpy_helper.to_array(attr.t)
                    if val.shape == () and abs(float(val) - 0.0625) < 1e-6:
                        scale_value = float(val)

    # Immediate producer of logits
    logits_node = producer.get("logits")

    return {
        "file": p.name,
        "size_bytes": p.stat().st_size,
        "total_nodes": len(graph.node),
        "inputs": [i.name for i in graph.input],
        "outputs": [o.name for o in graph.output],
        "matmul_nbits_count": op_counts.get("MatMulNBits", 0),
        "gather_block_quantized_count": op_counts.get("GatherBlockQuantized", 0),
        "cast_count": op_counts.get("Cast", 0),
        "pointer_head_node_count": len(head_nodes),
        "pointer_head_ancestor_node_count": len(visited_head_nodes),
        "pointer_head_scale_constant": scale_value,
        "has_temperature_division_in_graph": has_temperature_division,
        "div_nodes_in_pointer_head_lineage": [n.name for n in div_nodes_in_head],
        "logits_producer_op": logits_node.op_type if logits_node else None,
    }


def main():
    default_dir = Path(os.environ.get("KEV_FILES_DIR", "/tmp/kev-files"))
    q4_path = sys.argv[1] if len(sys.argv) > 1 else str(default_dir / "onnx/model_q4.onnx")
    q4f16_path = sys.argv[2] if len(sys.argv) > 2 else str(default_dir / "onnx/model_q4f16.onnx")

    out = {}
    if Path(q4_path).exists():
        out["q4"] = inspect_graph(q4_path)
    if Path(q4f16_path).exists():
        out["q4f16"] = inspect_graph(q4f16_path)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
