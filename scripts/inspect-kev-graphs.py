#!/usr/bin/env python3
"""Inspect and compare ONNX graph structures for Kev-0.6B: q4 vs q4f16.

Verifies operator counts, pointer head topologies, scaling factors,
and lack of baked-in temperature division.
"""
import json
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

    # Inspect pointer head
    head_nodes = [n for n in graph.node if "head" in n.name.lower()]

    # Trace final logits output to inspect scale / temperature
    output_names = {o.name for o in graph.output}
    logits_producers = [n for n in graph.node if any(o in output_names for o in n.output)]

    scale_value = None
    has_temperature_division = False

    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    val = onnx.numpy_helper.to_array(attr.t)
                    if val.shape == () and abs(float(val) - 0.0625) < 1e-6:
                        scale_value = float(val)

    # Check if logits producer is a division (which would indicate temperature scaling)
    if logits_producers and logits_producers[0].op_type == "Div":
        has_temperature_division = True

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
        "pointer_head_scale_constant": scale_value,
        "has_temperature_division_in_graph": has_temperature_division,
        "logits_producer_op": logits_producers[0].op_type if logits_producers else None,
    }


def main():
    q4_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kev-files/onnx/model_q4.onnx"
    q4f16_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/kev-inspect/model_q4f16.onnx"

    out = {}
    if Path(q4_path).exists():
        out["q4"] = inspect_graph(q4_path)
    if Path(q4f16_path).exists():
        out["q4f16"] = inspect_graph(q4f16_path)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
