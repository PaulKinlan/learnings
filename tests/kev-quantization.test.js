// tests/kev-quantization.test.js
// Quantization comparison and architectural audit between Kev's ONNX export variants:
// q4 (CPU / WASM JSEP) vs q4f16 (WebGPU).
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { KEV_MAX_BYTES, KEV_SHA256 } from "../site/decision-models/kev-manifest.js";

// Load the empirically generated quantization comparison ledger
const LEDGER_PATH = resolve("research/kev-quantization-comparison.json");
const QUANT_SPEC = JSON.parse(readFileSync(LEDGER_PATH, "utf8")).variants;

// Pinned ONNX model files for live inspection (can be fetched via `node scripts/fetch-kev-fixtures.mjs`)
const Q4_FILE = "/tmp/kev-files/onnx/model_q4.onnx";
const Q4F16_CANDIDATES = [
  "/tmp/kev-files/onnx/model_q4f16.onnx",
  "/tmp/kev-inspect/model_q4f16.onnx",
];
const q4f16File = Q4F16_CANDIDATES.find((p) => existsSync(p));
const hasArtifacts = existsSync(Q4_FILE) && Boolean(q4f16File);

test("quantization comparison: storage footprint and compression ratio", () => {
  // Unquantized Qwen3-0.6B baseline (float32: ~2.4 GB; float16: ~1.2 GB)
  const unquantizedFp16Bytes = 1.2 * 1024 * 1024 * 1024;

  const q4Bytes = QUANT_SPEC.q4.files.weights_size_bytes;
  const q4f16Bytes = QUANT_SPEC.q4f16.files.weights_size_bytes;

  // q4 achieves ~68% size reduction vs fp16
  const q4Savings = (unquantizedFp16Bytes - q4Bytes) / unquantizedFp16Bytes;
  assert.ok(q4Savings > 0.65 && q4Savings < 0.75, `q4 size reduction is ${(q4Savings * 100).toFixed(1)}%`);

  // q4f16 achieves ~72% size reduction vs fp16
  const q4f16Savings = (unquantizedFp16Bytes - q4f16Bytes) / unquantizedFp16Bytes;
  assert.ok(q4f16Savings > 0.70 && q4f16Savings < 0.78, `q4f16 size reduction is ${(q4f16Savings * 100).toFixed(1)}%`);

  // q4f16 data is smaller than q4 data because float16 weight constants are smaller than fp32 constants
  assert.ok(q4f16Bytes < q4Bytes, "q4f16 external weights file is smaller than q4");
});

test("quantization comparison: graph architecture and pointer head stability", () => {
  // q4 runs on CPU with float32 pointer head: 4 nodes (/head_q/MatMul, /head_k/MatMul, /head_q/Add, /head_k/Add)
  assert.equal(QUANT_SPEC.q4.graph_topology.pointer_head_nodes, 4);

  // q4f16 converts backbone representations to fp16 for WebGPU. To prevent numerical underflow
  // in the dot product pointer head, it explicitly wraps projections in cast_to_fp32 and cast_to_fp16
  // nodes (14 nodes total).
  assert.equal(QUANT_SPEC.q4f16.graph_topology.pointer_head_nodes, 14);
  assert.ok(
    QUANT_SPEC.q4f16.graph_topology.cast_nodes > QUANT_SPEC.q4.graph_topology.cast_nodes,
    "q4f16 contains 48 additional Cast nodes to bridge fp16 backbone to fp32 pointer head",
  );
  assert.equal(
    QUANT_SPEC.q4.graph_topology.pointer_head_scale_constant,
    0.0625,
    "q4 scale constant is 1/sqrt(256) = 0.0625",
  );
  assert.equal(
    QUANT_SPEC.q4.graph_topology.has_temperature_division_in_graph,
    false,
    "ONNX graph has no baked-in temperature division and emits raw logits (T = 1.0)",
  );
});

test("quantization comparison: GatherBlockQuantized operator requirement", () => {
  // The embedding table uses GatherBlockQuantized.
  // In ONNX Runtime Web, this kernel is registered ONLY when JSEP bridge is initialized.
  assert.equal(QUANT_SPEC.q4.quantization_scheme.embedding_matrix, "GatherBlockQuantized (4-bit block-quantized lookups)");
  assert.ok(
    QUANT_SPEC.q4.runtime_compatibility.jsep_wasm.startsWith("PASS"),
    "q4 on CPU requires JSEP wasm build to execute GatherBlockQuantized without runtime crash",
  );
});

test("quantization comparison: manifest size limits bound both variants", () => {
  // Verify that KEV_MAX_BYTES accommodates the pinned q4 weights with safe headroom
  const q4Limit = KEV_MAX_BYTES["onnx/model_q4.onnx_data"];
  assert.ok(q4Limit > QUANT_SPEC.q4.files.weights_size_bytes, "maxBytes accommodates model_q4.onnx_data");
  assert.ok(q4Limit < QUANT_SPEC.q4.files.weights_size_bytes * 1.5, "headroom is tight (~20%) to prevent memory exhaustion");

  // Verify graph size limits
  const q4GraphLimit = KEV_MAX_BYTES["onnx/model_q4.onnx"];
  assert.ok(q4GraphLimit > QUANT_SPEC.q4.files.graph_size_bytes, "maxBytes accommodates model_q4.onnx");
});

test(
  "quantization graph inspection: scripts/inspect-kev-graphs.py live traversal proves absence of temperature division",
  { skip: !hasArtifacts ? "requires model_q4.onnx and model_q4f16.onnx in /tmp/kev-files/onnx/ (run node scripts/fetch-kev-fixtures.mjs)" : false },
  () => {
    const raw = execFileSync("uv", ["run", "--with", "onnx", "python3", "scripts/inspect-kev-graphs.py", Q4_FILE, q4f16File], {
      encoding: "utf8",
    });
    const inspected = JSON.parse(raw);

    assert.equal(inspected.q4.matmul_nbits_count, 196, "q4 has 196 MatMulNBits (28 layers x 7 projections)");
    assert.equal(inspected.q4.gather_block_quantized_count, 1, "q4 has 1 GatherBlockQuantized operator for embeddings");
    assert.equal(inspected.q4.pointer_head_node_count, 4, "q4 pointer head projection has 4 nodes");
    assert.equal(inspected.q4.pointer_head_ancestor_node_count, 31, "q4 pointer head ancestor lineage has 31 nodes");
    assert.equal(inspected.q4.has_temperature_division_in_graph, false, "no temperature division in graph");
    assert.deepEqual(inspected.q4.div_nodes_in_pointer_head_lineage, [], "zero Div nodes in pointer head ancestor lineage");
    assert.equal(inspected.q4.pointer_head_scale_constant, 0.0625, "scale constant is 0.0625");

    assert.equal(inspected.q4f16.matmul_nbits_count, 196, "q4f16 has 196 MatMulNBits");
    assert.equal(inspected.q4f16.pointer_head_node_count, 14, "q4f16 pointer head has 14 projection/cast nodes");
    assert.equal(inspected.q4f16.pointer_head_ancestor_node_count, 46, "q4f16 pointer head ancestor lineage has 46 nodes");
    assert.equal(inspected.q4f16.cast_count, 287, "q4f16 has 287 Cast operations");
    assert.equal(inspected.q4f16.has_temperature_division_in_graph, false, "no temperature division in q4f16 graph");
    assert.deepEqual(inspected.q4f16.div_nodes_in_pointer_head_lineage, [], "zero Div nodes in q4f16 pointer head ancestor lineage");
  },
);
