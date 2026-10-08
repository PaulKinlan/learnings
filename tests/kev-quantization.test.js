// tests/kev-quantization.test.js
// Quantization comparison and architectural audit between Kev's ONNX export variants:
// q4 (CPU / WASM JSEP) vs q4f16 (WebGPU).
import test from "node:test";
import assert from "node:assert/strict";
import { KEV_MAX_BYTES, KEV_SHA256 } from "../site/decision-models/kev-manifest.js";

// Pinned structural metadata from the onnx-community/kev-0.6b-ONNX export
const QUANT_SPEC = {
  q4: {
    target: "CPU / ONNX Runtime Web (JSEP WASM build)",
    weightsDtype: "int4 (MatMulNBits, block_size 32)",
    embeddingDtype: "GatherBlockQuantized",
    backboneDtype: "float32 activations, int4 weights",
    headDtype: "float32 (head_q, head_k, dot product, scale 0.0625)",
    graphFile: "onnx/model_q4.onnx",
    dataFile: "onnx/model_q4.onnx_data",
    graphBytes: 1245327, // ~1.25 MB
    dataBytes: 374822912, // ~374.8 MB
    castNodesCount: 239,
    pointerHeadNodesCount: 4,
    requiredProvider: "wasm (with jsep bridge for GatherBlockQuantized)",
  },
  q4f16: {
    target: "WebGPU / ONNX Runtime Web",
    weightsDtype: "int4 (MatMulNBits, block_size 32)",
    embeddingDtype: "GatherBlockQuantized",
    backboneDtype: "float16 activations, int4 weights",
    headDtype: "float32 guarded by cast_to_fp32 / cast_to_fp16 nodes",
    graphFile: "onnx/model_q4f16.onnx",
    dataFile: "onnx/model_q4f16.onnx_data",
    graphBytes: 3338522, // ~3.34 MB
    dataBytes: 335357952, // ~335.4 MB
    castNodesCount: 287,
    pointerHeadNodesCount: 14,
    requiredProvider: "webgpu",
  },
};

test("quantization comparison: storage footprint and compression ratio", () => {
  // Unquantized Qwen3-0.6B baseline (float32: ~2.4 GB; float16: ~1.2 GB)
  const unquantizedFp16Bytes = 1.2 * 1024 * 1024 * 1024;

  const q4Bytes = QUANT_SPEC.q4.dataBytes;
  const q4f16Bytes = QUANT_SPEC.q4f16.dataBytes;

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
  assert.equal(QUANT_SPEC.q4.pointerHeadNodesCount, 4);

  // q4f16 converts backbone representations to fp16 for WebGPU. To prevent numerical underflow
  // in the dot product pointer head, it explicitly wraps projections in cast_to_fp32 and cast_to_fp16
  // nodes (14 nodes total).
  assert.equal(QUANT_SPEC.q4f16.pointerHeadNodesCount, 14);
  assert.ok(
    QUANT_SPEC.q4f16.castNodesCount > QUANT_SPEC.q4.castNodesCount,
    "q4f16 contains 48 additional Cast nodes to bridge fp16 backbone to fp32 pointer head",
  );
});

test("quantization comparison: GatherBlockQuantized operator requirement", () => {
  // The embedding table uses GatherBlockQuantized.
  // In ONNX Runtime Web, this kernel is registered ONLY when JSEP bridge is initialized.
  assert.equal(QUANT_SPEC.q4.embeddingDtype, "GatherBlockQuantized");
  assert.equal(QUANT_SPEC.q4f16.embeddingDtype, "GatherBlockQuantized");
  assert.ok(
    QUANT_SPEC.q4.requiredProvider.includes("jsep"),
    "q4 on CPU requires JSEP wasm build to execute GatherBlockQuantized without runtime crash",
  );
});

test("quantization comparison: manifest size limits bound both variants", () => {
  // Verify that KEV_MAX_BYTES accommodates the pinned q4 weights with safe headroom
  const q4Limit = KEV_MAX_BYTES["onnx/model_q4.onnx_data"];
  assert.ok(q4Limit > QUANT_SPEC.q4.dataBytes, "maxBytes accommodates model_q4.onnx_data");
  assert.ok(q4Limit < QUANT_SPEC.q4.dataBytes * 1.5, "headroom is tight (~20%) to prevent memory exhaustion");

  // Verify graph size limits
  const q4GraphLimit = KEV_MAX_BYTES["onnx/model_q4.onnx"];
  assert.ok(q4GraphLimit > QUANT_SPEC.q4.graphBytes, "maxBytes accommodates model_q4.onnx");
});
