// tests/kev-memory.test.js
// Device memory bounds, lifecycle management, and architectural memory comparison for Kev.
import test from "node:test";
import assert from "node:assert/strict";
import { KevSession } from "../site/decision-models/kev-engine.js";
import { KEV_MAX_BYTES, KEV_DEFAULT_MAX_BYTES } from "../site/decision-models/kev-manifest.js";
import { LAYA_MAX_BYTES } from "../site/decision-models/laya-manifest.js";

test("device memory bounds: every Kev artifact has a bounded positive cap", () => {
  const artifacts = [
    "config.json",
    "tokenizer.json",
    "onnx/model_q4.onnx",
    "onnx/model_q4.onnx_data",
  ];

  for (const artifact of artifacts) {
    const cap = KEV_MAX_BYTES[artifact];
    assert.equal(typeof cap, "number", `${artifact} cap is a number`);
    assert.ok(Number.isFinite(cap), `${artifact} cap is finite`);
    assert.ok(cap > 0, `${artifact} cap is positive`);
  }

  assert.ok(
    Number.isFinite(KEV_DEFAULT_MAX_BYTES) && KEV_DEFAULT_MAX_BYTES > 0,
    "fallback default maxBytes is a finite positive number",
  );
});

test("device memory lifecycle: session disposal releases session handle and clears token cache", async () => {
  let releaseCalled = 0;
  const mockOrtSession = {
    release: async () => {
      releaseCalled++;
    },
  };

  const session = new KevSession({
    tokenizer: { encode: () => ({ ids: [1, 2, 3] }) },
    session: mockOrtSession,
    ids: { state: 1, question: 2, option_start: 3, option_end: 4, decide: 5 },
    checkpoint: "mock-checkpoint",
    loadMs: 100,
  });

  // Verify token cache populates with cached sequences
  session.tokenize("Hello world");
  session.tokenize("Ticket regarding billing issue");
  assert.equal(session.cache.size, 2, "token cache holds tokenized representation");

  // First disposal
  await session.delete();
  assert.equal(releaseCalled, 1, "release() called on ORT session");
  assert.equal(session.cache.size, 0, "session token cache cleared upon disposal");
  assert.equal(session.session, null, "ORT session reference nulled to break retain cycles");

  // Second disposal must be idempotent: does not call release on nulled session, does not throw
  await assert.doesNotReject(session.delete(), "subsequent delete() calls do not throw");
  assert.equal(releaseCalled, 1, "second delete does not re-invoke release on nulled session");

  // Disposal on a session created with no session object
  const nullSession = new KevSession({
    tokenizer: null,
    session: null,
    ids: null,
    checkpoint: "null-checkpoint",
    loadMs: 0,
  });
  await assert.doesNotReject(nullSession.delete(), "delete() handles null session safely");
});

test("architectural memory comparison: Kev vs Laya-LiteRT footprint", () => {
  // Kev q4 total bytes
  const kevBytes =
    KEV_MAX_BYTES["config.json"] +
    KEV_MAX_BYTES["tokenizer.json"] +
    KEV_MAX_BYTES["onnx/model_q4.onnx"] +
    KEV_MAX_BYTES["onnx/model_q4.onnx_data"];

  // Laya multilingual total bytes
  const layaBytes =
    LAYA_MAX_BYTES["tokenizer.json"] +
    LAYA_MAX_BYTES["token_embeddings_fp16.bin"] +
    LAYA_MAX_BYTES["laya_ml_s512_embeds_wfp16.tflite"] +
    LAYA_MAX_BYTES["laya_ml_act_head_fp32.tflite"] +
    LAYA_MAX_BYTES["laya_ml_calibration.json"];

  assert.ok(
    kevBytes < layaBytes,
    `Kev total download cap (${(kevBytes / 1048576).toFixed(0)} MB) is smaller than Laya (${(layaBytes / 1048576).toFixed(0)} MB)`,
  );

  // Kev performs single forward pass for all questions in batch, whereas Laya runs
  // sequential passes per question. Single-pass batching prevents duplicate activation memory allocations.
  assert.ok(
    kevBytes / layaBytes < 0.7,
    "Kev download footprint is over 30% more compact than Laya multilingual bundle",
  );
});
