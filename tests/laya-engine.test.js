// tests/laya-engine.test.js — the Laya weight provenance. The engine must fetch its graphs, its
// token table and the SHA256SUMS it checks them against from an immutable revision, and the
// digests it checks against must be the ones that revision published. No model, no browser, no
// network: everything here is derived from the checked-in manifest and the vendored sums file.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import {
  LAYA_REVISION,
  LAYA_SHA256SUMS_SHA256,
  LAYA_MAX_BYTES,
  LAYA_DEFAULT_MAX_BYTES,
} from "../site/decision-models/laya-manifest.js";
import { layaArtifactUrl, fetchBytes } from "../site/decision-models/laya-engine.js";

const REPOS = {
  multilingual: "litert-community/Laya-Multilingual-LiteRT",
  english: "litert-community/laya-LiteRT",
};

// Every path loadLaya can resolve per checkpoint: the artifacts it fetches plus the sums file it
// verifies them against. The sums file is in the list because a hash list fetched from a mutable
// ref is as replaceable as the bytes it describes.
const FETCHED = {
  multilingual: [
    "laya_ml_s512_embeds_wfp16.tflite",
    "laya_ml_act_head_fp32.tflite",
    "token_embeddings_fp16.bin",
    "tokenizer.json",
    "laya_ml_calibration.json",
    "SHA256SUMS",
  ],
  english: ["laya_en_s512_wfp16.tflite", "laya_act_head_fp32.tflite", "tokenizer.json", "SHA256SUMS"],
};

test("every artifact and the sums file are pinned to an immutable revision, not a branch", () => {
  for (const [checkpoint, files] of Object.entries(FETCHED)) {
    assert.match(LAYA_REVISION[checkpoint], /^[0-9a-f]{40}$/, `${checkpoint}: the pin is a commit sha`);
    for (const file of files) {
      const url = layaArtifactUrl(checkpoint, file);
      assert.equal(
        url,
        `https://huggingface.co/${REPOS[checkpoint]}/resolve/${LAYA_REVISION[checkpoint]}/${file}`,
      );
      assert.match(url, /\/resolve\/[0-9a-f]{40}\//, `${file} resolves through a commit sha`);
      assert.ok(!url.includes("/main/"), `${file} has no path through the mutable main branch`);
    }
  }
  assert.throws(() => layaArtifactUrl("nope", "tokenizer.json"), /unknown checkpoint 'nope'/);
});

test("the vendored multilingual sums are the pinned revision's own file, and cover what it fetches", () => {
  const bytes = readFileSync(new URL("../site/decision-models/laya/ml-SHA256SUMS", import.meta.url));
  const text = bytes.toString("utf8");
  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    LAYA_SHA256SUMS_SHA256.multilingual,
    "laya/ml-SHA256SUMS is byte-identical to SHA256SUMS at the pinned revision: re-pin both together",
  );
  for (const file of FETCHED.multilingual) {
    if (file === "SHA256SUMS") continue;
    const escaped = file.replaceAll(".", "\\.");
    assert.match(
      text,
      new RegExp(`^[0-9a-f]{64} {1,2}\\*?${escaped}$`, "m"),
      `${file} has a recorded digest, so its fetch is actually checked`,
    );
  }
  for (const [local, upstream] of [
    ["ml-tokenizer.json", "tokenizer.json"],
    ["laya_ml_act_head_fp32.tflite", "laya_ml_act_head_fp32.tflite"],
    ["laya_ml_calibration.json", "laya_ml_calibration.json"],
  ]) {
    const digest = createHash("sha256")
      .update(readFileSync(new URL(`../site/decision-models/laya/${local}`, import.meta.url)))
      .digest("hex");
    assert.ok(text.includes(`${digest}  ${upstream}\n`), `${local} matches the pinned ${upstream} digest`);
  }
});

test("every artifact has a documented maximum size cap in LAYA_MAX_BYTES", () => {
  for (const [checkpoint, files] of Object.entries(FETCHED)) {
    for (const file of files) {
      const cap = LAYA_MAX_BYTES[file];
      assert.equal(typeof cap, "number", `${checkpoint}/${file} has a numeric cap`);
      assert.ok(cap > 0, `${checkpoint}/${file} cap is positive`);
    }
  }
});

test("fetchBytes: Content-Length header exceeding maximum allowed size aborts before reading body", async () => {
  const originalFetch = globalThis.fetch;
  let bodyAccessed = false;
  const mockResponse = {
    ok: true,
    status: 200,
    headers: new Headers({ "content-length": "200" }),
    get body() {
      bodyAccessed = true;
      return new ReadableStream();
    },
  };
  globalThis.fetch = async () => mockResponse;
  try {
    await assert.rejects(
      fetchBytes("https://fixture.invalid/tokenizer.json", {
        sha256: "0".repeat(64),
        label: "fixture",
        maxBytes: 100,
      }),
      /Content-Length 200 exceeds maximum allowed size 100 bytes/,
    );
    assert.equal(bodyAccessed, false, "streaming body was never accessed because Content-Length failed fast");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("fetchBytes: streamed chunks exceeding maximum allowed size abort streaming before full body is buffered", async () => {
  let streamCancelled = false;
  let chunksEnqueued = 0;
  const chunk1 = new Uint8Array(80);
  const chunk2 = new Uint8Array(80);
  const chunk3 = new Uint8Array(80);
  const fullBytes = new Uint8Array(240);
  fullBytes.set(chunk1, 0);
  fullBytes.set(chunk2, 80);
  fullBytes.set(chunk3, 160);
  const goodDigest = createHash("sha256").update(fullBytes).digest("hex");

  const stream = new ReadableStream({
    pull(controller) {
      if (streamCancelled) return;
      chunksEnqueued++;
      if (chunksEnqueued === 1) controller.enqueue(chunk1);
      else if (chunksEnqueued === 2) controller.enqueue(chunk2);
      else if (chunksEnqueued === 3) controller.enqueue(chunk3);
      else controller.close();
    },
    cancel() {
      streamCancelled = true;
    },
  });

  const originalFetch = globalThis.fetch;
  // No Content-Length header to verify stream chunk enforcement
  globalThis.fetch = async () => new Response(stream);
  try {
    const outcome = await fetchBytes("https://fixture.invalid/custom", {
      sha256: goodDigest,
      label: "fixture",
      maxBytes: 100,
    }).then(
      (bytes) => ({ status: "buffered", byteLength: bytes.byteLength }),
      (err) => ({ status: "rejected", message: err.message }),
    );

    assert.equal(
      outcome.status,
      "rejected",
      `oversized body was buffered (${outcome.byteLength} bytes) instead of rejected`,
    );
    assert.match(outcome.message, /stream exceeded maximum allowed size 100 bytes/);
    assert.equal(streamCancelled, true, "underlying stream was cancelled via AbortController");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("inherited Object.prototype keys cannot suppress the bounded fallback cap", async () => {
  const oneByte = new Uint8Array([7]);
  const goodDigest = createHash("sha256").update(oneByte).digest("hex");
  for (const name of ["constructor", "__proto__", "toString"]) {
    const originalFetch = globalThis.fetch;
    const mockResponse = {
      ok: true,
      status: 200,
      headers: new Headers({ "content-length": "2000000000" }),
      get body() {
        return new ReadableStream({
          start(controller) {
            controller.enqueue(oneByte);
            controller.close();
          },
        });
      },
    };
    globalThis.fetch = async () => mockResponse;
    try {
      await assert.rejects(
        fetchBytes(`https://fixture.invalid/${name}`, { sha256: goodDigest }),
        new RegExp(`Content-Length 2000000000 exceeds maximum allowed size ${LAYA_DEFAULT_MAX_BYTES} bytes`),
        `${name}: an inherited Object.prototype member suppressed the bounded fallback cap`,
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  }
});

test("explicit non-finite or non-positive maxBytes fails closed instead of accepting an oversized body", async () => {
  const oneByte = new Uint8Array([7]);
  const goodDigest = createHash("sha256").update(oneByte).digest("hex");
  for (const bad of [NaN, Infinity, "invalid", 0, -1]) {
    const originalFetch = globalThis.fetch;
    const mockResponse = {
      ok: true,
      status: 200,
      headers: new Headers({ "content-length": "2000000000" }),
      get body() {
        return new ReadableStream({
          start(controller) {
            controller.enqueue(oneByte);
            controller.close();
          },
        });
      },
    };
    globalThis.fetch = async () => mockResponse;
    try {
      const outcome = await fetchBytes("https://fixture.invalid/custom", {
        sha256: goodDigest,
        label: "fixture",
        maxBytes: bad,
      }).then(
        (bytes) => ({ status: "accepted", byteLength: bytes.byteLength }),
        (err) => ({ status: "rejected", message: err.message }),
      );
      assert.equal(
        outcome.status,
        "rejected",
        `maxBytes=${String(bad)}: a 2 GB Content-Length body was accepted instead of failing closed`,
      );
      assert.match(
        outcome.message,
        /invalid maxBytes for fixture/,
        `maxBytes=${String(bad)}: rejection must name the invalid explicit cap`,
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  }
});
