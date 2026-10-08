// The Kev weight identity. The engine must run only bytes fetched from an immutable revision
// and matching the checked-in digests; these checks run without a model or a browser.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { KEV_REVISION, KEV_SHA256, KEV_MAX_BYTES, KEV_DEFAULT_MAX_BYTES } from "../site/decision-models/kev-manifest.js";
import { fetchVerified, kevArtifactUrl } from "../site/decision-models/kev-engine.js";

const ARTIFACTS = ["tokenizer.json", "config.json", "onnx/model_q4.onnx", "onnx/model_q4.onnx_data"];

test("every artifact is pinned to an immutable revision, not a branch", () => {
  assert.match(KEV_REVISION, /^[0-9a-f]{40}$/, "the revision is a commit sha");
  assert.deepEqual(
    Object.keys(KEV_SHA256).sort(),
    [...ARTIFACTS].sort(),
    "the manifest covers exactly the four artifacts the engine fetches",
  );
  for (const file of ARTIFACTS) {
    assert.match(KEV_SHA256[file], /^[0-9a-f]{64}$/, `${file} has a sha256`);
    const url = kevArtifactUrl(file);
    assert.equal(url, `https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/${KEV_REVISION}/${file}`);
    assert.ok(!url.includes("/resolve/main/"), "no URL resolves through the mutable main branch");
  }
});

test("a digest mismatch aborts before the bytes are returned", async () => {
  const bytes = Buffer.from("not the pinned kev weights");
  const url = `data:application/octet-stream;base64,${bytes.toString("base64")}`;
  const good = createHash("sha256").update(bytes).digest("hex");

  const verified = await fetchVerified(url, { sha256: good, label: "fixture" });
  assert.deepEqual(Buffer.from(verified), bytes, "matching bytes come back");

  await assert.rejects(
    fetchVerified(url, { sha256: "0".repeat(64), label: "fixture" }),
    /integrity check failed for fixture/,
  );
  await assert.rejects(fetchVerified(url, { label: "fixture" }), /no pinned sha256 for fixture/);
});

test("every artifact has a documented maximum size cap consistent with known sizes", () => {
  assert.deepEqual(
    Object.keys(KEV_MAX_BYTES).sort(),
    [...ARTIFACTS].sort(),
    "maximum size limits are configured for all known Kev artifacts",
  );
  for (const file of ARTIFACTS) {
    assert.equal(typeof KEV_MAX_BYTES[file], "number");
    assert.ok(KEV_MAX_BYTES[file] > 0, `${file} limit is positive`);
  }
});

test("Content-Length header exceeding maximum allowed size aborts before reading body", async () => {
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
      fetchVerified("https://fixture.invalid/tokenizer.json", {
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

test("streamed chunks exceeding maximum allowed size abort streaming before full body is buffered", async () => {
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
    const outcome = await fetchVerified("https://fixture.invalid/custom", {
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
        fetchVerified(`https://fixture.invalid/${name}`, { sha256: goodDigest }),
        new RegExp(`Content-Length 2000000000 exceeds maximum allowed size ${KEV_DEFAULT_MAX_BYTES} bytes`),
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
      const outcome = await fetchVerified("https://fixture.invalid/custom", {
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
