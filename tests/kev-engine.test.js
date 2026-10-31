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

test("a hub that accepts the connection and then stalls fails within the idle deadline (tm-download-no-timeout)", async () => {
  const originalFetch = globalThis.fetch;
  // The mock honours the signal it is handed: if the deadline is not actually wired to the
  // request, the read below never rejects and this test fails by hanging.
  let handedSignal = null;
  const mockResponse = {
    ok: true,
    status: 200,
    headers: new Headers({ "content-length": "10" }),
    get body() {
      return new ReadableStream({
        start(controller) {
          handedSignal?.addEventListener("abort", () => controller.error(handedSignal.reason));
        },
      });
    },
  };
  globalThis.fetch = async (url, options) => {
    handedSignal = options.signal;
    return mockResponse;
  };
  try {
    const t0 = Date.now();
    await assert.rejects(
      fetchVerified("https://fixture.invalid/tokenizer.json", {
        sha256: "0".repeat(64),
        label: "fixture",
        timeoutMs: 25,
      }),
      /no data within 25ms/,
    );
    assert.ok(handedSignal, "the abort signal was passed to fetch");
    assert.equal(handedSignal.reason?.name, "TimeoutError", "the deadline aborted with a TimeoutError");
    assert.ok(Date.now() - t0 < 5000, "the stalled download failed fast instead of hanging forever");
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

// learnings-rk51: the tokenizer hardening was guarded only by a source-text regex in
// kev-pack.test.js ("the engine must pass specialStrings: this.specials?.literals"), so a refactor
// could keep every string and still stop threading the value. This drives the real KevSession:
// loader-derived specials in, planted control tokens out. The unwired control below is the point —
// the same text leaks when the session carries no specials, so the assertions above are testing
// the wiring and not the tokenizer stand-in.
test("a KevSession carrying its tokenizer's specials drops planted control tokens at runtime", async () => {
  const { KevSession } = await import("../site/decision-models/kev-engine.js");
  const { tokenizerSpecials } = await import("../site/decision-models/tokenizer-specials.mjs");
  const ort = await import("../site/vendor/ort.bundle.min.mjs");

  const delimiterIds = { state: 1, question: 2, option_start: 3, option_end: 4, decide: 5 };
  const LITERAL = "<|endoftext|>"; // declared special:true, so the text-level neutralizer removes it
  const FOLDED = 901; // declared special id the tokenizer can produce from text no literal match sees
  const tokenizerJson = {
    added_tokens: [
      { id: 900, content: LITERAL, special: true },
      { id: FOLDED, content: "<CTRL>", special: true, normalized: true },
      { id: 902, content: "<notspecial>", special: false },
    ],
  };
  // What loadKev derives from the tokenizer it loaded (the loader half is regex-guarded above).
  const specials = tokenizerSpecials(tokenizerJson);
  assert.deepEqual(specials.ids, [900, FOLDED], "specials come from special:true added tokens only");

  // A tokenizer stand-in with the two behaviours that matter: it emits the declared literal's id,
  // and it produces the declared control id from ordinary-looking reader text ("control"), the way
  // a Strip/Replace normalizer could.
  const tokenize = (text) => {
    const out = [];
    let rest = String(text);
    while (rest) {
      if (rest.startsWith(LITERAL)) { out.push(900); rest = rest.slice(LITERAL.length); continue; }
      if (rest.startsWith("control")) { out.push(FOLDED); rest = rest.slice("control".length); continue; }
      const [ch] = Array.from(rest);
      if (!/\s/.test(ch)) out.push(ch.codePointAt(0) + 100);
      rest = rest.slice(ch.length);
    }
    return out;
  };

  const build = (threaded) => {
    const seen = [];
    const session = {
      run: async (feeds) => {
        const packed = Array.from(feeds.input_ids.data, Number);
        seen.push(packed);
        const data = new Float32Array(packed.length).fill(0);
        data[packed.length - 1] = 3; // make the last option_end win, deterministically
        return { logits: new ort.Tensor("float32", data, [1, packed.length]) };
      },
    };
    return {
      seen,
      kev: new KevSession({
        tokenizer: { encode: (text) => ({ ids: tokenize(text) }) },
        session,
        ids: delimiterIds,
        specials: threaded ? specials : null,
        checkpoint: "fixture",
        loadMs: 0,
      }),
    };
  };

  const state = `before ${LITERAL} control after`;
  const questions = [{ instruction: "is it safe", options: ["no", "yes"] }];

  const wired = build(true);
  const wiredResult = await wired.kev.decidePacked(state, questions);
  const packed = wired.seen.at(-1);
  assert.equal(packed.includes(900), false, "the declared literal's id never reaches the sequence");
  assert.equal(packed.includes(FOLDED), false, "the declared control id the tokenizer produced is dropped");
  assert.equal(packed[0], delimiterIds.state, "the state delimiter is still the first token");
  assert.ok(packed.includes("b".codePointAt(0) + 100) && packed.includes("a".codePointAt(0) + 100),
    "reader text either side of the planted token survives (over-neutralization guard)");
  assert.equal(wiredResult.distributions.length, 1, "one question, one distribution");
  assert.equal(wiredResult.distributions[0].length, 2, "read over the option delimiters");
  assert.ok(Math.abs(wiredResult.distributions[0].reduce((sum, p) => sum + p, 0) - 1) < 1e-9, "a distribution sums to 1");
  assert.equal(wiredResult.tokens, packed.length, "the reported token count is the packed length");
  assert.equal(wiredResult.flat, false, "two different scores are not a flat distribution");

  // CONTROL: the shape escape (layer 1 of the three) already rewrites `<|endoftext|>` to
  // `<¦endoftext¦>` before the tokenizer sees it, so the literal alone proves nothing about the
  // declared-specials wiring. The normalized id is the half ONLY the wiring can stop: with no
  // specials threaded it leaks, which is what a source regex cannot show.
  const unwired = build(false);
  await unwired.kev.decidePacked(state, questions);
  const leaky = unwired.seen.at(-1);
  assert.ok(leaky.includes(FOLDED), "unwired: the declared control id the tokenizer produces leaks");
  assert.equal(leaky.includes(900), false, "the shape escape already handled the literal with no specials at all");
});
