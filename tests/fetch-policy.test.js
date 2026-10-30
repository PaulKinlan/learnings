// tests/fetch-policy.test.js — the request policy every artifact download must carry
// (learnings-902), plus the redirect check that runs before a byte is consumed.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import {
  ARTIFACT_HOST_SUFFIXES,
  artifactFetchOptions,
  artifactHostAllowed,
  assertArtifactResponse,
} from "../site/decision-models/fetch-policy.mjs";
import { fetchVerified, loadKev } from "../site/decision-models/kev-engine.js";
import { PROVIDER_FETCH_OPTIONS } from "../site/decision-models/core.js";
import { fetchBytes } from "../site/decision-models/laya-engine.js";
import { LAYA_SHA256SUMS_SHA256 } from "../site/decision-models/laya-manifest.js";
import { serve } from "../scripts/serve.mjs";

test("artifact requests carry no credentials and no referrer", () => {
  const options = artifactFetchOptions({ signal: "sig" });
  assert.equal(options.credentials, "omit", "an artifact request must not carry ambient credentials");
  assert.equal(options.referrerPolicy, "no-referrer", "the reader's page URL must not leave the origin");
  assert.equal(options.signal, "sig", "the caller's abort signal still reaches fetch");
});

test("only https artifact hosts and their subdomains are accepted", () => {
  for (const host of ["huggingface.co", "cdn-lfs.huggingface.co", "hf.co", "cdn-lfs-us-1.hf.co", "cas-bridge.xethub.hf.co"])
    assert.ok(artifactHostAllowed(host), `${host} serves artifacts`);
  for (const host of ["evil.example", "huggingface.co.evil.example", "not-hf.co", "", null, "127.0.0.1"])
    assert.ok(!artifactHostAllowed(host), `${host} must be refused`);
  assert.deepEqual(ARTIFACT_HOST_SUFFIXES, ["huggingface.co", "hf.co"]);
});

test("a response that redirected off the artifact hosts is refused, and one that stayed is accepted", () => {
  const hub = "https://huggingface.co/repo/resolve/rev/weights.bin";
  // The legitimate path: huggingface.co redirects to its CDN.
  assert.doesNotThrow(() => assertArtifactResponse({ url: "https://cdn-lfs-us-1.hf.co/x/y" }, "weights", hub));
  // A response with no url (a locally built one) is left to the sha256 check.
  assert.doesNotThrow(() => assertArtifactResponse({}, "weights", hub));
  assert.throws(() => assertArtifactResponse({ url: "https://evil.example/weights" }, "weights", hub),
    /redirect left the artifact hosts \(evil\.example\)/);
  // A locally built fixture never leaves the origin: it is judged by the sha256 pin alone.
  assert.doesNotThrow(() => assertArtifactResponse({ url: "data:application/octet-stream;base64,AAAA" }, "fixture", hub));
  assert.doesNotThrow(() => assertArtifactResponse({ url: "blob:https://example.test/1" }, "fixture", hub));
  assert.throws(() => assertArtifactResponse({ url: "http://huggingface.co/weights" }, "weights", hub),
    /redirect downgraded to http:/);
  assert.throws(() => assertArtifactResponse({ url: "not a url" }, "weights", hub), /unparseable final URL/);
  // A cross-origin request cannot escape by landing on some other site.
  assert.throws(() => assertArtifactResponse({ url: "https://paulkinlan.github.io/learnings/x" }, "weights", hub),
    /redirect left the artifact hosts \(paulkinlan\.github\.io\)/);
});

// learnings-0px: Laya vendors its SHA256SUMS, tokenizer, act head and calibration first-party
// (site/decision-models/laya/), so the deployed page fetches them from its own origin. A
// host-only check rejected them and Laya could not load in production.
test("hosted provider calls carry redirect/credentials/referrer hardening (INV-FETCH-HYGIENE, learnings-hwi)", async () => {
  // The policy has one source of truth.
  assert.deepEqual(PROVIDER_FETCH_OPTIONS, {
    redirect: "error",
    credentials: "omit",
    referrerPolicy: "no-referrer",
  });
  // image-lab's hosted TypeSafe call, exercised end to end with a stubbed transport.
  let captured = null;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    captured = { url, init };
    return { ok: true, json: async () => ({ answers: { tool: { type: "noul", noul: 0.9 } } }) };
  };
  const { decideImage } = await import("../site/decision-models/image-lab.js");
  const imageSpec = { title: "t", description: "d", state: "s", questions: { tool: { type: "noul", instructions: "Is it?" } } };
  const imagePayload = "data:image/png;base64,AA==";
  try {
    await decideImage({ spec: imageSpec, imagePayload, engine: "api", key: "synthetic-test-only" });
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(captured.url, "https://api.typesafe.ai/v1/systemone");
  for (const [option, value] of Object.entries(PROVIDER_FETCH_OPTIONS)) {
    assert.equal(captured.init[option], value, `the hosted TypeSafe call carries ${option}: ${value}`);
  }
  // learnings-dcp: the LOCAL loopback transport carries the same payload — the uploaded image
  // and the reader's questions — so it is bound by the same policy, not just the hosted call.
  captured = null;
  globalThis.fetch = async (url, init) => {
    captured = { url, init };
    return { ok: true, json: async () => ({ answers: { tool: { type: "noul", noul: 0.9 } } }) };
  };
  try {
    await decideImage({ spec: imageSpec, imagePayload, engine: "local", endpoint: "http://127.0.0.1:9/v1/systemone" });
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(captured.url, "http://127.0.0.1:9/v1/systemone");
  for (const [option, value] of Object.entries(PROVIDER_FETCH_OPTIONS)) {
    assert.equal(captured.init[option], value, `the local loopback call carries ${option}: ${value}`);
  }
  // The playground's hosted Jev call is a page driver with no exportable boundary, so pin that
  // the call site spreads the same shared policy instead of re-litigating three literals.
  const playgroundSrc = readFileSync(new URL("../site/decision-models/playground.js", import.meta.url), "utf8");
  const hostedCall = playgroundSrc.match(/fetch\("https:\/\/api\.typesafe\.ai\/v1\/systemone", \{[\s\S]*?\}\);/);
  assert.ok(hostedCall, "the playground hosted fetch call site is still recognizable");
  assert.match(hostedCall[0], /\.\.\.PROVIDER_FETCH_OPTIONS/, "the playground spreads the shared provider fetch policy");
});

test("a first-party vendored artifact is accepted while foreign redirects still fail", () => {
  const vendored = "https://paulkinlan.github.io/learnings/decision-models/laya/ml-SHA256SUMS";
  assert.doesNotThrow(() => assertArtifactResponse({ url: vendored }, "vendored sums", vendored),
    "the site's own vendored artifact must load");
  // The allowance is source-aware: it never accepts a foreign final URL for a cross-origin
  // request, and it does not accept a same-origin artifact that downgrades to http.
  assert.throws(() => assertArtifactResponse({ url: vendored }, "weights", "https://huggingface.co/x"),
    /redirect left the artifact hosts/);
  assert.throws(() => assertArtifactResponse({ url: "http://paulkinlan.github.io/learnings/decision-models/laya/ml-SHA256SUMS" }, "vendored sums", vendored),
    /redirect downgraded to http:/);
});
// Wiring, exercised through the real downloader rather than a source-string match: a stubbed
// fetch sees the policy options, and a redirect off the hub fails before the body is read.
test("fetchVerified sends the artifact policy and rejects an off-host redirect", async () => {
  const original = globalThis.fetch;
  const seen = [];
  try {
    globalThis.fetch = async (url, options) => {
      seen.push(options);
      return { ok: true, url: "https://evil.example/weights.bin", headers: new Headers({}), body: null };
    };
    await assert.rejects(
      fetchVerified("https://huggingface.co/repo/resolve/rev/weights.bin", { sha256: "a".repeat(64), label: "weights" }),
      /redirect left the artifact hosts/,
    );
    assert.equal(seen.length, 1);
    assert.equal(seen[0].credentials, "omit");
    assert.equal(seen[0].referrerPolicy, "no-referrer");
    assert.ok(seen[0].signal, "the abort signal is still passed");
  } finally {
    globalThis.fetch = original;
  }
});

test("both engines route their downloads through the one shared bounded fetch (learnings-0ef)", () => {
  const expectations = {
    "kev-engine.js": ["fetchArtifactBytes"],
    "laya-engine.js": ["boundedArtifactRead", "fetchArtifactBytes", "ownCap", "verifyArtifactSha256"],
  };
  for (const [file, helpers] of Object.entries(expectations)) {
    const source = readFileSync(new URL(`../site/decision-models/${file}`, import.meta.url), "utf8");
    assert.match(source, /import \{[^}]*\} from "\.\/fetch-policy\.mjs";/, `${file} must import the shared policy`);
    for (const helper of helpers) {
      assert.match(source, new RegExp(`\\b${helper}\\b`), `${file} must use the shared ${helper}`);
    }
    // The deadline is the policy module's too: no per-engine timeout constants left to drift.
    assert.match(source, /ARTIFACT_DOWNLOAD_TIMEOUT_MS/, `${file} must share the artifact deadline`);
    assert.doesNotMatch(source, /_DOWNLOAD_TIMEOUT_MS\s*=/, `${file} must not define its own timeout constant`);
    assert.equal((source.match(/await fetch\(/g) ?? []).length, 0, `${file}: no direct fetch may remain outside the policy`);
    assert.equal((source.match(/assertArtifactResponse\(/g) ?? []).length, 0, `${file}: response validation lives in the policy now`);
  }
});

// The real load path, not a stub: the DEFAULT vendored SHA256SUMS Laya fetches in production,
// served over HTTP from this origin, through the hardened downloader. This is the exact request
// that failed with "redirect left the artifact hosts" before the source-aware check (learnings-0px).
test("the default vendored SHA256SUMS fetches through the hardened path over real HTTP", async () => {
  const local = await serve();
  try {
    const sumsUrl = new URL("decision-models/laya/ml-SHA256SUMS", local.url).href;
    const bytes = await fetchBytes(sumsUrl, { sha256: LAYA_SHA256SUMS_SHA256.multilingual, label: "ml-SHA256SUMS" });
    assert.ok(bytes.byteLength > 0, "the vendored sums file came back");
    assert.match(Buffer.from(bytes).toString("utf8"), /[0-9a-f]{64}/, "and it is the real sums table");
    // The same hardened path still refuses a foreign final URL; no network is touched here.
    const original = globalThis.fetch;
    try {
      globalThis.fetch = async () => ({ ok: true, url: "https://evil.example/weights.bin", headers: new Headers({}), body: null });
      await assert.rejects(
        fetchVerified("https://huggingface.co/repo/resolve/rev/x", { sha256: "a".repeat(64), label: "x" }),
        /redirect left the artifact hosts/,
      );
    } finally {
      globalThis.fetch = original;
    }
  } finally {
    await new Promise((resolve) => local.server.close(resolve));
  }
});

// learnings-rzp: the sibling of the explicitCap fail-closed check. An invalid deadline reaches
// setTimeout, where NaN/0 fire immediately and abort EVERY download at once — a silent total
// outage of the download path instead of a loud bug.
test("an invalid timeoutMs fails loudly instead of aborting every download (learnings-rzp)", async () => {
  for (const bad of [NaN, Infinity, 0, -1, "500"]) {
    await assert.rejects(
      fetchVerified("data:text/plain,x", { sha256: "0".repeat(64), label: "t", timeoutMs: bad }),
      /invalid timeoutMs for t/,
      `timeoutMs=${String(bad)} must be refused at the boundary`,
    );
    await assert.rejects(
      fetchBytes("data:text/plain,x", { sha256: "0".repeat(64), label: "t", timeoutMs: bad }),
      /invalid timeoutMs for t/,
      `fetchBytes: timeoutMs=${String(bad)} must be refused at the boundary`,
    );
  }
});

// learnings-q61: the deadline is an IDLE bound, so a download that keeps moving is never killed
// even when its total transfer time exceeds timeoutMs many times over.
test("a slow-but-progressing download survives a deadline its total time exceeds (learnings-q61)", async () => {
  const bytes = new Uint8Array(40).map((_, i) => i);
  const good = createHash("sha256").update(bytes).digest("hex");
  let trickle = null;
  const stream = new ReadableStream({
    start(controller) {
      let sent = 0;
      const tick = () => {
        if (sent >= bytes.length) {
          controller.close();
          return;
        }
        controller.enqueue(bytes.subarray(sent, sent + 5));
        sent += 5;
        trickle = setTimeout(tick, 40); // 5 bytes every 40ms: 160ms total for a 100ms deadline
      };
      tick();
    },
    cancel() {
      clearTimeout(trickle);
    },
  });
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(stream);
  try {
    const seen = [];
    const received = await fetchVerified("https://fixture.invalid/trickle.bin", {
      sha256: good,
      label: "trickle",
      timeoutMs: 100,
      onProgress: (n) => seen.push(n),
    });
    assert.deepEqual(Buffer.from(received), Buffer.from(bytes), "the whole slow body arrived");
    assert.ok(seen.at(-1) === 40 && seen.length >= 4, "progress was reported for every chunk");
  } finally {
    clearTimeout(trickle);
    globalThis.fetch = original;
  }
});

// learnings-qxtn: header arrival is progress too — fetch-policy.mjs re-arms the idle deadline
// when the response headers land. Without that reset, a hub that is slow to RESPOND but then
// streams healthily is aborted even though it never went idle: the exact regression class the
// chunk-reset test above cannot catch, because there the first chunk arrives before the
// original deadline expires.
test("slow response headers reset the idle deadline before the first body byte (learnings-qxtn)", async () => {
  const bytes = new Uint8Array(16).map((_, i) => i);
  const good = createHash("sha256").update(bytes).digest("hex");
  const timers = [];
  let handedSignal = null;
  const original = globalThis.fetch;
  globalThis.fetch = (_url, options) => {
    handedSignal = options.signal;
    return new Promise((resolve) => {
      // Headers land at ~120ms of the 200ms deadline armed at t=0: past halfway, so the
      // header-arrival reset is the only thing that keeps the download alive.
      timers.push(setTimeout(() => {
        resolve(new Response(new ReadableStream({
          start(controller) {
            // The mock honours the signal: if the deadline fires, the pending read rejects
            // instead of hanging (or silently passing an un-aborted stream).
            handedSignal?.addEventListener("abort", () => controller.error(handedSignal.reason));
            // First chunk at ~140ms AFTER the headers (t≈260) — past the ORIGINAL deadline,
            // within the re-armed one. Second chunk 80ms later, then close: the stream never
            // goes idle for 200ms, so total time (~340ms) exceeds timeoutMs and only a
            // working header-arrival reset lets it finish.
            timers.push(setTimeout(() => controller.enqueue(bytes.subarray(0, 8)), 140));
            timers.push(setTimeout(() => {
              controller.enqueue(bytes.subarray(8));
              controller.close();
            }, 220));
          },
        })));
      }, 120));
    });
  };
  try {
    const received = await fetchVerified("https://fixture.invalid/slow-headers.bin", {
      sha256: good,
      label: "slow-headers",
      timeoutMs: 200,
    });
    assert.deepEqual(Buffer.from(received), Buffer.from(bytes),
      "a download whose HEADERS took longer than half the idle deadline still completed: header arrival re-armed it");
  } finally {
    for (const t of timers) clearTimeout(t);
    globalThis.fetch = original;
  }
});

// learnings-q61: a download that goes quiet MID-STREAM must not claim "no data": the message
// names the bytes that had already arrived, so a stalled CDN is distinguishable from a dead one.
test("a download that stalls mid-stream fails naming the bytes it received (learnings-q61)", async () => {
  const original = globalThis.fetch;
  let handedSignal = null;
  globalThis.fetch = async (_url, options) => {
    handedSignal = options.signal;
    return new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(5));
        // ...and then silence. The mock honours the signal: the pending read rejects on abort.
        handedSignal?.addEventListener("abort", () => controller.error(handedSignal.reason));
      },
    }));
  };
  try {
    await assert.rejects(
      fetchVerified("https://fixture.invalid/stall.bin", { sha256: "0".repeat(64), label: "stall", timeoutMs: 50 }),
      (error) => error.name === "TimeoutError" &&
        /stalled: nothing further within 50ms after 5 bytes/.test(error.message),
    );
  } finally {
    globalThis.fetch = original;
  }
});

// learnings-604: loadKev now threads its timeoutMs to every artifact download, like loadLaya.
test("loadKev threads its timeoutMs to the artifact downloads (learnings-604)", async () => {
  const stalled = createServer((_req, res) => {
    res.writeHead(200, { "content-length": "100" });
    // and then nothing: the connection is up, the body never comes.
  });
  await new Promise((resolve) => stalled.listen(0, "127.0.0.1", resolve));
  try {
    const base = `http://127.0.0.1:${stalled.address().port}/`;
    const urls = Object.fromEntries(
      ["tokenizer.json", "config.json", "onnx/model_q4.onnx", "onnx/model_q4.onnx_data"].map((f) => [f, base + f]),
    );
    const t0 = Date.now();
    await assert.rejects(
      loadKev({ timeoutMs: 50, urls }),
      /no data within 50ms/,
      "the caller's deadline must reach the downloads instead of the hardcoded default",
    );
    assert.ok(Date.now() - t0 < 5000, "the stalled load failed fast instead of hanging");
  } finally {
    stalled.closeAllConnections?.();
    stalled.close();
  }
});
