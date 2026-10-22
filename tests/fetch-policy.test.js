// tests/fetch-policy.test.js — the request policy every artifact download must carry
// (learnings-902), plus the redirect check that runs before a byte is consumed.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  ARTIFACT_HOST_SUFFIXES,
  artifactFetchOptions,
  artifactHostAllowed,
  assertArtifactResponse,
} from "../site/decision-models/fetch-policy.mjs";
import { fetchVerified } from "../site/decision-models/kev-engine.js";
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

test("both artifact engines route their downloads through the policy", () => {
  for (const file of ["kev-engine.js", "laya-engine.js"]) {
    const source = readFileSync(new URL(`../site/decision-models/${file}`, import.meta.url), "utf8");
    assert.match(source, /import \{ artifactFetchOptions, assertArtifactResponse \} from "\.\/fetch-policy\.mjs";/, `${file} must import the policy`);
    const fetches = source.match(/await fetch\(/g) ?? [];
    const hardened = source.match(/await fetch\([^;]*artifactFetchOptions\(/g) ?? [];
    assert.equal(hardened.length, fetches.length, `${file}: every fetch must carry the artifact policy`);
    assert.equal((source.match(/assertArtifactResponse\(/g) ?? []).length, fetches.length, `${file}: every response must be validated`);
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
