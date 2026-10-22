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
  // The legitimate path: huggingface.co redirects to its CDN.
  assert.doesNotThrow(() => assertArtifactResponse({ url: "https://cdn-lfs-us-1.hf.co/x/y" }, "weights"));
  // A response with no url (a locally built one) is left to the sha256 check.
  assert.doesNotThrow(() => assertArtifactResponse({}, "weights"));
  assert.throws(() => assertArtifactResponse({ url: "https://evil.example/weights" }, "weights"),
    /redirect left the artifact hosts \(evil\.example\)/);
  // A locally built fixture never leaves the origin: it is judged by the sha256 pin alone.
  assert.doesNotThrow(() => assertArtifactResponse({ url: "data:application/octet-stream;base64,AAAA" }, "fixture"));
  assert.doesNotThrow(() => assertArtifactResponse({ url: "blob:https://example.test/1" }, "fixture"));
  assert.throws(() => assertArtifactResponse({ url: "http://huggingface.co/weights" }, "weights"),
    /redirect downgraded to http:/);
  assert.throws(() => assertArtifactResponse({ url: "not a url" }, "weights"), /unparseable final URL/);
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
