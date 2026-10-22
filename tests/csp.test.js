// tests/csp.test.js — INV-CSP, enforced repo-wide instead of per page/suite.
//
// Every HTML entry point under site/ ships a meta Content-Security-Policy. The per-page and
// per-chapter suites only check the pages they already know about, so a new site/*.html could
// ship without one and nothing would fail (learnings-5h6). This test walks the tree, so a new
// page is covered the moment it exists.
//
// The invariants themselves are the repository's: default-src 'none' first, an explicit
// non-wildcard connect-src allowlist, and no script source that would defeat 'self' (a bare *,
// 'unsafe-inline' or 'unsafe-eval'; 'wasm-unsafe-eval' is deliberate and allowed).
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const site = fileURLToPath(new URL("../site/", import.meta.url));

/** Every .html under site/, minus the vendored third-party runtimes. */
function htmlFiles(dir = site) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === "vendor" ? [] : htmlFiles(path);
    return path.endsWith(".html") ? [path] : [];
  }).sort();
}

/** Attribute order is not guaranteed, so read attributes rather than matching one spelling. */
function cspPolicy(html) {
  const metas = [...html.matchAll(/<meta\b[^>]*>/gi)]
    .map((m) => m[0])
    .filter((tag) => /\bhttp-equiv\s*=\s*"content-security-policy"/i.test(tag));
  assert.equal(metas.length, 1, "exactly one CSP meta tag");
  const content = metas[0].match(/\bcontent\s*=\s*"([^"]*)"/i);
  assert.ok(content, "the CSP meta tag has a content attribute");
  return content[1];
}

/** "default-src 'none'; script-src 'self'" -> [{name, sources}] in policy order. */
function directives(policy) {
  return policy
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [name, ...sources] = part.split(/\s+/);
      return { name: name.toLowerCase(), sources };
    });
}

const lookup = (list, name) => list.find((directive) => directive.name === name);

// A source that opens the policy beyond an explicit list: the bare wildcard, a whole scheme,
// or a wildcard host. A port wildcard (http://127.0.0.1:*) is explicit enough to keep.
//
// Two wildcard HOSTS are deliberate and reviewed: model weights are fetched from the
// HuggingFace CDN, which rotates across subdomains (cdn-lfs.huggingface.co,
// cdn-lfs-us-1.hf.co, ...), so pinning one host would break loading a checkpoint. Because the
// exception is a closed set, a new page still cannot add its own wildcard.
const REVIEWED_WILDCARD_HOSTS = new Set(["https://*.huggingface.co", "https://*.hf.co"]);
const wildcardSource = (source) =>
  !REVIEWED_WILDCARD_HOSTS.has(source) &&
  (source === "*" ||
    /^[a-z][a-z0-9+.-]*:$/i.test(source) ||
    /^[a-z][a-z0-9+.-]*:\/\/\*(\/|:|$)/i.test(source) ||
    /^[a-z][a-z0-9+.-]*:\/\/\*\./i.test(source));

for (const file of htmlFiles()) {
  const rel = relative(site, file);
  const html = readFileSync(file, "utf8");
  const policy = cspPolicy(html);
  const list = directives(policy);

  test(`${rel}: CSP starts with default-src 'none'`, () => {
    assert.equal(list[0]?.name, "default-src", "default-src must be the first directive");
    assert.deepEqual(list[0].sources, ["'none'"], "default-src must deny everything not named later");
  });

  test(`${rel}: CSP has an explicit non-wildcard connect-src allowlist`, () => {
    const connect = lookup(list, "connect-src");
    assert.ok(connect, "connect-src must be listed, not inherited from default-src 'none'");
    assert.ok(connect.sources.length > 0, "connect-src must name at least one source");
    const wild = connect.sources.filter(wildcardSource);
    assert.deepEqual(wild, [], `connect-src must not widen to ${wild.join(", ")}`);
  });

  test(`${rel}: CSP script-src keeps 'self' meaningful`, () => {
    const script = lookup(list, "script-src");
    assert.ok(script, "script-src must be listed explicitly");
    for (const forbidden of ["*", "'unsafe-inline'", "'unsafe-eval'"])
      assert.ok(!script.sources.includes(forbidden), `script-src must not contain ${forbidden}`);
  });
}

// An inline script is only allowed when script-src hash-pins it: the hash must be of the exact
// bytes the browser will verify, so a later edit to the script cannot silently keep the page
// working under a stale hash.
test("every inline script is hash-pinned by the page's own CSP", () => {
  for (const file of htmlFiles()) {
    const rel = relative(site, file);
    const html = readFileSync(file, "utf8");
    const inline = [...html.matchAll(/<script\b((?![^>]*\bsrc=)[^>]*)>([\s\S]*?)<\/script>/gi)];
    if (!inline.length) continue;
    const script = lookup(directives(cspPolicy(html)), "script-src");
    const hashes = new Set(script.sources.filter((source) => source.startsWith("'sha256-")).map((source) => source.slice(1, -1)));
    for (const [, , body] of inline) {
      const digest = `sha256-${createHash("sha256").update(body, "utf8").digest("base64")}`;
      assert.ok(hashes.has(digest), `${rel}: inline script needs ${digest} in script-src`);
    }
  }
});

test("the only wildcard hosts in the repository are the reviewed HuggingFace CDN pair", () => {
  const seen = new Set();
  for (const file of htmlFiles()) {
    const connect = lookup(directives(cspPolicy(readFileSync(file, "utf8"))), "connect-src");
    for (const source of connect?.sources ?? [])
      if (REVIEWED_WILDCARD_HOSTS.has(source)) seen.add(source);
  }
  assert.deepEqual([...seen].sort(), [...REVIEWED_WILDCARD_HOSTS].sort(),
    "the reviewed exception must stay the exception, not become dead code");
});

test("the walk really covers every page (guards against an empty or vendor-only run)", () => {
  const files = htmlFiles();
  assert.ok(files.length >= 30, `expected the whole site, found ${files.length} pages`);
  assert.ok(!files.some((file) => file.includes(`${site}vendor`)), "site/vendor must stay out of scope");
  assert.ok(files.some((file) => file.endsWith("decision-models/lab.html")), "a known entry point must be present");
});
