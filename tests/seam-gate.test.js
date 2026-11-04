// learnings-dkd (tm-test-seam-ungated): the __LAYA_URLS/__KEV_URLS asset-override seams were
// read unconditionally on the published origin. SHA-256 verification stops tampered MODEL
// EXECUTION, but an off-loopback page script could still point the 375-680 MB downloads at
// arbitrary endpoints and stall the tab. Every seam is now honoured only where the harness can
// run — loopback, or outside a browser entirely — the same gate core.js always gave
// __MOCK_DECIDE__. Functional half below; drift half: a repo-wide scan that fails if ANY read
// of a URL seam under site/decision-models is not gated by onLoopback() on the same line.
import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { onLoopback } from "../site/decision-models/core.js";

test("onLoopback is true outside a browser and on loopback hosts only", () => {
  // Node harness: no location at all — the harness owns the realm, so seams work (this is how
  // the node-driven smoke modules keep functioning).
  assert.equal(typeof globalThis.location, "undefined", "sanity: node has no location");
  assert.equal(onLoopback(), true, "outside a browser the harness owns the realm");

  const saved = Object.getOwnPropertyDescriptor(globalThis, "location");
  try {
    for (const hostname of ["127.0.0.1", "localhost", "[::1]"]) {
      Object.defineProperty(globalThis, "location", { configurable: true, value: { hostname } });
      assert.equal(onLoopback(), true, `${hostname} is loopback: the harness serves there`);
    }
    for (const hostname of ["paul.kinland.me", "localhost.evil.test", "127.0.0.1.evil.test", ""]) {
      Object.defineProperty(globalThis, "location", { configurable: true, value: { hostname } });
      assert.equal(onLoopback(), false, `${hostname || "an empty hostname"} is not loopback: every seam is dead there`);
    }
  } finally {
    if (saved) Object.defineProperty(globalThis, "location", saved);
    else delete globalThis.location;
  }
});

// learnings-7jst: this scan used to pin ONE spelling — `globalThis.__(LAYA|KEV)_URLS` — so a
// `window.__LAYA_URLS`, `self.__KEV_URLS` or `window['__LAYA_URLS']` read passed it green. A seam
// read is the seam NAME, on any receiver or bare, so the walk below finds it wherever it appears
// in code. Two things are not reads: a seam name in a comment (core.js documents them) and one
// inside a plain string, which is how the approved accessor is spelled — seamValue("__KEV_URLS").
// A bracket KEY is a read again: window["__LAYA_URLS"].
const SEAM_IN_CODE = /^__(?:LAYA|KEV)_URLS\b/;
const SEAM_BRACKET_KEY = /^__(?:LAYA|KEV)_URLS$/;

// 1-based line numbers in `source` that read a seam, whatever the spelling.
function seamReadLines(source) {
  const hits = new Set();
  let line = 1;
  let lastCode = ""; // the last non-space character seen in code: `[` makes the next string a key
  let comment = null; // "line" or "block" while inside one
  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === "\n") {
      line += 1;
      if (comment === "line") comment = null;
      continue;
    }
    if (comment === "block") {
      if (ch === "*" && source[i + 1] === "/") { comment = null; i += 1; }
      continue;
    }
    if (comment === "line") continue;
    if (ch === "/" && source[i + 1] === "/") { comment = "line"; i += 1; continue; }
    if (ch === "/" && source[i + 1] === "*") { comment = "block"; i += 1; continue; }
    if (ch === '"' || ch === "'" || ch === "`") {
      // One whole literal: a bracket key if it names a seam, a string otherwise. Bracket keys are
      // plain literals, so no escape handling is needed to read their content.
      const close = source.indexOf(ch, i + 1);
      const literal = source.slice(i + 1, close === -1 ? source.length : close);
      if (SEAM_BRACKET_KEY.test(literal) && lastCode === "[") hits.add(line);
      line += literal.split("\n").length - 1; // template literals can span lines
      i = close === -1 ? source.length : close;
      continue;
    }
    if (ch === "_" && SEAM_IN_CODE.test(source.slice(i))) { hits.add(line); continue; }
    if (ch.trim()) lastCode = ch;
  }
  return [...hits];
}

// Every .js/.mjs under site/decision-models, subdirectories included: the drivers are flat today,
// but a future decision-models/<subdir>/*.mjs must not be invisible to the scan (learnings-7jst).
const driverFiles = (base, prefix = "") =>
  readdirSync(new URL(prefix, base), { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory()
      ? driverFiles(base, `${prefix}${entry.name}/`)
      : /\.(js|mjs)$/.test(entry.name) ? [`${prefix}${entry.name}`] : []);

test("every __LAYA_URLS/__KEV_URLS read under site/decision-models is gated by onLoopback()", () => {
  const dir = new URL("../site/decision-models/", import.meta.url);
  const reads = [];
  for (const file of driverFiles(dir)) {
    const source = readFileSync(new URL(file, dir), "utf8");
    const lines = source.split("\n");
    for (const line of seamReadLines(source)) {
      reads.push(`${file}:${line}`);
      assert.match(
        lines[line - 1],
        /onLoopback\(\)/,
        `${file}:${line} reads a URL seam without the loopback gate (learnings-dkd): off-loopback it must be dead`,
      );
    }
  }
  // The known readers: playground.js (2), ui.js (2), classifier-api.js (1), measure.mjs (1),
  // gate-smoke.mjs (1), kev-smoke.mjs (1). If a refactor moves or removes them, update this
  // floor deliberately — never by weakening the same-line gate assertion above.
  assert.ok(reads.length >= 8, `the scan saw only ${reads.length} seam reads (${reads.join(", ")}); expected the known drivers`);
});

test("the drift scan catches every spelling of a seam read, and no other name", () => {
  // learnings-7jst regression fixture: the bead's repro spelled the one seam three ways and the
  // old scan saw none of them.
  const spellings = [
    "const a = globalThis.__LAYA_URLS;", // the one spelling the old scan pinned
    "const b = window.__LAYA_URLS;", // a different receiver
    "const c = self.__KEV_URLS;", // the other seam name
    "const d = window['__LAYA_URLS'];", // single-quoted bracket key
    'const e = globalThis["__KEV_URLS"];', // double-quoted bracket key
    "const f = __LAYA_URLS;", // the bare global
    "const g = page.__KEV_URLS;", // any receiver, not just a global object
  ];
  assert.deepEqual(seamReadLines(spellings.join("\n")), spellings.map((_, index) => index + 1));

  const notReads = [
    "// window.__LAYA_URLS is a test seam", // a line comment
    "/* __LAYA_URLS/__KEV_URLS live in the drivers */", // a block comment
    'urls = seamValue("__KEV_URLS") ?? {},', // the approved accessor: a string, not a read
    'const note = "no window.__LAYA_URLS here";', // a seam name inside a message string
    "const other = window.__LAYA_URL;", // not a seam name
    "const loaded = window.__LAYA_URLS_LOADED;", // a longer identifier, not the seam
  ];
  assert.deepEqual(seamReadLines(notReads.join("\n")), []);
});

// ---- behavioural half (learnings-2b5r) --------------------------------------------------------
// Above: onLoopback() in isolation, and a scan that every seam READ is gated. Neither proves the
// GATE'S VALUE reaches the consumer — a refactor could keep `onLoopback() && globalThis.X` and
// return undefined, or thread the gate into a place the loader never reads. These drive real code.

const withLocation = async (hostname, fn) => {
  const saved = Object.getOwnPropertyDescriptor(globalThis, "location");
  try {
    if (hostname === undefined) delete globalThis.location;
    else Object.defineProperty(globalThis, "location", { configurable: true, value: { hostname } });
    return await fn();
  } finally {
    if (saved) Object.defineProperty(globalThis, "location", saved);
    else delete globalThis.location;
  }
};

test("core.js exposes one gate for every seam: honoured on loopback, null off it", async () => {
  const { seamValue } = await import("../site/decision-models/core.js");
  // BOTH names through the same three cases, not Kev alone (learnings-jw4d). seamValue is
  // name-parametric today (`globalThis[name]`), so a __KEV_URLS-only test passed green while the
  // structural comment below and the landing record claimed "both names" — and a per-name branch
  // added later would have been caught only for Kev. Each name carries its own override object, so
  // a gate that answered with the OTHER seam's value fails the deepEqual as well.
  const seams = [
    ["__KEV_URLS", { "tokenizer.json": "https://seam.invalid/tokenizer.json" }],
    ["__LAYA_URLS", { SHA256SUMS: "https://seam.invalid/SHA256SUMS" }],
  ];
  for (const [name, override] of seams) {
    const saved = Object.getOwnPropertyDescriptor(globalThis, name);
    try {
      globalThis[name] = override;
      for (const hostname of ["127.0.0.1", "localhost", "[::1]", undefined]) {
        await withLocation(hostname, () => {
          assert.deepEqual(seamValue(name), override, `${name}: ${hostname ?? "no location"} honours the seam`);
        });
      }
      for (const hostname of ["paul.kinland.me", "127.0.0.2", "localhost.evil.test", ""]) {
        await withLocation(hostname, () => {
          assert.equal(seamValue(name), null, `${name}: ${hostname || "an empty hostname"} gets no seam value`);
        });
      }
      delete globalThis[name];
      await withLocation("127.0.0.1", () => {
        assert.equal(seamValue(name), null, `${name}: an unset seam is null even on loopback, never undefined`);
      });
    } finally {
      if (saved) Object.defineProperty(globalThis, name, saved);
      else delete globalThis[name];
    }
  }
});

test("loadKev takes the Kev seam only on loopback, and the pinned URL everywhere else", async () => {
  // The consumer, driven for real: loadKev() with no `urls` argument, a stubbed fetch, and the
  // URL it asks for first. (No model runs: the stub throws on the first request.)
  const { loadKev } = await import("../site/decision-models/kev-engine.js");
  const { kevArtifactUrl } = await import("../site/decision-models/kev-engine.js");
  const override = { "tokenizer.json": "https://seam.invalid/tokenizer.json" };
  const savedFetch = globalThis.fetch;
  const savedSeam = Object.getOwnPropertyDescriptor(globalThis, "__KEV_URLS");
  const firstRequest = async () => {
    const seen = [];
    globalThis.fetch = async (url) => {
      seen.push(String(url));
      throw new Error("stub: no artefacts are downloaded in a test");
    };
    await loadKev({}).catch(() => {});
    assert.ok(seen.length, "loadKev made a request, so the URL it chose is observable");
    return seen[0];
  };
  try {
    globalThis.__KEV_URLS = override;
    const offLoopback = await withLocation("paul.kinland.me", firstRequest);
    assert.equal(offLoopback, kevArtifactUrl("tokenizer.json"),
      "off loopback the seam is ignored: the pinned hub URL is requested");
    assert.equal(offLoopback.includes("seam.invalid"), false, "no request leaves for the injected endpoint");

    const onLoopback = await withLocation("127.0.0.1", firstRequest);
    assert.equal(onLoopback, override["tokenizer.json"],
      "on loopback the same call honours the seam (the control: the seam still works)");
  } finally {
    globalThis.fetch = savedFetch;
    if (savedSeam) Object.defineProperty(globalThis, "__KEV_URLS", savedSeam);
    else delete globalThis.__KEV_URLS;
  }
});

test("both engines resolve their urls DEFAULT through the one gate; LitERT's half is structural", () => {
  // loadKev is driven end to end above. loadLaya cannot be: LitERT boots first and cannot boot
  // outside a DOM, so no URL is ever chosen in node and any "it rejected" assertion would be a
  // fake test. The honest statement of Laya's coverage is this structural pair — the default is
  // the gated seam, and the gated seam itself is behaviourally tested above for BOTH names (the
  // seamValue loop drives __KEV_URLS and __LAYA_URLS through the same honoured/ignored/unset
  // cases, learnings-jw4d) — so the reader can see exactly which half is behavioural and which
  // is not: what is NOT driven for Laya is loadLaya's own default resolution.
  const dir = new URL("../site/decision-models/", import.meta.url);
  for (const [file, seam] of [["laya-engine.js", "__LAYA_URLS"], ["kev-engine.js", "__KEV_URLS"]]) {
    const source = readFileSync(new URL(file, dir), "utf8");
    assert.match(source, new RegExp(`urls = seamValue\\("${seam}"\\) \\?\\? \\{\\}`),
      `${file} must resolve its urls default through seamValue (learnings-2b5r)`);
  }
});
