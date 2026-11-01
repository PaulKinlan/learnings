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

test("every __LAYA_URLS/__KEV_URLS read under site/decision-models is gated by onLoopback()", () => {
  const dir = new URL("../site/decision-models/", import.meta.url);
  const files = readdirSync(dir).filter((f) => /\.(js|mjs)$/.test(f));
  const reads = [];
  for (const file of files) {
    const lines = readFileSync(new URL(file, dir), "utf8").split("\n");
    for (const [i, line] of lines.entries()) {
      if (!/globalThis\.__(LAYA|KEV)_URLS/.test(line)) continue;
      reads.push(`${file}:${i + 1}`);
      assert.match(
        line,
        /onLoopback\(\)/,
        `${file}:${i + 1} reads a URL seam without the loopback gate (learnings-dkd): off-loopback it must be dead`,
      );
    }
  }
  // The known readers: playground.js (2), ui.js (2), classifier-api.js (1), measure.mjs (1),
  // gate-smoke.mjs (1), kev-smoke.mjs (1). If a refactor moves or removes them, update this
  // floor deliberately — never by weakening the same-line gate assertion above.
  assert.ok(reads.length >= 8, `the scan saw only ${reads.length} seam reads (${reads.join(", ")}); expected the known drivers`);
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
  const override = { "tokenizer.json": "https://seam.invalid/tokenizer.json" };
  const saved = Object.getOwnPropertyDescriptor(globalThis, "__KEV_URLS");
  try {
    globalThis.__KEV_URLS = override;
    for (const hostname of ["127.0.0.1", "localhost", "[::1]", undefined]) {
      await withLocation(hostname, () => {
        assert.deepEqual(seamValue("__KEV_URLS"), override, `${hostname ?? "no location"} honours the seam`);
      });
    }
    for (const hostname of ["paul.kinland.me", "127.0.0.2", "localhost.evil.test", ""]) {
      await withLocation(hostname, () => {
        assert.equal(seamValue("__KEV_URLS"), null, `${hostname || "an empty hostname"} gets no seam value`);
      });
    }
    delete globalThis.__KEV_URLS;
    await withLocation("127.0.0.1", () => {
      assert.equal(seamValue("__KEV_URLS"), null, "an unset seam is null even on loopback, never undefined");
    });
  } finally {
    if (saved) Object.defineProperty(globalThis, "__KEV_URLS", saved);
    else delete globalThis.__KEV_URLS;
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
  // the gated seam, and the gated seam itself is behaviourally tested above for BOTH names — so
  // the reader can see exactly which half is behavioural and which is not.
  const dir = new URL("../site/decision-models/", import.meta.url);
  for (const [file, seam] of [["laya-engine.js", "__LAYA_URLS"], ["kev-engine.js", "__KEV_URLS"]]) {
    const source = readFileSync(new URL(file, dir), "utf8");
    assert.match(source, new RegExp(`urls = seamValue\\("${seam}"\\) \\?\\? \\{\\}`),
      `${file} must resolve its urls default through seamValue (learnings-2b5r)`);
  }
});
