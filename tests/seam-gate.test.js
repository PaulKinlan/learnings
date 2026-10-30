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
