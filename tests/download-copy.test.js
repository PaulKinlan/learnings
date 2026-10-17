import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = (file) => readFileSync(new URL(`../site/decision-models/${file}`, import.meta.url), "utf8");

// Cheap companion to the browser gate: a cache-copy regression must fail the fast Node suite,
// not wait for someone to run a real browser or re-download 375–680 MB of weights.
test("decision-models copy does not promise a persistent weight cache (learnings-mje)", () => {
  for (const file of ["lab.html", "adaptation.html", "examples.html", "ui.js",
    "laya-engine.js", "kev-engine.js", "index.html", "on-device.html", "playground.html"]) {
    assert.doesNotMatch(source(file), /weights download once|once per browser|cached after that|cached by the browser afterwards/i, file);
  }
  for (const file of ["lab.html", "adaptation.html", "examples.html", "ui.js"]) {
    assert.match(source(file), /a reload may fetch it again/, `${file} must explain re-download risk`);
  }
});

test("playground load estimate agrees with its 680 MB Laya notice (learnings-1w2)", () => {
  assert.match(source("playground.js"), /checkpointNote:\s*"[^"]*~680 MB of verified downloads"/);
  assert.match(source("playground.html"), /Laya about <strong>680 MB<\/strong>/);
});
