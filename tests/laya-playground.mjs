// Opt-in, heavy real-browser acceptance for the Laya playground: downloads ~650 MB of pinned
// Laya artifacts from the hub and compiles the graphs in a real Chromium. NOT part of `npm
// test` or the fast gate — run it explicitly when the playground, classifier sharing or the
// Laya load path changes:
//
//   npm run test:laya-playground
//
// TEST_URL overrides the served base (an already-running `npm run serve`), and EVIDENCE_DIR
// controls where the receipt (laya-playground.json) and screenshot land (default: research/).
// An unrun harness looks like coverage; this one is a named, documented opt-in instead
// (learnings-v92).
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { launch } from "./lib/cdp.mjs";
import { serve } from "../scripts/serve.mjs";

const local = process.env.TEST_URL ? null : await serve();
const base = process.env.TEST_URL ?? local.url;
const out = resolve(process.env.EVIDENCE_DIR ?? "research");
await mkdir(out, { recursive: true });
let page;
try {
  page = await launch({ width: 1280, height: 1000 });
  await page.goto(`${base}decision-models/playground.html`);
  assert.equal(await page.evaluate(() => document.querySelector("#demos").hidden), true);
  const before = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => /huggingface\.co|hf\.co/.test(entry.name)).length);
  assert.equal(before, 0, "no model weights before explicit load");
  await page.click("#load");
  await page.waitFor(() => !document.querySelector("#load").disabled ||
    document.querySelector("#progress").textContent.startsWith("Error"),
  { timeout: 900000, label: "Laya load or error" });
  const progress = await page.evaluate(() => document.querySelector("#progress").textContent);
  assert.equal(await page.evaluate(() => document.querySelector("#demos").hidden), false, progress);

  const demos = [];
  for (const id of ["tool-selection", "agent-routing", "state-machine", "moderation"]) {
    await page.click(`[data-demo="${id}"] .run-demo`);
    await page.waitFor((name) => {
      const section = document.querySelector(`[data-demo="${name}"]`);
      return !section.querySelector(".run-demo").disabled || section.querySelector(".results").textContent.includes("Error");
    }, { timeout: 120000, label: `${id} inference`, args: [id] });
    const result = await page.evaluate((name) => {
      const section = document.querySelector(`[data-demo="${name}"]`);
      return { id: name, text: section.querySelector(".results").textContent,
        answers: section.querySelectorAll(".results .answer").length };
    }, id);
    assert.ok(result.answers > 0 && /\d+\.\d+%/.test(result.text) && !/NaN|Run failed:/.test(result.text),
      `${id}: ${result.text}`);
    demos.push(result);
  }
  const beforeClassifier = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => /huggingface\.co|hf\.co/.test(entry.name)).length);
  await page.click("#run-classifier");
  await page.waitFor(() => !document.querySelector("#run-classifier").disabled ||
    document.querySelector("#classifier-results").textContent.includes("Classifier error:"),
  { timeout: 120000, label: "Classifier inference" });
  const classifier = await page.evaluate(() => ({
    text: document.querySelector("#classifier-results").textContent,
    meters: [...document.querySelectorAll("#classifier-results meter")].map((m) => m.value),
    hubRequests: performance.getEntriesByType("resource")
      .filter((entry) => /huggingface\.co|hf\.co/.test(entry.name)).length,
  }));
  assert.ok(classifier.meters.length > 0 && classifier.meters.every(Number.isFinite), classifier.text);
  assert.equal(classifier.hubRequests, beforeClassifier, "Classifier reuses Laya; no second weight fetch");
  // The exact same object must back both entry points, not just rely on HTTP cache behavior.
  const sameSession = await page.evaluate(async () => {
    const { loadSharedLaya, Classifier } = await import("./classifier-api.js");
    const shared = await loadSharedLaya();
    const wrapper = await Classifier.create({ questions: [{ id: "check", type: "binary", prompt: "Relevant?" }] });
    const same = wrapper._session === shared;
    wrapper.destroy();
    return same;
  });
  assert.equal(sameSession, true, "Classifier and playground share the compiled Laya session");
  await page.screenshot(resolve(out, "laya-playground.png"), { fullPage: true });
  const evidence = { at: new Date().toISOString(), url: `${base}decision-models/playground.html`,
    progress, demos, classifier, sameSession, beforeClassifier, screenshot: "laya-playground.png" };
  await writeFile(resolve(out, "laya-playground.json"), JSON.stringify(evidence, null, 2));
  console.log(`PASS real Laya: ${demos.length} demos, Classifier; ${beforeClassifier} initial Hub requests, ${classifier.hubRequests} after Classifier`);
} finally {
  await page?.close();
  local?.server.close();
}
