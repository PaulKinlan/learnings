// Drive the on-device page in a real browser.
//
// The claim under test is not "the page renders". It is that a decision model runs in this tab,
// on this machine, with nothing sent anywhere, and that the numbers it prints came from the
// reader's own run rather than from a table in the source.
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { launch } from "./lib/cdp.mjs";
import { serve } from "../scripts/serve.mjs";

const local = process.env.TEST_URL ? null : await serve();
const base = process.env.TEST_URL ?? local.url;
const out = resolve(process.env.EVIDENCE_DIR ?? "research");
await mkdir(out, { recursive: true });

const page = await launch({ width: 1280, height: 1000 });
const checks = [];
const check = (name, condition, detail = "") => {
  assert.ok(condition, `${name}${detail ? ` — ${detail}` : ""}`);
  checks.push(name);
};

// What the page fetched, read from the page's own resource timeline. Using the page's record
// rather than a driver-side listener means this cannot pass by the driver failing to subscribe.
const fetchedHosts = () =>
  page.evaluate(() =>
    [...new Set(performance.getEntriesByType("resource").map((e) => new URL(e.name).host))].sort(),
  );
const fetchedFromHub = () =>
  page.evaluate(() =>
    performance.getEntriesByType("resource").filter((e) => /huggingface\.co|hf\.co/.test(e.name)).length,
  );

try {
  await page.goto(`${base}decision-models/on-device.html`, { timeout: 30000 });

  // 1. The model must not load on page load. This is the check that would fail if anyone ever
  //    makes the download automatic, which is the change that would cost a visitor 375 MB.
  await new Promise((r) => setTimeout(r, 2500));
  const beforeHosts = await fetchedHosts();
  const beforeHub = await fetchedFromHub();
  check(
    "nothing is downloaded before the reader asks",
    beforeHub === 0 && (await page.evaluate(() => document.querySelector("#workbench").hidden)),
    `hosts already contacted: ${beforeHosts.join(", ")}; hub fetches: ${beforeHub}`,
  );
  check("the page states the download size before it starts", await page.evaluate(() =>
    document.querySelector("main").textContent.includes("375 MB"),
  ));

  // 2. Load. This is the long step: ~375 MB of weights plus runtime initialisation.
  await page.click("#load");
  const loaded = await page
    .waitFor(() => !document.querySelector("#workbench").hidden, { timeout: 900000, label: "the model to load" })
    .then(() => true)
    .catch(() => false);

  if (!loaded) {
    const state = await page.evaluate(() => ({
      progress: document.querySelector("#progress").textContent,
      loadState: document.querySelector("#load-state").textContent,
      log: document.querySelector("#log").textContent,
    }));
    throw new Error(`the model did not load: ${JSON.stringify(state, null, 2)}`);
  }

  const loadedIn = await page.evaluate(() => document.querySelector("#progress").textContent);
  check("the load reports how long it took", /Ready in [\d.]+s/.test(loadedIn), loadedIn);
  const hubFetches = await fetchedFromHub();
  const hosts = await fetchedHosts();
  check("weights were fetched from the Hub", hubFetches > 0, `${hubFetches} hub fetches`);
  check(
    "no third-party CDN was contacted",
    !hosts.some((h) => /jsdelivr|unpkg|cdnjs|cdn-lsf/.test(h)),
    `hosts: ${hosts.join(", ")}`,
  );

  const backend = await page.evaluate(() => document.querySelector("#backend").textContent);
  check("the backend actually used is reported, not assumed", /Session reports:|did not report/.test(backend), backend);

  // 3. The exercises. A uniform distribution is the failure signature of a broken readout, so
  //    uniformity is asserted against explicitly rather than being read as a confident tie.
  const summary = await page.evaluate(() => document.querySelector("#summary").textContent);
  const flatBoxes = await page.evaluate(
    () => [...document.querySelectorAll("#exercises .answer")].filter((b) => b.textContent.includes("identical")).length,
  );
  check("no exercise produced a flat score vector", flatBoxes === 0, `${flatBoxes} flat`);
  const matched = Number((summary.match(/(\d+) of (\d+)/) || [])[1]);
  const total = Number((summary.match(/(\d+) of (\d+)/) || [])[2]);
  check("every exercise ran and reported a count", Number.isInteger(matched) && total > 0, summary);

  const latency = (summary.match(/fastest (\d+) ms, median (\d+) ms, slowest (\d+) ms/) || []).slice(1).map(Number);
  check("latency is measured in this tab", latency.length === 3 && latency[0] > 0, summary);

  // 4. The reversed-options control: the answer must follow the label, not the position.
  const reversal = await page.evaluate(() => {
    const boxes = [...document.querySelectorAll("#exercises .answer")];
    const box = boxes.find((b) => b.textContent.includes("options in reverse order"));
    if (!box) return null;
    const labels = [...box.querySelectorAll(".bar")].map((row) => ({
      label: row.querySelector("span").textContent,
      value: Number(row.querySelector("output").textContent.replace("%", "")),
    }));
    const best = labels.slice().sort((a, b) => b.value - a.value)[0];
    return { best: best.label, value: best.value, position: labels.findIndex((l) => l.label === best.label) };
  });
  check(
    "with the options reversed the answer follows the label, not the position",
    reversal && reversal.best.startsWith("billing") && reversal.position > 0,
    JSON.stringify(reversal),
  );

  // 5. One call, many questions.
  await page.click("#batch");
  await page.waitFor(() => document.querySelector("#exercises h2")?.textContent.includes("one forward pass"), {
    timeout: 120000,
    label: "the batch run",
  });
  const batch = await page.evaluate(() => document.querySelector("#exercises .answer p").textContent);
  check("four questions are answered in one forward pass", /4 questions were packed into a single \d+-token sequence/.test(batch), batch);

  // 6. Reader-supplied questions, which is the point of the page.
  await page.evaluate(() => {
    document.querySelector("#custom-state").value = "The invoice shows the wrong VAT number and must be fixed before month end.";
  });
  await page.click("#custom");
  await page.waitFor(() => document.querySelector("#exercises h4") !== null, { timeout: 120000, label: "the custom run" });
  const custom = await page.evaluate(() => ({
    questions: [...document.querySelectorAll("#exercises h4")].map((h) => h.textContent),
    reading: [...document.querySelectorAll("#exercises .answer p")].map((p) => p.textContent),
  }));
  check("a reader can supply their own state and questions", custom.questions.length === 3, JSON.stringify(custom.questions));
  check("the custom run explains how to read each answer", custom.reading.some((t) => /Read as (noul|choice)/.test(t)));

  const evidence = {
    what: "on-device page, driven in Chrome",
    when: new Date().toISOString(),
    url: `${base}decision-models/on-device.html`,
    loadedIn,
    backend,
    summary,
    exercisesMatched: matched,
    exercisesTotal: total,
    latencyMs: { fastest: latency[0], median: latency[1], slowest: latency[2] },
    flatScoreVectors: flatBoxes,
    reversedOptionControl: reversal,
    batch,
    hostsContacted: hosts,
    hubFetches,
    checks,
  };
  await writeFile(resolve(out, "on-device-measurements.json"), JSON.stringify(evidence, null, 2));

  console.log(checks.map((c) => `PASS  ${c}`).join("\n"));
  console.log("");
  console.log(`  loaded in ${loadedIn.replace("Ready in ", "")}`);
  console.log(`  ${summary}`);
  console.log(`  backend: ${backend}`);
  console.log(`  reversed-options control: chose "${reversal?.best}" at position ${reversal?.position} (0-based)`);
  console.log(`  ${batch}`);
  console.log(`  Hub fetches: ${hubFetches} · hosts contacted: ${hosts.join(", ")}`);
  console.log(`  checks: ${checks.length} passed`);
} finally {
  await page.close?.();
  local?.server.close();
}
