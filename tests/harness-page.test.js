// tests/harness-page.test.js — the harness deep dive and its playground.
//
// Two halves, deliberately in one file:
//   1. the simulation's arithmetic, asserted directly against sim.js, so a label that stops
//      tracking its control fails without a browser;
//   2. a real browser driving every control on the page, because the requirement is that each
//      control visibly changes the trace AND the numbers — a property of the rendered page, not
//      of the module. One browser, closed in a finally, every wait bounded.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEMO_MODELS,
  THINKING_LEVELS,
  TOOLS,
  defaultAllowlist,
  compactMessages,
  simulateRun,
} from "../site/harness/sim.js";

const pagePath = fileURLToPath(new URL("../site/harness/index.html", import.meta.url));
const appPath = fileURLToPath(new URL("../site/harness/app.js", import.meta.url));
const readPage = () => readFileSync(pagePath, "utf8");
const readApp = () => readFileSync(appPath, "utf8");

const kindCount = (run, kind) => run.events.filter((event) => event.kind === kind).length;
const LITE = DEMO_MODELS[0];
const PRO = DEMO_MODELS[1];

// ---------------------------------------------------------------------------------------------
// The simulation's arithmetic
// ---------------------------------------------------------------------------------------------

test("the compaction line is contextWindow - reserveTokens for every reserve, not a constant", () => {
  for (const model of DEMO_MODELS) {
    for (const reserve of [0, 512, 8192, model.contextWindow - 1024]) {
      const run = simulateRun({ model: model.id, reserveTokens: reserve });
      assert.equal(run.threshold, model.contextWindow - reserve, `${model.id} at reserve ${reserve}`);
      for (const event of run.events.filter((entry) => entry.kind === "request")) {
        assert.equal(event.threshold, run.threshold, "every request shows the same derived line");
      }
    }
  }
});

test("compaction fires only above the derived line, and raising reserveTokens makes it fire", () => {
  const low = simulateRun({ model: PRO.id, reserveTokens: 8192 });
  const high = simulateRun({ model: PRO.id, reserveTokens: PRO.contextWindow - 4096 });
  assert.equal(high.threshold, 4096);
  assert.equal(kindCount(low, "compaction"), 0, "a 32k window with an 8k reserve never crosses its line");
  assert.ok(kindCount(high, "compaction") > 0, "a 4k reserve on the same transcript crosses it");
  for (const event of high.events.filter((entry) => entry.kind === "compaction")) {
    assert.ok(event.before > high.threshold, "compaction runs only once the context is past the line");
    assert.ok(event.after < event.before, "compaction has to actually shrink the transcript");
  }
});

test("keepRecentTokens decides what survives a compaction, and a huge value makes it a no-op", () => {
  const base = { model: LITE.id, reserveTokens: 8192 };
  const tight = simulateRun({ ...base, keepRecentTokens: 0 });
  const loose = simulateRun({ ...base, keepRecentTokens: LITE.contextWindow });
  assert.ok(kindCount(tight, "compaction") > 0, "the tight case still compacts");
  assert.equal(kindCount(loose, "compaction"), 0, "nothing is outside a window that covers everything");
  const skipped = loose.events.filter((entry) => entry.kind === "compaction-skipped");
  assert.ok(skipped.length > 0, "the no-op is reported rather than silently ignored");
  assert.match(skipped[0].reason, /keepRecentTokens/, "the reason names the setting that caused it");
  // A reserve so large that the line sits under the system prompt is unfixable by compaction, and
  // the reason says so instead of blaming the keep window.
  const impossible = simulateRun({ model: LITE.id, reserveTokens: LITE.contextWindow - 1024 });
  assert.ok(impossible.threshold < impossible.metrics.promptTokens, "the line is under the prompt itself");
  assert.match(
    impossible.events.find((entry) => entry.kind === "compaction-skipped").reason,
    /system prompt and tool schemas alone/,
  );
  assert.notEqual(
    tight.metrics.billedInputTokens,
    loose.metrics.billedInputTokens,
    "the kept tail changes what later requests re-send",
  );
  // The kept window is counted from the end: a bigger keep retains more messages.
  const small = compactMessages([{ tokens: 100 }, { tokens: 200 }, { tokens: 300 }, { tokens: 400 }], 700);
  assert.deepEqual(small.recent.map((m) => m.tokens), [300, 400]);
  assert.deepEqual(small.older.map((m) => m.tokens), [100, 200]);
  assert.equal(small.summaryTokens, Math.round(400 + 0.15 * 300), "the summary size comes from what it read");
});

test("sequential tool calls cost more requests and more re-sent context than one batch", () => {
  const parallel = simulateRun({ batchMode: "parallel" });
  const sequential = simulateRun({ batchMode: "sequential" });
  assert.ok(sequential.metrics.requests > parallel.metrics.requests, "one call per response means more requests");
  assert.ok(
    sequential.metrics.billedInputTokens > parallel.metrics.billedInputTokens,
    "each extra request re-sends the transcript",
  );
  assert.equal(sequential.metrics.toolCalls, parallel.metrics.toolCalls, "the same calls happen either way");
  assert.equal(kindCount(sequential, "request"), sequential.metrics.requests);
});

test("the tool allowlist changes the prompt, the refusal count, and the run's outcome", () => {
  assert.deepEqual(defaultAllowlist(), TOOLS.filter((tool) => tool.defaultOn).map((tool) => tool.id));
  const withGrep = simulateRun({ tools: [...defaultAllowlist(), "grep"] });
  const withoutGrep = simulateRun({ tools: defaultAllowlist() });
  assert.equal(
    withGrep.metrics.promptTokens - withoutGrep.metrics.promptTokens,
    TOOLS.find((tool) => tool.id === "grep").promptTokens,
    "adding a tool adds its schema to the prompt",
  );
  assert.equal(withGrep.metrics.rejectedCalls, 0, "with grep allowed, nothing the script calls is refused");
  assert.equal(withoutGrep.metrics.rejectedCalls, 1);
  assert.equal(withoutGrep.status, "settled", "a refused read-class tool does not block the run");

  const noWrite = simulateRun({ tools: ["read", "bash", "grep"] });
  assert.equal(noWrite.status, "blocked", "a refused write-class call takes the blocked branch");
  assert.ok(kindCount(noWrite, "run-end") > 0, "even a blocked run reaches agent_end");
  assert.ok(
    kindCount(noWrite, "request") < kindCount(withoutGrep, "request"),
    "the blocked branch stops earlier than a run that kept its write tools",
  );
});

test("a 429 is absorbed only when auto-retry is on, and it costs a request but no tokens", () => {
  const baseline = simulateRun({ inject429: "none" });
  const retried = simulateRun({ inject429: "request-2", autoRetry: true });
  assert.equal(retried.metrics.retries, 1);
  assert.equal(retried.metrics.requests, baseline.metrics.requests + 1);
  assert.equal(
    retried.metrics.billedInputTokens,
    baseline.metrics.billedInputTokens,
    "a refused request consumed no tokens, so the bill matches the clean run",
  );
  assert.equal(retried.status, "settled");
  const refused = retried.events.find((event) => event.kind === "request" && event.status === "429");
  assert.equal(refused.billed, 0);

  const noRetry = simulateRun({ inject429: "request-2", autoRetry: false });
  assert.equal(noRetry.status, "failed");
  assert.equal(kindCount(noRetry, "run-end"), 0, "a failed run never reaches agent_end");
  assert.ok(noRetry.metrics.requests < baseline.metrics.requests, "everything after the failure never happened");
});

test("steering lands inside the run; a follow-up starts a second run after the agent settles", () => {
  const baseline = simulateRun({ prompt: "none" });
  const steered = simulateRun({ prompt: "steer" });
  const followUp = simulateRun({ prompt: "follow-up" });

  const steerEvent = steered.events.findIndex((event) => event.kind === "prompt" && event.role === "steer");
  const firstEnd = steered.events.findIndex((event) => event.kind === "run-end");
  assert.ok(steerEvent > 0 && steerEvent < firstEnd, "a steer is delivered inside the run that was running");
  assert.ok(steered.metrics.billedInputTokens > baseline.metrics.billedInputTokens, "and it is re-sent afterwards");
  assert.equal(steered.metrics.runs, 1);

  const settled = followUp.events.findIndex((event) => event.kind === "settled");
  const runStart = followUp.events.findIndex((event) => event.kind === "run-start");
  assert.ok(settled > 0 && runStart > settled, "a follow-up starts only after agent_settled");
  assert.equal(followUp.metrics.runs, 2);
  assert.equal(followUp.metrics.requests, baseline.metrics.requests + 1);
  assert.ok(followUp.metrics.billedInputTokens > baseline.metrics.billedInputTokens);
});

test("thinking level buys reasoning tokens, which are re-sent on every later request", () => {
  const bills = THINKING_LEVELS.map((level) => simulateRun({ thinking: level.id, model: PRO.id }));
  for (let i = 1; i < bills.length; i += 1) {
    assert.ok(bills[i].metrics.reasoningTokens > bills[i - 1].metrics.reasoningTokens, "more thinking, more reasoning tokens");
    assert.ok(
      bills[i].metrics.billedInputTokens > bills[i - 1].metrics.billedInputTokens,
      "reasoning content comes back on every later request, so the input bill follows",
    );
  }
  assert.equal(bills[0].metrics.reasoningTokens, 0);
});

test("a run is deterministic: the same controls produce the same trace", () => {
  const controls = { model: PRO.id, thinking: "high", reserveTokens: 4096, keepRecentTokens: 1024 };
  assert.deepEqual(simulateRun(controls), simulateRun(controls));
});

// ---------------------------------------------------------------------------------------------
// The page: CSP, structure, controls that exist, and no hardcoded numbers
// ---------------------------------------------------------------------------------------------

function localTargets(html) {
  const targets = [];
  for (const match of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    if (/^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i.test(match[1])) continue;
    targets.push(match[1]);
  }
  return targets;
}

test("every local link on the page resolves", () => {
  const page = readPage();
  const targets = localTargets(page);
  assert.ok(targets.length >= 8, `expected the nav, stylesheets and module script, found ${targets.length}`);
  for (const target of ["../style.css", "./harness.css", "./app.js", "../index.html", "../decision-models/"]) {
    assert.ok(targets.includes(target), `the page links ${target}`);
  }
  for (const target of targets) {
    const path = target.split("#")[0].split("?")[0];
    if (!path) continue;
    let resolved = resolve(dirname(pagePath), path);
    if (existsSync(resolved) && statSync(resolved).isDirectory()) resolved = join(resolved, "index.html");
    assert.ok(existsSync(resolved), `${target} does not resolve to a file`);
  }
  for (const anchor of page.matchAll(/href="#([^"]+)"/g)) {
    assert.match(page, new RegExp(`id="${anchor[1]}"`), `#${anchor[1]} is an anchor on this page`);
  }
});

test("the page's CSP is intact, pins its inline speculation rules, and forbids inline styles", () => {
  const page = readPage();
  const csp = page.match(/http-equiv="Content-Security-Policy" content="([^"]+)"/)[1];
  assert.match(csp, /^default-src 'none';/, "default-src 'none' comes first");
  assert.match(csp, /script-src 'self'/, "scripts stay same-origin");
  assert.doesNotMatch(csp, /'unsafe-inline'|'unsafe-eval'/, "no eval, no inline script source");
  assert.match(csp, /connect-src 'self'/, "the page may only talk to its own origin");
  assert.match(page, /class="skip" href="#main"/, "the skip link");
  assert.match(page, /<main id="main">/);
  assert.doesNotMatch(page, /\sstyle="/, "no inline style attributes");
  assert.doesNotMatch(page, /<style\b/, "no inline style block");

  const blocks = [...page.matchAll(/<script\b((?![^>]*\bsrc=)[^>]*)>([\s\S]*?)<\/script>/g)];
  assert.equal(blocks.length, 1, "one inline block, the speculation rules");
  assert.match(blocks[0][1], /type="speculationrules"/);
  const digest = `sha256-${createHash("sha256").update(blocks[0][2], "utf8").digest("base64")}`;
  assert.ok(csp.includes(`'${digest}'`), `the speculation block needs ${digest} in script-src`);
  const rules = JSON.parse(blocks[0][2]);
  assert.deepEqual(rules.prerender[0].urls, ["../index.html"], "prerender the hub, the way the sibling sections do");
  for (const url of rules.prefetch[0].urls) {
    assert.ok(existsSync(resolve(dirname(pagePath), url, "index.html")), `${url} is a real section`);
  }
});

test("every control the simulation understands exists on the page, with the ids app.js reads", () => {
  const page = readPage();
  for (const id of ["model", "thinking", "batch", "failure", "autoretry", "prompt", "reserve", "keep", "status", "metrics", "compaction-line", "trace", "sim-inputs"]) {
    assert.match(page, new RegExp(`id="${id}"`), `#${id} must exist`);
  }
  for (const model of DEMO_MODELS) assert.match(page, new RegExp(`<option value="${model.id}"`), `${model.id} is selectable`);
  for (const level of THINKING_LEVELS) assert.match(page, new RegExp(`<option value="${level.id}"`), `${level.id} is selectable`);
  for (const tool of TOOLS) assert.match(page, new RegExp(`id="tool-${tool.id}"`), `${tool.id} has a checkbox`);
  assert.match(page, /<input type="range" id="reserve"/);
  assert.match(page, /<input type="range" id="keep"/);
});

test("the renderer holds no numbers of its own: app.js imports them from the simulation", () => {
  const app = readApp();
  assert.match(app, /from "\.\/sim\.js"/, "the renderer takes its model from sim.js");
  for (const literal of ["8192", "819,2", "16384", "32768", "24576", "4096"]) {
    assert.ok(!app.includes(literal), `app.js must not hardcode ${literal}`);
  }
  // The compaction label is built from the run object, never typed as a sentence with a number.
  assert.match(app, /\$\{number\(run\.threshold\)\}/, "the line prints the computed threshold");
  assert.match(app, /\$\{number\(run\.model\.contextWindow\)\}/, "and the computed window");
  assert.match(app, /\$\{number\(run\.reserveTokens\)\}/, "and the reserve it was derived from");
});

// ---------------------------------------------------------------------------------------------
// The browser: every control, really clicked, really changing the page
// ---------------------------------------------------------------------------------------------

const browserBinary = ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"].find((path) => existsSync(path));

test("harness playground: each control changes the trace and the numbers, and nothing leaves the page", { skip: !browserBinary && "no chromium on this machine" }, async () => {
  const { serve } = await import("../scripts/serve.mjs");
  const { launch } = await import("./lib/cdp.mjs");
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: [browserBinary], width: 1280, height: 1000 });
    const requested = [];
    page.on("Network.requestWillBeSent", (params) => requested.push(params.request.url));
    await page.send("Network.enable");
    const open = () =>
      page.goto(url + "harness/", { label: "the harness page", timeout: 60000 });

    // Snapshot everything observable in one round trip, so a change cannot be half-rendered
    // between two reads.
    const snapshot = () =>
      page.evaluate(() => ({
        status: document.querySelector("#status").textContent,
        compaction: document.querySelector("#compaction-line").textContent,
        metrics: document.querySelector("#metrics").textContent,
        trace: document.querySelector("#trace").textContent,
        entries: document.querySelectorAll("#trace .ev").length,
        bars: [...document.querySelectorAll(".ctx-fill")].map((fill) => fill.style.width).join(","),
        marks: [...document.querySelectorAll(".ctx-mark")].map((mark) => mark.style.left).join(","),
        inputs: document.querySelector("#sim-inputs").textContent,
      }));
    const metric = (label) =>
      page.evaluate((wanted) => {
        const term = [...document.querySelectorAll("#metrics dt")].find((dt) => dt.textContent === wanted);
        return term ? term.nextElementSibling.textContent : null;
      }, label);
    const setSelect = (id, value) =>
      page.evaluate((controlId, wanted) => {
        const control = document.getElementById(controlId);
        control.value = wanted;
        control.dispatchEvent(new Event("input", { bubbles: true }));
        control.dispatchEvent(new Event("change", { bubbles: true }));
        return control.value;
      }, id, value);
    const setRange = (id, value) =>
      page.evaluate((controlId, wanted) => {
        const control = document.getElementById(controlId);
        control.value = String(wanted);
        control.dispatchEvent(new Event("input", { bubbles: true }));
        return control.value;
      }, id, value);
    // A real mouse press on the track, at a chosen fraction of its width: the only way to show a
    // slider is wired to the renderer without setting .value from script.
    const clickRangeAt = async (id, fraction) => {
      const at = await page.evaluate((controlId, portion) => {
        const control = document.getElementById(controlId);
        control.scrollIntoView({ block: "center" });
        const box = control.getBoundingClientRect();
        return { x: box.x + box.width * portion, y: box.y + box.height / 2 };
      }, id, fraction);
      await page.clickAt(at.x, at.y);
      return page.evaluate((controlId) => Number(document.getElementById(controlId).value), id);
    };
    // "The line came from reserveTokens": parse the number the page printed and check the
    // arithmetic itself, for whatever the control happens to be set to.
    const compactionLine = async () => {
      const text = await page.evaluate(() => document.querySelector("#compaction-line").textContent);
      const parsed = text.match(/fires above ([\d,]+) tokens: a ([\d,]+)-token window minus reserveTokens ([\d,]+)/);
      assert.ok(parsed, `the compaction line states its arithmetic, saw: ${text}`);
      const values = parsed.slice(1).map((part) => Number(part.replace(/,/g, "")));
      return { above: values[0], window: values[1], reserve: values[2], text };
    };

    await open();
    await page.waitFor(() => document.querySelector("#status").textContent.includes("settled"), {
      label: "the initial run to render",
      timeout: 30000,
    });

    const initial = await snapshot();
    assert.match(initial.status, /^settled/, `the default run settles, saw: ${initial.status}`);
    assert.ok(initial.entries > 10, `the trace renders its entries, found ${initial.entries}`);
    assert.match(initial.trace, /agent_end — this run is closed/);
    assert.match(initial.trace, /agent_settled — no automatic work remains/);
    assert.match(initial.inputs, /base prompt 900 tokens/, "the simulation's constants are published on the page");
    const defaultLine = await compactionLine();
    assert.equal(defaultLine.above, defaultLine.window - defaultLine.reserve, "line = window - reserve");
    const defaultReserve = await page.evaluate(() => Number(document.querySelector("#reserve").value));
    assert.equal(defaultLine.reserve, defaultReserve, "and the line's reserve is the slider's own value");

    // 1. The model's context window.
    assert.equal(await setSelect("model", "deepseek-v4-pro"), "deepseek-v4-pro");
    await page.waitFor(() => document.querySelector("#compaction-line").textContent.includes("32,768"), {
      label: "the wider window to reach the compaction line",
      timeout: 15000,
    });
    const wider = await snapshot();
    const widerLine = await compactionLine();
    assert.equal(widerLine.window, PRO.contextWindow);
    assert.equal(widerLine.above, PRO.contextWindow - widerLine.reserve);
    assert.notEqual(wider.status, initial.status, "a wider window changes the run's numbers");
    assert.notEqual(wider.trace, initial.trace, "and its trace");

    // 2. Thinking level: reasoning tokens are printed, and the bill follows.
    const reasoningBefore = await metric("Reasoning tokens");
    assert.equal(await setSelect("thinking", "high"), "high");
    const reasoningAfter = await metric("Reasoning tokens");
    assert.notEqual(reasoningAfter, reasoningBefore, "the thinking level moves the reasoning token count");
    const thinkingSnap = await snapshot();
    assert.match(thinkingSnap.trace, /reasoning tokens, kept on the message/, "the trace explains the re-send");
    await setSelect("thinking", "off");
    assert.equal(await metric("Reasoning tokens"), "0", "and off means zero");

    // 3. A tool allowlist checkbox, clicked for real.
    const refusedBefore = Number((await metric("Tool calls refused")).replace(/,/g, ""));
    await page.click("#tool-grep");
    const refusedAfter = Number((await metric("Tool calls refused")).replace(/,/g, ""));
    assert.ok(refusedAfter < refusedBefore, `allowing grep removes a refusal (${refusedBefore} → ${refusedAfter})`);
    assert.equal(await page.evaluate(() => document.querySelector("#tool-grep").checked), true, "the click landed");
    await page.click("#tool-grep");
    assert.equal(await page.evaluate(() => document.querySelector("#tool-grep").checked), false, "and the same click turns it back off");

    // The write-class omission is the reviewer story: the run's outcome changes.
    await page.click("#tool-edit");
    await page.waitFor(() => document.querySelector("#status").textContent.startsWith("blocked"), {
      label: "the blocked branch to render",
      timeout: 15000,
    });
    const blocked = await snapshot();
    assert.match(blocked.status, /blocked/);
    assert.notEqual(blocked.trace, thinkingSnap.trace, "a refused write-class call rewrites the run");
    await page.click("#tool-edit");
    await page.waitFor(() => document.querySelector("#status").textContent.startsWith("settled"), {
      label: "the run to settle again",
      timeout: 15000,
    });

    // 4. Batching: one batch per response versus one call per response.
    const parallel = await snapshot();
    const parallelRequests = await metric("Model requests");
    assert.equal(await setSelect("batch", "sequential"), "sequential");
    const sequentialRequests = await metric("Model requests");
    assert.ok(
      Number(sequentialRequests.replace(/,/g, "")) > Number(parallelRequests.replace(/,/g, "")),
      `sequential calls cost more requests (${parallelRequests} → ${sequentialRequests})`,
    );
    const sequential = await snapshot();
    assert.notEqual(sequential.bars, parallel.bars, "and more requests means more context bars on the page");
    assert.notEqual(sequential.metrics, parallel.metrics, "and a bigger bill");
    await setSelect("batch", "parallel");

    // 5. reserveTokens, with real presses on the range's track and then exact values.
    const lowClick = await clickRangeAt("reserve", 0.2);
    const lowLine = await compactionLine();
    assert.equal(lowLine.reserve, lowClick, "a real press moves the slider");
    assert.equal(lowLine.above, lowLine.window - lowClick, "and the printed line is window - the slider");
    const highClick = await clickRangeAt("reserve", 0.8);
    const highLine = await compactionLine();
    assert.ok(highClick > lowClick, `a press further along the track reserves more (${lowClick} → ${highClick})`);
    assert.equal(highLine.above, highLine.window - highClick);
    assert.notEqual(lowLine.above, highLine.above, "the trace's compaction line moved with the control");
    await setRange("reserve", 0);
    const noReserve = await compactionLine();
    assert.equal(noReserve.above, noReserve.window, "a zero reserve puts the line at the window's edge");
    assert.equal(await page.evaluate(() => document.querySelector("#reserve-value").textContent), "0", "the output follows the range");
    await setRange("reserve", PRO.contextWindow - 4096);
    const tight = await compactionLine();
    assert.equal(tight.above, 4096, "reserve 28672 on a 32k window draws the line at 4096");
    const tightCompactions = Number((await metric("Compactions")).replace(/,/g, ""));
    assert.ok(tightCompactions > 0, "and the run compacts");
    assert.match((await snapshot()).trace, /compaction/, "the trace shows it");

    // 6. keepRecentTokens, still on a line low enough for compactions to happen: it changes what a
    //    compaction leaves behind, so both the numbers and the bars move.
    const keepLoose = await snapshot();
    await setRange("keep", 0);
    const keepTight = await snapshot();
    assert.notEqual(keepTight.metrics, keepLoose.metrics, "keepRecentTokens changes the run's numbers");
    assert.notEqual(keepTight.bars, keepLoose.bars, "and the context bars");
    await setRange("keep", PRO.contextWindow);
    assert.match((await snapshot()).trace, /compaction skipped/, "a keep window that covers everything is reported, not ignored");
    await setRange("keep", 4096);
    await setRange("reserve", 8192);

    // 7. The 429: absorbed with auto-retry, fatal without it.
    const cleanRequests = Number((await metric("Model requests")).replace(/,/g, ""));
    assert.equal(await setSelect("failure", "request-2"), "request-2");
    assert.match((await snapshot()).trace, /HTTP 429/, "the refused request appears in the trace");
    assert.match((await snapshot()).trace, /retry · attempt 1/, "and the retry that followed");
    assert.equal(Number((await metric("Retries after a 429"))), 1);
    assert.equal(Number((await metric("Model requests")).replace(/,/g, "")), cleanRequests + 1);
    assert.equal(await setSelect("autoretry", "off"), "off");
    await page.waitFor(() => document.querySelector("#status").textContent.startsWith("failed"), {
      label: "the failed run to render",
      timeout: 15000,
    });
    const failed = await snapshot();
    assert.match(failed.status, /failed/);
    assert.doesNotMatch(failed.trace, /agent_settled — no automatic work remains/, "a failed run never settles");
    await setSelect("autoretry", "on");
    await setSelect("failure", "none");

    // 8. Steering versus a follow-up: the same prompt, two different places in the transcript.
    const quiet = await snapshot();
    assert.equal(await setSelect("prompt", "steer"), "steer");
    const steered = await snapshot();
    assert.match(steered.trace, /user message · steer/);
    assert.equal((steered.trace.match(/agent_settled — no automatic work remains/g) ?? []).length, 1, "a steer does not start a second run");
    assert.notEqual(steered.metrics, quiet.metrics, "a steer costs re-sent tokens");
    assert.equal(await setSelect("prompt", "follow-up"), "follow-up");
    const followed = await snapshot();
    assert.match(followed.trace, /run 2 starts over the same transcript/);
    assert.equal((followed.trace.match(/agent_settled — no automatic work remains/g) ?? []).length, 2, "a follow-up settles twice");
    assert.equal(await metric("Runs"), "2");
    await setSelect("prompt", "none");

    // A phone-width column must not push the page sideways; the trace's long identifiers are the
    // usual cause, and the tables are already inside their own scroll container.
    await page.emulateViewport({ width: 390, height: 844 });
    const narrow = await page.evaluate(() => ({
      client: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth,
    }));
    assert.equal(
      narrow.scroll <= narrow.client,
      true,
      `no horizontal overflow at 390px (scrollWidth ${narrow.scroll} vs clientWidth ${narrow.client})`,
    );
    await page.clearViewport();

    // Everything the page needed came from this loopback server: no key, no provider, no beacon.
    await page.reload();
    const foreign = requested.filter((target) => !target.startsWith(url) && !target.startsWith("data:"));
    assert.deepEqual(foreign, [], `the page must not request anything off-origin, saw ${foreign.join(", ")}`);
    assert.ok(requested.length >= 3, `expected the page and its modules, saw ${requested.join(", ")}`);
  } finally {
    if (page) await page.close();
    await new Promise((done) => server.close(done));
  }
});
