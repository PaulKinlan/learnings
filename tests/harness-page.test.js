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
import {
  ANNOTATION_TOKENS,
  BLOCKED_TOKENS,
  DEACTIVATED_TOKENS,
  EXPLORER_TOOLS,
  LINT_TOOL,
  PLUGINS,
  REDACTION_TOKENS,
  STAMP_TOKENS,
  SUMMARY_TOKENS,
  explore,
} from "../site/harness/plugins.js";

const pagePath = fileURLToPath(new URL("../site/harness/index.html", import.meta.url));
const appPath = fileURLToPath(new URL("../site/harness/app.js", import.meta.url));
const explorerPath = fileURLToPath(new URL("../site/harness/explorer.js", import.meta.url));
const readExplorer = () => readFileSync(explorerPath, "utf8");
const readPage = () => readFileSync(pagePath, "utf8");
const readApp = () => readFileSync(appPath, "utf8");
const sumTokens = (entries) => entries.reduce((total, entry) => total + entry.tokens, 0);

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

test("a run ends with exactly one closing answer, not a scripted one and a loop one", () => {
  for (const options of [{}, { batchMode: "sequential" }, { tools: ["read", "bash"] }, { prompt: "follow-up" }]) {
    const run = simulateRun(options);
    const answers = run.events.filter((event) => event.kind === "assistant" && event.calls.length === 0);
    // A follow-up legitimately buys a second run, and so a second answer; anything else is a double-bill.
    assert.equal(answers.length, options.prompt === "follow-up" ? 2 : 1, JSON.stringify(options));
    const tail = run.events.slice(-3).map((event) => event.kind);
    assert.deepEqual(tail, ["assistant", "run-end", "settled"], "the closing answer is the last request's reply");
  }
});

test("a request can be over the window when the kept tail leaves nothing to compact, and it is reported", () => {
  const run = simulateRun({ thinking: "high", keepRecentTokens: LITE.contextWindow });
  const requests = run.events.filter((event) => event.kind === "request");
  const over = requests.filter((event) => event.contextTokens > LITE.contextWindow);
  assert.ok(over.length > 0, "the transcript grows past a window nothing may be dropped from");
  assert.ok(
    run.events.some((event) => event.kind === "compaction-skipped"),
    "and the reason is on the trace rather than left implicit",
  );
  assert.ok(run.metrics.peakContext > LITE.contextWindow, "the metrics say so too");
});

test("a run is deterministic: the same controls produce the same trace", () => {
  const controls = { model: PRO.id, thinking: "high", reserveTokens: 4096, keepRecentTokens: 1024 };
  assert.deepEqual(simulateRun(controls), simulateRun(controls));
});

// ---------------------------------------------------------------------------------------------
// The explorer's graph: forks, compaction cuts, deactivation and hooks
// ---------------------------------------------------------------------------------------------

test("a fork copies the prefix into a new file and leaves the abandoned branch readable", () => {
  for (const [after, prefixEntries] of [["turn-1", 3], ["turn-2", 6], ["turn-3", 8]]) {
    const run = explore({ forkAfter: after });
    assert.equal(run.fileB.parentSession, "A", "the new file names the session it came from");
    assert.equal(run.fork.prefixEntries, prefixEntries, `${after} forks after ${prefixEntries} entries`);
    assert.deepEqual(run.activeBranch, run.fileB.entries, "the active branch is the new file's branch");
    assert.equal(run.fork.newFileEntries, prefixEntries + 5, "the new file is the prefix plus the alternative branch");
    // The two files share the prefix and nothing else, and the original keeps what it had.
    assert.equal(run.fileA.entries.length, run.fork.prefixEntries + run.fork.abandonedEntries);
    assert.equal(run.abandoned.length, run.fileA.entries.length - prefixEntries);
    assert.deepEqual(
      run.abandoned.map((entry) => entry.id),
      run.fileA.entries.slice(prefixEntries).map((entry) => entry.id),
      "the abandoned branch is still in file A, in order",
    );
    assert.equal(run.fork.deleted, 0, "a fork deletes nothing");
    // The branch taken instead is two turns and an answer, whatever the fork point was.
    assert.equal(run.fileB.entries.length - prefixEntries, 2 * 2 + 1, "two turns of one call each, plus the closing answer");
  }
  const none = explore({ forkAfter: "none" });
  assert.equal(none.fileB, null);
  assert.deepEqual(none.abandoned, []);
  assert.equal(none.fork.deleted, 0);
});

test("the compaction cut is a position in the active branch, and firstKeptEntryId is the entry there", () => {
  const run = explore({ forkAfter: "turn-2", plugins: ["context-redactor"] });
  for (let cut = 0; cut < run.activeBranch.length; cut += 1) {
    const withCut = explore({ forkAfter: "turn-2", plugins: ["context-redactor"], cut });
    const compaction = withCut.compaction;
    assert.equal(compaction.cutIndex, cut);
    assert.equal(compaction.firstKeptEntryId, withCut.activeBranch[cut].id, `cut ${cut} names its own entry`);
    assert.equal(compaction.kept.length, withCut.activeBranch.length - cut, "the kept range runs to the leaf");
    assert.equal(compaction.kept[0].id, compaction.firstKeptEntryId, "firstKeptEntryId is the first kept entry");
    assert.equal(compaction.keptTokens, sumTokens(withCut.activeBranch.slice(cut)), "kept tokens are the tail");
    assert.equal(compaction.replacedTokens, sumTokens(withCut.activeBranch.slice(0, cut)), "replaced tokens are the head");
    assert.equal(
      compaction.summaryTokens,
      cut ? Math.round(SUMMARY_TOKENS.base + SUMMARY_TOKENS.ratio * compaction.replacedTokens) : 0,
      "the summary's size comes from what it read",
    );
    assert.equal(
      compaction.contextAfter,
      withCut.metrics.promptTokens + compaction.summaryTokens + compaction.keptTokens - REDACTION_TOKENS,
      "the rebuilt context is prompt + summary + kept tail",
    );
    assert.equal(compaction.entriesBefore, withCut.activeBranch.length, "every entry is still there to cut");
    assert.equal(compaction.deleted, 0, "compaction replaces context, it does not delete entries");
  }
  // Moving the cut moves the context: a shallow cut saves less than a deep one.
  const shallow = explore({ cut: 1 });
  const deep = explore({ cut: 8 });
  assert.ok(deep.compaction.replacedTokens > shallow.compaction.replacedTokens);
  assert.ok(deep.compaction.contextAfter < shallow.compaction.contextAfter, "compacting more leaves less to send");
  assert.notEqual(deep.compaction.firstKeptEntryId, shallow.compaction.firstKeptEntryId);
});

test("the cut is clamped to the branch, so a fork into a shorter branch cannot point past the leaf", () => {
  const forked = explore({ forkAfter: "turn-3", cut: 99 });
  assert.equal(forked.compaction.cutIndex, forked.activeBranch.length - 1);
  assert.equal(forked.compaction.firstKeptEntryId, forked.activeBranch.at(-1).id);
});

test("deactivating a tool mid-run fails the next call and leaves the registry alone", () => {
  const before = explore();
  const after = explore({ deactivateTool: "edit", deactivateAfter: 2 });
  assert.equal(after.metrics.deactivated, 1, "the edit call after turn 2 fails");
  assert.equal(after.metrics.executed, before.metrics.executed - 1, "and it does not run");
  const failed = after.fileA.entries.find((entry) => entry.outcome === "deactivated");
  assert.ok(failed, "the failure is an entry with its own outcome");
  assert.equal(failed.id, "t3.1", "it is the call the run made after the deactivation");
  assert.equal(failed.tokens, DEACTIVATED_TOKENS);
  // Registered is not active: the tool is still in the registry, and still offered at construction.
  assert.deepEqual(after.registry, before.registry, "the registry did not change");
  assert.equal(after.unregister.exists, false, "and there is no unregisterTool to change it with");
  assert.equal(after.metrics.requests, before.metrics.requests, "a failed call does not end the run");
  assert.deepEqual(
    after.notes.filter((note) => /stays registered/.test(note)).length,
    1,
    "the page says the tool stays registered",
  );
  // A deactivation that lands on no call is reported rather than silently ignored.
  const idle = explore({ deactivateTool: "write", deactivateAfter: 1 });
  assert.equal(idle.metrics.deactivated, 0);
  assert.ok(idle.notes.some((note) => note.startsWith("nothing was deactivated")), "the no-op is named");
});

test("terminate is a vote: a unanimous one-call batch stops the run, a split two-call batch does not", () => {
  const split = explore({ guard: "bash-terminate" });
  assert.equal(split.metrics.terminateVotes, 2, "both bash calls voted");
  assert.equal(split.metrics.terminated, true, "but only the second batch was unanimous");
  assert.equal(split.metrics.turns, 4, "the run got as far as turn 4");
  assert.equal(split.metrics.requests, 4, "and made no closing request");

  const unanimous = explore({ guard: "edit-terminate" });
  assert.equal(unanimous.metrics.turns, 3, "turn 3 carries one call, so its single vote is the batch");
  assert.equal(unanimous.metrics.requests, 3, "no follow-up request after an agreed terminate");
  assert.equal(unanimous.metrics.terminated, true);
  assert.equal(
    unanimous.fileA.entries.at(-1).id,
    "t3.1",
    "the run ends at the batch that terminated: no closing answer entry",
  );
  assert.ok(unanimous.trace.some((event) => /every one of the 1 finalized result/.test(event.text ?? "")));

  const plain = explore({ guard: "edit" });
  assert.equal(plain.metrics.terminated, false, "blocking without terminate never stops the loop");
  assert.equal(plain.metrics.blocked, 1);
  assert.equal(plain.metrics.requests, 5);
  assert.equal(plain.fileA.entries.find((entry) => entry.outcome === "blocked").tokens, BLOCKED_TOKENS);
});

test("a tool that is not offered is refused before any hook sees it", () => {
  const offered = explore({ offered: ["read", "bash", "write"] });
  assert.equal(offered.metrics.notOffered, 1, "the edit call is not in the schema");
  assert.equal(offered.metrics.blocked, 0);
  const both = explore({ offered: ["read", "bash", "write"], guard: "edit", deactivateTool: "edit", deactivateAfter: 1 });
  assert.equal(both.metrics.blocked, 0, "no tool_call dispatch: the tool was never offered");
  assert.equal(both.metrics.deactivated, 0, "and no deactivation to report either");
  assert.ok(both.notes.some((note) => /nothing was deactivated/.test(note)));
  // Offering a name nothing registered is neither a tool nor a schema.
  const unregistered = explore();
  assert.ok(!unregistered.registry.registered.includes(LINT_TOOL.id));
  assert.equal(unregistered.metrics.calls, 5);
  assert.ok(unregistered.notes.some((note) => /nothing registered it/.test(note)), "the page says offering is not registering");
});

test("each plugin's dispatch count follows the loop, and an observer moves no number", () => {
  const base = explore();
  const observer = explore({ plugins: ["turn-logger"] });
  assert.deepEqual(observer.metrics, base.metrics, "turn_end observes, so no number moves");
  assert.equal(observer.plugins[0].fires, observer.metrics.turns, "one turn_end dispatch per turn");

  const redacted = explore({ plugins: ["context-redactor"] });
  assert.equal(redacted.plugins[0].fires, redacted.metrics.requests, "context fires per request");
  assert.equal(redacted.metrics.redacted, REDACTION_TOKENS * redacted.metrics.requests);
  assert.equal(
    redacted.compaction.contextBefore,
    base.compaction.contextBefore - REDACTION_TOKENS,
    "redaction is subtracted from what is sent, not from the transcript",
  );
  assert.equal(
    redacted.fileA.tokens,
    base.fileA.tokens,
    "the entries the redaction covers are still in the file",
  );

  const stamped = explore({ plugins: ["prompt-stamp"] });
  assert.equal(stamped.plugins[0].fires, 1, "before_agent_start fires once per user prompt, not once per request");
  assert.ok(stamped.metrics.requests > 1, "even though the run makes several requests");
  assert.equal(stamped.metrics.promptTokens, base.metrics.promptTokens + STAMP_TOKENS);

  const annotated = explore({ plugins: ["result-annotator"] });
  assert.equal(annotated.plugins[0].fires, annotated.metrics.executed, "tool_result fires per result that ran");
  assert.equal(annotated.metrics.annotated, ANNOTATION_TOKENS * annotated.metrics.executed);

  const withTool = explore({ plugins: ["lint-tool"] });
  assert.ok(withTool.registry.registered.includes(LINT_TOOL.id), "registerTool adds to the registry");
  assert.equal(withTool.metrics.executed, base.metrics.executed + 1, "and the run calls the tool it did not have");
  assert.equal(
    withTool.metrics.promptTokens,
    base.metrics.promptTokens + LINT_TOOL.schemaTokens,
    "the new tool's schema is in the prompt, and it is not free",
  );

  const everyPlugin = explore({ plugins: PLUGINS.map((plugin) => plugin.id) });
  assert.deepEqual(
    everyPlugin.plugins.map((plugin) => plugin.order),
    [1, 2, 3, 4, 5],
    "the load order is the order they were added",
  );
  assert.equal(everyPlugin.plugins.map((plugin) => plugin.id).join(","), PLUGINS.map((plugin) => plugin.id).join(","));
});

test("the unregister attempt finds no API and changes no registry", () => {
  const run = explore({ unregisterAttempt: true });
  assert.equal(run.unregister.exists, false);
  assert.equal(run.unregister.hits, 0);
  assert.equal(run.unregister.registeredBefore, run.unregister.registeredAfter, "nothing left the registry");
  assert.equal(run.unregister.activeAfter, run.registry.offeredCount);
  assert.equal(run.registry.registeredCount, EXPLORER_TOOLS.length, "the four built-ins are still registered");
  assert.equal(explore().unregister.attempted, false, "the result is only shown once the button is pressed");
});

test("the explorer is deterministic: the same controls produce the same graph", () => {
  const controls = {
    plugins: ["turn-logger", "lint-tool"],
    offered: ["read", "bash", "lint"],
    guard: "bash",
    deactivateTool: "read",
    deactivateAfter: 1,
    forkAfter: "turn-1",
    cut: 3,
    unregisterAttempt: true,
  };
  assert.deepEqual(explore(controls), explore(controls));
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
  for (const target of ["../style.css", "./harness.css", "./app.js", "./explorer.js", "../index.html", "../decision-models/"]) {
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

test("the explorer's controls exist on the page, with the ids explorer.js reads", () => {
  const page = readPage();
  for (const id of [
    "add-turn-logger",
    "add-context-redactor",
    "add-prompt-stamp",
    "add-result-annotator",
    "add-lint-tool",
    "plugin-clear",
    "plugin-order",
    "deactivate-tool",
    "deactivate-after",
    "unregister-attempt",
    "unregister-result",
    "guard",
    "fork",
    "cut",
    "cut-value",
    "explorer-status",
    "explorer-metrics",
    "cut-line",
    "branch-line",
    "tree",
    "explorer-trace",
    "explorer-notes",
  ]) {
    assert.match(page, new RegExp(`id="${id}"`), `#${id} must exist`);
  }
  for (const plugin of PLUGINS) assert.match(page, new RegExp(`id="add-${plugin.id}"`), `${plugin.id} can be added`);
  for (const tool of [...EXPLORER_TOOLS, LINT_TOOL]) {
    assert.match(page, new RegExp(`id="explore-tool-${tool.id}"`), `${tool.id} can be offered`);
  }
  for (const guard of ["none", "edit", "bash", "bash-terminate", "edit-terminate"]) {
    assert.match(page, new RegExp(`<option value="${guard}"`), `the ${guard} guard is selectable`);
  }
  for (const point of ["none", "turn-1", "turn-2", "turn-3"]) {
    assert.match(page, new RegExp(`<option value="${point}"`), `the ${point} fork point is selectable`);
  }
  assert.match(page, /<input type="range" id="cut"/);
  assert.match(page, /<script type="module" src="\.\/explorer\.js">/, "the explorer's renderer is loaded");
});

test("the explorer's renderer holds no numbers of its own: explorer.js imports them from plugins.js", () => {
  const explorer = readExplorer();
  assert.match(explorer, /from "\.\/plugins\.js"/, "the renderer takes its model from plugins.js");
  for (const literal of ["1,820", "1820", "7740", "7,740", "300", "0.18", "220", "260", "240", "200", "190"]) {
    assert.ok(!explorer.includes(literal), `explorer.js must not hardcode ${literal}`);
  }
  // The numbers it prints come out of the returned run, not out of a sentence typed by hand.
  assert.match(explorer, /\$\{number\(run\.compaction\.contextBefore\)\}/, "the context before is computed");
  assert.match(explorer, /\$\{number\(run\.compaction\.contextAfter\)\}/, "and so is the context after the summary");
  assert.match(explorer, /run\.compaction\.firstKeptEntryId/, "firstKeptEntryId is the run's, not a literal");
  assert.match(explorer, /run\.registry\.registeredCount/, "the registry size is the run's");
  // And the honesty labels the section promises are actually on the page, not only in the module.
  const page = readPage();
  assert.match(page, /deactivated, not unregistered/i, "the deactivation is labelled for what it is");
  assert.match(page, /What this does not model/, "the explorer states its limits");
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
    const firstPaint = await page.evaluate(() => ({
      reserve: document.querySelector("#reserve").value,
      reserveOut: document.querySelector("#reserve-value").textContent,
      keep: document.querySelector("#keep").value,
      keepOut: document.querySelector("#keep-value").textContent,
    }));
    assert.equal(firstPaint.reserveOut, firstPaint.reserve, "the reserve output is filled by the first render");
    assert.equal(firstPaint.keepOut, firstPaint.keep, "and so is the keep output");
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

    // A configuration that cannot compact anything leaves requests over the window; the trace has
    // to say so, because the bar is capped at 100%.
    await setSelect("model", "deepseek-v4-lite");
    await setSelect("thinking", "high");
    await setRange("keep", LITE.contextWindow);
    const overWindow = await snapshot();
    assert.match(overWindow.trace, /over the window by [\d,]+ tokens/, "an over-window request is explained");
    assert.match(overWindow.trace, /nothing was left to compact/, "and the reason is named");
    await setRange("keep", 4096);
    await setSelect("thinking", "low");

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

test("plugin explorer: every control changes the graph, the numbers or the trace", { skip: !browserBinary && "no chromium on this machine" }, async () => {
  const { serve } = await import("../scripts/serve.mjs");
  const { launch } = await import("./lib/cdp.mjs");
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: [browserBinary], width: 1280, height: 1000 });
    await page.goto(url + "harness/", { label: "the harness page", timeout: 60000 });
    await page.waitFor(() => document.querySelector("#explorer-status").textContent.includes("request"), {
      label: "the explorer to render",
      timeout: 30000,
    });

    const snapshot = () =>
      page.evaluate(() => ({
        status: document.querySelector("#explorer-status").textContent,
        metrics: document.querySelector("#explorer-metrics").textContent,
        cut: document.querySelector("#cut-line").textContent,
        branch: document.querySelector("#branch-line").textContent,
        order: document.querySelector("#plugin-order").textContent,
        trace: document.querySelector("#explorer-trace").textContent,
        notes: document.querySelector("#explorer-notes").textContent,
        unregister: document.querySelector("#unregister-result").hidden
          ? ""
          : document.querySelector("#unregister-result").textContent,
        files: [...document.querySelectorAll("#tree .file h4")].map((heading) => heading.textContent).join(" | "),
        entries: document.querySelectorAll("#tree .entry").length,
        abandoned: document.querySelectorAll("#tree .entry-abandoned").length,
        replaced: document.querySelectorAll("#tree .entry-replaced").length,
        summaries: document.querySelectorAll("#tree .entry-summary").length,
        cutMax: Number(document.querySelector("#cut").max),
        cutValue: document.querySelector("#cut").value,
        cutLabel: document.querySelector("#cut-value").textContent,
      }));
    const metric = (label) =>
      page.evaluate((wanted) => {
        const term = [...document.querySelectorAll("#explorer-metrics dt")].find((dt) => dt.textContent === wanted);
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
    const pressRangeAt = async (id, fraction) => {
      const at = await page.evaluate((controlId, portion) => {
        const control = document.getElementById(controlId);
        control.scrollIntoView({ block: "center" });
        const box = control.getBoundingClientRect();
        return { x: box.x + box.width * portion, y: box.y + box.height / 2 };
      }, id, fraction);
      await page.clickAt(at.x, at.y);
      return page.evaluate((controlId) => Number(document.getElementById(controlId).value), id);
    };
    const firstKept = async () => {
      const text = await metric("firstKeptEntryId");
      assert.ok(text, "the metrics panel names firstKeptEntryId");
      return text;
    };

    // The explorer's first paint: one file, one branch, and a cut that is already a position.
    const base = await snapshot();
    assert.equal(base.files, "file A · the session that ran", `one file to start with, saw: ${base.files}`);
    assert.equal(base.abandoned, 0, "nothing is abandoned before a fork");
    assert.equal(base.summaries, 1, "the compaction marker is drawn where the cut falls");
    assert.ok(base.replaced > 0, "and the entries before it are marked as replaced");
    assert.match(base.status, /^5 requests, 5 tool calls/, `the default run, saw: ${base.status}`);
    assert.equal(base.cutLabel, await firstKept(), "the cut label is firstKeptEntryId");
    assert.match(base.order, /Nothing loaded/, "no plugin is loaded to begin with");
    assert.equal(await metric("Entries deleted"), "0");

    // 1. A plugin that only observes: the load order and the trace move, no number does.
    await page.click("#add-turn-logger");
    const logged = await snapshot();
    assert.match(logged.order, /1\. turn-logger — on\("turn_end"\)/, "the load order lists the plugin");
    assert.match(logged.order, /4 dispatches on this run/, "and how many times its hook fired");
    assert.match(logged.trace, /turn-logger ran; it only observes/, "the trace shows the dispatch");
    assert.equal(logged.metrics, base.metrics, "an observer moves no number");

    // 2. A plugin that rewrites the context: every request is smaller.
    await page.click("#add-context-redactor");
    const redacted = await snapshot();
    assert.match(redacted.order, /2\. context-redactor/, "load order is the order they were added");
    assert.match(redacted.trace, /context handler dropped 180 tokens/);
    assert.notEqual(redacted.metrics, logged.metrics, "redaction changes the numbers");
    const before = Number((await metric("Context before compaction")).replace(/,/g, ""));
    assert.ok(before > 0);
    const cutBefore = redacted.cut;
    assert.notEqual(cutBefore, base.cut, "and the cut line reports the new context");

    // 3. A plugin that registers a tool: the registry grows and the run calls it.
    await page.click("#add-lint-tool");
    const withTool = await snapshot();
    assert.match(await metric("Tools registered"), /5 · read, bash, edit, write, lint/, "registerTool adds a tool");
    assert.match(await metric("Tool calls"), /6 \(6 ran/, "and the run calls it");
    assert.notEqual(withTool.metrics, redacted.metrics);

    // 4. Clearing the load set puts every number back.
    await page.click("#plugin-clear");
    const cleared = await snapshot();
    assert.match(cleared.order, /Nothing loaded/);
    assert.equal(cleared.metrics, base.metrics, "with no plugins, the numbers are the first paint's");
    assert.equal(await page.evaluate(() => document.querySelector("#add-turn-logger").disabled), false, "the button is usable again");

    // 5. A fork: a second file, an abandoned branch that is still readable, nothing deleted.
    assert.equal(await setSelect("fork", "turn-2"), "turn-2");
    const forked = await snapshot();
    assert.match(forked.files, /file A · the session that ran \| file B · forked after turn 2/, "two files");
    assert.ok(forked.abandoned > 0, "the abandoned branch is drawn");
    assert.match(forked.branch, /File A is untouched/, "and the page says the original is intact");
    assert.match(forked.branch, /Entries deleted: 0/);
    assert.notEqual(forked.metrics, cleared.metrics, "the graph is a different graph");
    // The cut's rails are the active branch: an earlier fork leaves a shorter one.
    assert.equal(await setSelect("fork", "turn-1"), "turn-1");
    const earlier = await snapshot();
    assert.ok(earlier.cutMax < forked.cutMax, `an earlier fork leaves a shorter branch (${earlier.cutMax} < ${forked.cutMax})`);
    assert.ok(earlier.abandoned > forked.abandoned, "and abandons more of the original branch");
    assert.equal(await setSelect("fork", "turn-2"), "turn-2");

    // 6. The compaction cut, moved with a real press on the track.
    const pressed = await pressRangeAt("cut", 0.8);
    const moved = await snapshot();
    assert.equal(moved.cutValue, String(pressed), "a real press moves the cut");
    assert.notEqual(moved.cutLabel, forked.cutLabel, "firstKeptEntryId slides with the cut");
    assert.equal(moved.cutLabel, await firstKept(), "the label and the metric agree");
    assert.notEqual(moved.cut, forked.cut, "the cut line reports what the new cut replaces");
    assert.notEqual(moved.replaced, forked.replaced, "the entries marked as replaced follow the cut");
    assert.equal(await metric("Entries deleted"), "0", "moving the cut deletes nothing");

    // 7. Deactivate a tool mid-run: the call fails, the registry does not change.
    assert.equal(await setSelect("deactivate-tool", "edit"), "edit");
    assert.equal(await setSelect("deactivate-after", "2"), "2");
    const deactivated = await snapshot();
    assert.match(await metric("Deactivated mid-run"), /^edit after turn 2 · still registered/);
    assert.equal(Number((await metric("Tool calls")).match(/(\d+) deactivated/)[1]), 1, "one call was deactivated");
    assert.match(deactivated.trace, /deactivated, not unregistered/, "the trace says what it is");
    assert.match(await metric("Tools registered"), /4 · read, bash, edit, write/, "the tool is still registered");
    await setSelect("deactivate-tool", "");

    // 8. A terminate vote in a one-call batch stops the run; in a two-call batch it does not.
    assert.equal(await setSelect("guard", "bash-terminate"), "bash-terminate");
    const split = await snapshot();
    assert.match(split.status, /stopped by a unanimous terminate vote after turn 4/, "one vote of two is not a veto");
    assert.equal(await metric("Model requests"), "4", "the run stopped without a closing request");
    assert.equal(await setSelect("guard", "edit-terminate"), "edit-terminate");
    const unanimous = await snapshot();
    assert.match(unanimous.status, /after turn 3/, "the one-call batch agreed, so the loop stopped there");
    assert.equal(await metric("Model requests"), "3");
    assert.notEqual(unanimous.entries, split.entries, "and the graph is shorter");
    await setSelect("guard", "none");

    // 9. A construction-time allowlist is not the same thing as a registry.
    await page.click("#explore-tool-edit");
    const withheld = await snapshot();
    assert.match(await metric("Tool calls"), /1 not offered/, "the edit call is not in the schema");
    assert.match(withheld.trace, /is not in this session's tool set/);
    await page.click("#explore-tool-edit");

    // 10. The removal that does not exist, attempted for real.
    assert.equal(await page.evaluate(() => document.querySelector("#unregister-result").hidden), true);
    await page.click("#unregister-attempt");
    const attempted = await snapshot();
    assert.match(attempted.unregister, /no such API/, "the honest result");
    assert.match(attempted.unregister, /Tools registered before: 4; after: 4/, "and the registry is unchanged");
    assert.match(attempted.unregister, /setActiveTools/, "with the supported alternative named");

    // 11. A phone-width column must not push the page sideways, graph entries included.
    await page.emulateViewport({ width: 390, height: 844 });
    const narrow = await page.evaluate(() => ({
      client: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth,
    }));
    assert.equal(narrow.scroll <= narrow.client, true, `no horizontal overflow at 390px (${narrow.scroll} vs ${narrow.client})`);
    await page.clearViewport();
  } finally {
    if (page) await page.close();
    await new Promise((done) => server.close(done));
  }
});
