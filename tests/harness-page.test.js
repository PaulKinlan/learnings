// tests/harness-page.test.js — the dsh/Cordis page and its playground.
//
// Two halves, deliberately in one file:
//   1. the playground's state machine, asserted directly against runtime.js, so a label that stops
//      tracking its control fails without a browser: the ledger returns to zero in reverse
//      registration order, a provider's dependents deactivate before it unwinds, a seam swap
//      re-registers the consumer without editing it, a rejected pre-step closes a durable turn with
//      zero steps, and a goal blocks with `round-limit`;
//   2. a real browser driving every control on the page, because the requirement is that each
//      control visibly changes the ledger, the services or the numbers — a property of the
//      rendered page, not of the module. One browser, closed in a finally, every wait bounded.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_ROUND_CAP,
  PLUGINS,
  SERVICE_KEYS,
  hotSwapLoop,
  initialRuntime,
  mount,
  mountFork,
  runGoal,
  runTurn,
  shellProvider,
  swapShell,
  unmount,
  unmountFork,
} from "../site/harness/runtime.js";

const pagePath = fileURLToPath(new URL("../site/harness/index.html", import.meta.url));
const appPath = fileURLToPath(new URL("../site/harness/app.js", import.meta.url));
const readPage = () => readFileSync(pagePath, "utf8");
const readApp = () => readFileSync(appPath, "utf8");

const pluginById = (id) => PLUGINS.find((plugin) => plugin.id === id);
const rowsFor = (state, id) => state.effects.filter((row) => row.plugin === id).map((row) => row.seq);
const seqsOf = (rows) => rows.map((row) => row.seq);

// ---------------------------------------------------------------------------------------------
// The playground's state machine
// ---------------------------------------------------------------------------------------------

test("the default composition mounts dsh-base's plugins and publishes every count from them", () => {
  const state = initialRuntime();
  assert.equal(state.counts.mounted, PLUGINS.filter((plugin) => plugin.defaultMounted).length);
  assert.equal(state.counts.active, state.counts.mounted, "every default plugin can satisfy its injects");
  assert.equal(state.counts.registrations, state.effects.length);
  assert.equal(
    state.counts.registrations,
    state.active.reduce((total, id) => total + pluginById(id).effects.length, 0),
    "the ledger is exactly the active plugins' declared registrations",
  );
  // dsh-base mounts the sandboxed provider behind ctx.shell, not the unconfined one.
  assert.equal(state.services.shell, "bash-sandbox");
  assert.equal(state.services.approval, "approval");
  assert.equal(state.services.permissionPresets, "permission-presets");
  assert.equal(pluginById("bash-local").defaultMounted, undefined, "the unconfined executor is not the default");
  for (const key of SERVICE_KEYS) assert.ok(key in state.services || state.services[key] === undefined);
});

test("unmounting unwinds in reverse registration order, and unmounting everything ends at zero", () => {
  const start = initialRuntime();
  const toolBash = rowsFor(start, "tool-bash");
  assert.equal(toolBash.length, 3);

  const after = unmount(start, "tool-bash");
  assert.deepEqual(seqsOf(after.unwind.rows), [...toolBash].reverse(), "the inverses run newest first");
  assert.equal(after.counts.registrations, start.counts.registrations - 3);
  assert.deepEqual(rowsFor(after, "tool-bash"), [], "the ledger holds no row for an unmounted plugin");
  assert.ok(!after.registry.tools.some((tool) => tool.name === "bash"), "and its tool left the registry");

  // Every plugin off, one at a time: 28 registrations become 0 and nothing is left over.
  let state = start;
  for (const plugin of PLUGINS) state = unmount(state, plugin.id);
  assert.equal(state.counts.registrations, 0);
  assert.equal(state.effects.length, 0);
  assert.equal(state.counts.active, 0);
  assert.equal(state.registry.tools.length, 0);
  assert.equal(state.counts.mounted, 0);
  assert.equal(
    new Set(state.detached).size,
    PLUGINS.filter((plugin) => plugin.defaultMounted).length,
    "every mounted plugin is detached exactly once (unmounting a plugin that was never mounted is a no-op)",
  );
  assert.deepEqual(state.services, {}, "no provider is left claiming a key");
});

test("a provider's dependents deactivate before its own effects unwind", () => {
  const start = initialRuntime();
  const toolBash = rowsFor(start, "tool-bash");
  const provider = rowsFor(start, "bash-sandbox");

  const after = unmount(start, "bash-sandbox");
  const order = after.unwind.rows.map((row) => row.plugin);
  assert.deepEqual(order, ["tool-bash", "tool-bash", "tool-bash", "bash-sandbox", "bash-sandbox"]);
  assert.deepEqual(
    seqsOf(after.unwind.rows.slice(0, 3)),
    [...toolBash].reverse(),
    "the dependent's own registrations still come off newest-first",
  );
  assert.deepEqual(seqsOf(after.unwind.rows.slice(3)), [...provider].reverse());
  assert.equal(after.inactive.length, 1);
  assert.equal(after.inactive[0].id, "tool-bash");
  assert.deepEqual(after.inactive[0].waitingFor, ["shell"], "the missing coeffect is named");
  assert.equal(after.counts.registrations, start.counts.registrations - 5);

  // Put the provider back: the dependent reactivates and re-registers at the end of the ledger.
  const back = mount(after, "bash-sandbox");
  assert.equal(back.counts.registrations, start.counts.registrations);
  assert.deepEqual(back.inactive, []);
  assert.ok(Math.min(...rowsFor(back, "tool-bash")) > Math.max(...toolBash), "reactivation registers afresh");
  assert.ok(back.registry.tools.some((tool) => tool.name === "bash"));
});

test("a consumer mounted before its provider waits in PENDING, then activates", () => {
  const withConsumer = mount(initialRuntime(), "tool-fs");
  assert.deepEqual(withConsumer.inactive, [{ id: "tool-fs", waitingFor: ["fs"] }]);
  assert.equal(withConsumer.counts.registrations, initialRuntime().counts.registrations, "no registration is live");
  assert.ok(!withConsumer.registry.tools.some((tool) => tool.name === "read"), "and no schema was added");

  const satisfied = mount(withConsumer, "fs-local");
  assert.deepEqual(satisfied.inactive, []);
  const names = satisfied.registry.tools.map((tool) => tool.name);
  for (const name of ["read", "edit", "write"]) assert.ok(names.includes(name), `${name} is registered`);
  assert.equal(
    satisfied.registry.schemaTokens,
    initialRuntime().registry.schemaTokens + 610,
    "the three schemas cost their declared tokens",
  );
});

test("swapping the shell seam re-registers the consumer and never edits it", () => {
  const start = initialRuntime();
  const before = rowsFor(start, "tool-bash");
  const toolEffectCount = pluginById("tool-bash").effects.length;

  const swapped = swapShell(start, "bash-local");
  assert.equal(shellProvider(swapped).id, "bash-local", "ctx.shell now resolves to the unconfined provider");
  assert.equal(swapped.services.shell, "bash-local");
  assert.equal(swapped.counts.registrations, start.counts.registrations, "the same number of registrations");
  const after = rowsFor(swapped, "tool-bash");
  assert.equal(after.length, toolEffectCount, "the tool's registration count did not change");
  assert.ok(Math.min(...after) > Math.max(...before), "the consumer unloaded and reloaded, so its rows moved to the end");
  assert.equal(pluginById("tool-bash").pkg, "@deepseek-ai/dsh-tool-bash", "the consumer package is the same one");
  assert.deepEqual(swapped.registry.tools.map((tool) => tool.name), ["bash"], "and so is the tool it registers");
  assert.equal(swapped.unwind.rows[0].plugin, "tool-bash", "the consumer unwound before the provider it lost");

  const turn = runTurn(swapped, {});
  assert.equal(turn.turn.provider, "@deepseek-ai/dsh-bash-local", "calls now route to the new provider");
  const back = swapShell(swapped, "bash-sandbox");
  assert.equal(runTurn(back, {}).turn.provider, "@deepseek-ai/dsh-bash-sandbox");
});

test("a rejected pre-step closes a durable turn with zero steps", () => {
  const start = initialRuntime();
  const before = start.sessionLog.length;
  const rejected = runTurn(start, { preStep: "reject" });

  assert.equal(rejected.turn.steps, 0, "no step ran");
  assert.equal(rejected.turn.requests, 0, "no model request was made");
  assert.equal(rejected.turn.toolCalls, 0);
  assert.ok(rejected.sessionLog.length > before, "the turn is still in the durable log");
  const turnEvents = rejected.sessionLog.slice(before).map((event) => event.type);
  assert.equal(turnEvents[0], "turn/start");
  assert.equal(turnEvents.at(-1), "turn/end", "the turn opens and closes");
  assert.ok(!turnEvents.includes("step/start"), "and no step is between them");
  assert.equal(rejected.turnCount, 1);
});

test("an accepted turn is one step: one request, one tool call, the active provider", () => {
  const entered = runTurn(initialRuntime(), {});
  assert.equal(entered.turn.steps, 1);
  assert.equal(entered.turn.requests, 1);
  assert.equal(entered.turn.toolCalls, 1);
  assert.equal(entered.turn.provider, "@deepseek-ai/dsh-bash-sandbox");
  const names = entered.turn.flow.map((entry) => entry.name);
  assert.deepEqual(names, [
    "turn/start",
    "agent/pre-step",
    "step/start",
    "agent/request",
    "llm/stream",
    "assistant/message",
    "tool/call",
    "tool/result",
    "step/end",
    "agent/turn-stopping",
    "turn/end",
  ]);
  assert.equal(entered.sessionLog.filter((event) => event.turn === entered.turn.id).length, 11);
});

test("a goal runs one turn per round and blocks with round-limit at the cap", () => {
  const goal = runGoal(initialRuntime(), { cap: 4 });
  assert.equal(goal.goal.phase, "blocked");
  assert.equal(goal.goal.rounds, 4);
  assert.equal(goal.goal.cap, 4);
  assert.equal(goal.goal.blocker.code, "round-limit");
  assert.equal(goal.goal.steps, 4, "one step per round, because each round is one accepted turn");
  assert.equal(goal.turnCount, 4, "a round is a turn in the same session");
  assert.ok(goal.sessionLog.length > initialRuntime().sessionLog.length + 4 * 10);
  assert.equal(DEFAULT_ROUND_CAP, 256, "the shipped cap is the one the docs name");
});

test("the goal cannot schedule a round without its service or its driver", () => {
  const withoutService = runGoal(unmount(initialRuntime(), "goal"), { cap: 3 });
  assert.equal(withoutService.goal.phase, "inactive");
  assert.equal(withoutService.goal.steps, 0);
  assert.equal(withoutService.turnCount, 0, "nothing was scheduled");

  const withoutDriver = runGoal(unmount(initialRuntime(), "goal-round-driver"), { cap: 3 });
  assert.equal(withoutDriver.goal.phase, "inactive");
  assert.equal(withoutDriver.turnCount, 0);
});

test("hot-swapping the loop unwinds only the loop and leaves the session log alone", () => {
  const start = runTurn(initialRuntime(), {});
  const before = start.sessionLog.length;
  const swapped = hotSwapLoop(start);

  assert.equal(swapped.loopGeneration, 2);
  assert.equal(swapped.sessionLog.length, before, "the log is owned by dsh-session, not by the swapped plugin");
  assert.deepEqual(
    swapped.sessionLog.map((event) => event.seq),
    start.sessionLog.map((event) => event.seq),
    "and its events are the same events",
  );
  assert.deepEqual(
    seqsOf(swapped.unwind.rows),
    [...rowsFor(start, "agent-loop")].reverse(),
    "the old generation's inverses ran newest-first",
  );
  assert.ok(Math.min(...rowsFor(swapped, "agent-loop")) > Math.max(...rowsFor(start, "agent-loop")));
  assert.equal(swapped.counts.registrations, start.counts.registrations, "the same composition, one generation later");

  const nextTurn = runTurn(swapped, {});
  assert.ok(nextTurn.sessionLog.length > swapped.sessionLog.length);
  assert.equal(nextTurn.sessionLog.at(-1).seq, nextTurn.sessionLog.length, "the log keeps counting");
});

test("an isolated scope's registrations are not in the root ledger", () => {
  const start = initialRuntime();
  const forked = mountFork(start);
  assert.notEqual(forked.fork, null);
  assert.equal(forked.fork.effects.length, pluginById("bash-sandbox").effects.length);
  assert.equal(forked.counts.registrations, start.counts.registrations, "the root ledger is untouched");
  assert.equal(forked.services.shell, "bash-sandbox", "and the root provider is unchanged");

  const unmounted = unmountFork(forked);
  assert.equal(unmounted.fork, null);
  assert.equal(unmounted.forkUnwind.rows.length, forked.fork.effects.length, "every isolated inverse ran");
  assert.equal(unmounted.counts.registrations, start.counts.registrations);
  assert.deepEqual(unmounted.services, start.services);
});

test("the same controls produce the same state", () => {
  const drive = () => {
    let state = initialRuntime();
    state = unmount(state, "bash-sandbox");
    state = mount(state, "bash-sandbox");
    state = mount(state, "tool-fs");
    state = mount(state, "fs-local");
    state = swapShell(state, "bash-local");
    state = runTurn(state, { preStep: "reject" });
    state = runTurn(state, {});
    state = hotSwapLoop(state);
    state = mountFork(state);
    state = runGoal(state, { cap: 2 });
    return state;
  };
  assert.deepEqual(drive(), drive());
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
  assert.match(page, /<html lang="en">/);
  assert.match(page, /<meta name="viewport"/);
  assert.match(page, /<meta name="description" content="[^"]{80,}"/, "the description is about this page");

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

test("every id on the page is unique, and the ledger is its own ordered list", () => {
  const page = readPage();
  const ids = [...page.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
  const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
  assert.deepEqual(duplicates, [], `duplicate ids make getElementById return the wrong element: ${duplicates.join(", ")}`);
  assert.match(page, /<h3 id="effects">/, "§8 keeps the #effects anchor the contents list links to");
  assert.match(page, /<ol id="effect-ledger" class="effect-list"><\/ol>/, "the ledger is its own ordered list");
  assert.match(page, /<output id="effect-count" for="effect-ledger">/, "the count is bound to that list");
  const app = readApp();
  assert.match(app, /renderLedger\(\$\("effect-ledger"\)/, "the renderer fills the list, not the heading");
  assert.doesNotMatch(app, /\$\("effects"\)/, "and never reaches for the heading");
});

test("every control the playground understands exists on the page, with the ids app.js reads", () => {
  const page = readPage();
  for (const id of [
    "plugin-list",
    "runtime-status",
    "inactive",
    "effect-ledger",
    "effect-count",
    "unwind",
    "unwind-title",
    "services",
    "registry",
    "hot-swap-loop",
    "swap-shell",
    "reset",
    "swap-status",
    "session-status",
    "session-tail",
    "mount-fork",
    "unmount-fork",
    "fork-status",
    "fork-effects",
    "fork-unwind",
    "fork-unwind-title",
    "pre-step",
    "run-turn",
    "turn-status",
    "turn-events",
    "goal-cap",
    "goal-cap-value",
    "run-goal",
    "goal-status",
    "playground-limits",
  ]) {
    assert.match(page, new RegExp(`id="${id}"`), `#${id} must exist`);
  }
  for (const decision of ["enter", "reject"]) {
    assert.match(page, new RegExp(`<option value="${decision}"`), `the ${decision} pre-step is selectable`);
  }
  assert.match(page, /<input type="range" id="goal-cap"/);
  assert.match(page, /<select id="pre-step">/);
  assert.match(page, /<script type="module" src="\.\/app\.js">/, "the renderer is loaded as a module");
  // The honesty labels are on the page, not only in the module.
  assert.match(page, /What this does not model/);
  assert.match(page, /No model is called/i);
  assert.match(page, /arXiv 2608\.25512/);
});

test("the renderer holds no numbers of its own: app.js imports them from runtime.js", () => {
  const app = readApp();
  assert.match(app, /from "\.\/runtime\.js"/, "the renderer takes its model from runtime.js");
  for (const literal of ["28", "13 ", "300", "190", "220", "200", "256", "1,820"]) {
    assert.ok(!app.includes(literal), `app.js must not hardcode ${literal}`);
  }
  assert.match(app, /DEFAULT_ROUND_CAP/, "the round cap default is imported, not typed");
  // The counts printed come out of the returned state, not out of a sentence typed by hand.
  assert.match(app, /number\(counts\.registrations\)/, "registrations are counted from the state");
  assert.match(app, /number\(counts\.mounted\)/);
  assert.match(app, /state\.inactive/, "the waiting list is the state's");
  assert.match(app, /state\.registry/, "the schema cost is the state's");
});

test("the page states the corrections it owes the reader", () => {
  const page = readPage();
  assert.match(page, /a turn is not a Cordis subcontext/i, "the turn/subcontext correction is explicit");
  assert.match(page, /there is no <code>ctx\.fork\(\)<\/code>/i, "and the fork correction");
  assert.match(page, /not sandboxing, and it is not replay/i, "spatiotemporal is not misread as sandboxing or replay");
  assert.match(page, /network access and process visibility remain outside its guarantees/i, "the sandbox's limit is quoted");
  assert.match(page, /developer preview/i);
  assert.match(page, /HMR is config-only by default|config-only by default in <code>dsh-base<\/code>/i);
});

// ---------------------------------------------------------------------------------------------
// The browser: every control, really clicked, really changing the page
// ---------------------------------------------------------------------------------------------

const browserBinary = ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"].find((path) => existsSync(path));

test("harness playground: every control changes the ledger, the services or the numbers", { skip: !browserBinary && "no chromium on this machine" }, async () => {
  const { serve } = await import("../scripts/serve.mjs");
  const { launch } = await import("./lib/cdp.mjs");
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: [browserBinary], width: 1280, height: 1000 });
    const requested = [];
    page.on("Network.requestWillBeSent", (params) => requested.push(params.request.url));
    await page.send("Network.enable");
    await page.goto(url + "harness/", { label: "the harness page", timeout: 60000 });

    const snapshot = () =>
      page.evaluate(() => {
        const text = (selector) => document.querySelector(selector).textContent;
        const rows = (selector) =>
          [...document.querySelectorAll(`${selector} .effect`)].map((row) => ({
            seq: Number(row.dataset.seq),
            plugin: row.dataset.plugin,
            label: row.querySelector(".effect-label").textContent,
          }));
        return {
          status: text("#runtime-status"),
          effectCount: text("#effect-count"),
          effectRows: rows("#effect-ledger"),
          ledgerTag: document.querySelector("#effect-ledger").tagName,
          ledgerChildren: document.querySelectorAll("#effect-ledger > li").length,
          headingTag: document.querySelector("#effects").tagName,
          headingChildren: document.querySelector("#effects").children.length,
          headingText: document.querySelector("#effects").textContent,
          unwind: rows("#unwind"),
          unwindTitle: text("#unwind-title"),
          pluginStates: [...document.querySelectorAll("#plugin-list button[data-plugin]")].map((button) => ({
            id: button.dataset.plugin,
            pressed: button.getAttribute("aria-pressed"),
            text: button.textContent,
          })),
          services: text("#services"),
          registry: text("#registry"),
          inactive: text("#inactive"),
          turn: text("#turn-status"),
          turnEvents: [...document.querySelectorAll("#turn-events .event-name")].map((node) => node.textContent),
          session: text("#session-status"),
          sessionTail: [...document.querySelectorAll("#session-tail .event-name")].map((node) => node.textContent),
          goal: text("#goal-status"),
          swap: text("#swap-status"),
          fork: text("#fork-status"),
          forkRows: rows("#fork-effects"),
          forkUnwind: rows("#fork-unwind"),
        };
      });

    const registrations = async () => Number((await page.evaluate(() => document.querySelector("#effect-count").textContent)).replace(/,/g, ""));
    const waitForCount = (wanted, label) =>
      page.waitFor(
        (count) => Number(document.querySelector("#effect-count").textContent.replace(/,/g, "")) === count,
        { args: [wanted], label: label ?? `the ledger to settle at ${wanted}`, timeout: 20000 },
      );
    const clickPlugin = (id) => page.click(`#plugin-list button[data-plugin="${id}"]`);
    const ledgerSeqs = () =>
      page.evaluate(() => [...document.querySelectorAll("#effect-ledger > li")].map((row) => Number(row.dataset.seq)));
    // Watch the live ledger itself while it unwinds, with a MutationObserver started before the
    // click. The rows must leave it one at a time, and in the order the model unwound them: newest
    // first. This is the temporal claim made visible, not something read back from the record.
    const watchUnwind = async (id, expected) => {
      const before = await ledgerSeqs();
      await page.evaluate((count) => {
        window.__removed = [];
        window.__removalBatches = [];
        const target = document.getElementById("effect-ledger");
        const observer = new MutationObserver((records) => {
          for (const record of records) {
            const seqs = [...record.removedNodes].map((node) => Number(node.dataset?.seq)).filter((seq) => Number.isFinite(seq));
            if (seqs.length) window.__removalBatches.push(seqs.length);
            for (const seq of seqs) {
              if (window.__removed.length < count) window.__removed.push(seq);
            }
          }
          // Stop before the finishing re-render, which clears what is left of the list.
          if (window.__removed.length >= count) observer.disconnect();
        });
        observer.observe(target, { childList: true });
      }, expected);
      await clickPlugin(id);
      return {
        before,
        removed: () => page.evaluate(() => window.__removed),
        batches: () => page.evaluate(() => window.__removalBatches),
      };
    };
    const setSelect = (id, value) =>
      page.evaluate((controlId, wanted) => {
        const control = document.getElementById(controlId);
        control.value = wanted;
        control.dispatchEvent(new Event("input", { bubbles: true }));
        control.dispatchEvent(new Event("change", { bubbles: true }));
        return control.value;
      }, id, value);

    await page.waitFor(() => document.querySelector("#runtime-status").textContent.includes("plugins mounted"), {
      label: "the composition to render",
      timeout: 30000,
    });

    // First paint: the whole composition is published, not hidden behind a click.
    const initial = await snapshot();
    const defaultMounted = PLUGINS.filter((plugin) => plugin.defaultMounted).length;
    assert.match(initial.status, new RegExp(`^${defaultMounted} plugins mounted`), `saw: ${initial.status}`);
    assert.equal(Number(initial.effectCount.replace(/,/g, "")), initial.effectRows.length);
    // The ledger is an ordered list of registrations, not a heading that swallowed them, and the
    // count on the page is the model's own.
    assert.equal(initial.ledgerTag, "OL", "the ledger renders into its own <ol>");
    assert.equal(initial.ledgerChildren, initial.effectRows.length, "every row is a direct <li> of that list");
    assert.equal(initial.headingTag, "H3", "§8's heading keeps the #effects anchor");
    assert.equal(initial.headingChildren, 0, "and nothing was appended into the heading");
    assert.match(initial.headingText, /^8\. /, "the heading still reads as a heading");
    assert.equal(initial.effectRows.length, initialRuntime().counts.registrations, "the rendered ledger is the model's");
    assert.match(initial.services, /ctx\.shell/);
    assert.match(initial.services, /dsh-bash-sandbox/, "the default ctx.shell provider is named");
    assert.match(initial.registry, /1 tools · bash/, "the tool registry is the model's");
    assert.match(initial.swap, /generation 1/);
    assert.match(initial.fork, /ctx\.shell resolves to/);
    const initialCount = initial.effectRows.length;

    // 1. Unmount the shell provider: its consumer deactivates first, then the provider unwinds.
    const consumerBefore = initial.effectRows.filter((row) => row.plugin === "tool-bash").map((row) => row.seq);
    const providerBefore = initial.effectRows.filter((row) => row.plugin === "bash-sandbox").map((row) => row.seq);
    const unwound = initialCount - consumerBefore.length - providerBefore.length;
    const watch = await watchUnwind("bash-sandbox", 5);
    await waitForCount(unwound);
    assert.deepEqual(await watch.removed(), [...watch.before].slice(-5).reverse(), "the live ledger lost its registrations newest-first");
    const batches = await watch.batches();
    assert.ok(batches.length >= 3, `they left one at a time rather than in one re-render, saw ${batches.length} batches`);
    assert.deepEqual(await ledgerSeqs(), watch.before.slice(0, unwound), "and what is left is exactly the older rows");
    const withoutShell = await snapshot();
    const unwindOrder = withoutShell.unwind.map((row) => row.plugin);
    assert.deepEqual(unwindOrder, ["tool-bash", "tool-bash", "tool-bash", "bash-sandbox", "bash-sandbox"]);
    assert.deepEqual(
      withoutShell.unwind.slice(0, 3).map((row) => row.seq),
      [...consumerBefore].reverse(),
      "the consumer's inverses ran newest-first",
    );
    assert.match(withoutShell.inactive, /dsh-tool-bash is mounted and inactive/, "the dependent is named");
    assert.match(withoutShell.inactive, /ctx\.shell/, "and the missing coeffect is named");
    assert.doesNotMatch(withoutShell.registry, /bash/, "its tool left the registry with it");
    assert.equal(
      withoutShell.pluginStates.find((plugin) => plugin.id === "tool-bash").pressed,
      "true",
      "the consumer is still mounted: it is inactive, not uninstalled",
    );

    // Remount it: the dependent reactivates and both re-register at the end of the ledger.
    await clickPlugin("bash-sandbox");
    await waitForCount(initialCount);
    const restored = await snapshot();
    assert.deepEqual(restored.unwind, [], "a plain mount unwinds nothing, so the inverse ledger clears");
    const restoredSeqs = restored.effectRows.filter((row) => row.plugin === "tool-bash").map((row) => row.seq);
    assert.ok(Math.min(...restoredSeqs) > Math.max(...consumerBefore), "reactivation re-registered, it did not resume");
    assert.match(restored.registry, /bash/, "and the tool is back in the registry");

    // 2. Swap the shell seam: same tool, new provider, the consumer's rows move.
    const beforeSwap = await snapshot();
    await page.click("#swap-shell");
    await waitForCount(initialCount);
    await page.waitFor(() => document.querySelector("#services").textContent.includes("dsh-bash-local"), {
      label: "ctx.shell to resolve to the local provider",
      timeout: 20000,
    });
    const swapped = await snapshot();
    assert.match(swapped.services, /ctx\.shell[\s\S]*dsh-bash-local/);
    assert.match(swapped.registry, /1 tools · bash/, "dsh-tool-bash is unchanged: same tool, same count");
    const swappedSeqs = swapped.effectRows.filter((row) => row.plugin === "tool-bash").map((row) => row.seq);
    const priorSeqs = beforeSwap.effectRows.filter((row) => row.plugin === "tool-bash").map((row) => row.seq);
    assert.ok(Math.min(...swappedSeqs) > Math.max(...priorSeqs), "the consumer unloaded and reloaded under the new provider");
    assert.equal(swapped.effectRows.length, initialCount, "the ledger size did not change");

    // 3. Hot-swap the loop: the generation moves, the session log does not.
    const beforeSwapLog = swapped.sessionTail.length;
    const sessionCount = Number(swapped.session.replace(/[^\d]/g, ""));
    await page.click("#hot-swap-loop");
    await waitForCount(initialCount);
    await page.waitFor(() => document.querySelector("#swap-status").textContent.includes("generation 2"), {
      label: "the new loop generation",
      timeout: 20000,
    });
    const afterHotSwap = await snapshot();
    assert.match(afterHotSwap.swap, /generation 2/);
    assert.match(afterHotSwap.swap, /3 registrations unwound and 3 re-registered/, "the loop's own registrations moved");
    assert.match(afterHotSwap.swap, new RegExp(`session log ${sessionCount} events before the swap, ${sessionCount} after`));
    assert.equal(afterHotSwap.sessionTail.length, beforeSwapLog, "the log's tail is unchanged");

    // 4. A rejected pre-step: a durable turn with zero steps, and the log still grows.
    assert.equal(await setSelect("pre-step", "reject"), "reject");
    await page.click("#run-turn");
    await page.waitFor(() => document.querySelector("#turn-status").textContent.includes("0 steps"), {
      label: "the rejected turn to render",
      timeout: 20000,
    });
    const rejected = await snapshot();
    assert.match(rejected.turn, /agent\/pre-step → reject · 0 steps · 0 model requests/);
    assert.match(rejected.turn, /tool call/);
    assert.deepEqual(rejected.turnEvents.slice(0, 2), ["turn/start", "agent/pre-step"]);
    assert.equal(rejected.turnEvents.at(-1), "turn/end", "the turn opens and closes with no step between");
    assert.ok(!rejected.turnEvents.includes("step/start"));
    const afterRejectCount = Number(rejected.session.replace(/[^\d]/g, ""));
    assert.ok(afterRejectCount > sessionCount, "the rejected turn is durable: the log grew");

    // 5. An accepted one: one step, one request, routed through the current provider.
    assert.equal(await setSelect("pre-step", "enter"), "enter");
    await page.click("#run-turn");
    await page.waitFor(() => document.querySelector("#turn-status").textContent.includes("1 step"), {
      label: "the accepted turn to render",
      timeout: 20000,
    });
    const entered = await snapshot();
    assert.match(entered.turn, /1 step · 1 model request · 1 tool call · shell: @deepseek-ai\/dsh-bash-local/);
    assert.ok(entered.turnEvents.includes("llm/stream"), "the flow shows the model call");
    assert.ok(entered.turnEvents.includes("tool/result"), "and the tool result");

    // 6. The goal: rounds until the cap, then the round-limit blocker.
    await page.evaluate(() => {
      const control = document.getElementById("goal-cap");
      control.value = "5";
      control.dispatchEvent(new Event("input", { bubbles: true }));
    });
    assert.equal(await page.evaluate(() => document.querySelector("#goal-cap-value").textContent), "5", "the output follows the range");
    await page.click("#run-goal");
    await page.waitFor(() => document.querySelector("#goal-status").textContent.includes("blocked"), {
      label: "the goal to block at its cap",
      timeout: 20000,
    });
    const goal = await snapshot();
    assert.match(goal.goal, /phase blocked · 5 of 5 rounds ran/);
    assert.match(goal.goal, /round-limit/, "the blocker code is named");
    assert.match(goal.goal, /5 steps owed to the goal/, "one step per round");

    // 7. The isolated scope: mount and unmount it, and the root ledger never moves.
    const rootBeforeFork = await registrations();
    await page.click("#mount-fork");
    await page.waitFor(() => document.querySelector("#fork-effects").children.length === 2, {
      label: "the isolated registration to render",
      timeout: 20000,
    });
    const forked = await snapshot();
    assert.match(forked.fork, /ctx\.isolate\("shell"\)/);
    assert.match(forked.fork, /root ledger \(\d+ registrations\)/);
    assert.equal(forked.forkRows.length, 2, "the isolated scope has the provider's registrations");
    assert.equal(await registrations(), rootBeforeFork, "the root ledger did not change");
    await page.click("#unmount-fork");
    await page.waitFor(() => document.querySelector("#fork-unwind").children.length === 2, {
      label: "the isolated inverse to run",
      timeout: 20000,
    });
    const unforked = await snapshot();
    assert.match(unforked.fork, /the root ledger still has \d+ registrations/);
    assert.match(unforked.fork, /ctx\.shell still resolves to @deepseek-ai\/dsh-bash-local/);
    assert.equal(await registrations(), rootBeforeFork, "still unchanged after the unmount");

    // 8. A consumer mounted before its provider: PENDING, then active.
    const beforeFs = await snapshot();
    await clickPlugin("tool-fs");
    await page.waitFor(() => document.querySelector("#inactive").textContent.includes("dsh-tool-fs"), {
      label: "the consumer to wait for its service",
      timeout: 20000,
    });
    const waiting = await snapshot();
    assert.match(waiting.inactive, /no active provider for ctx\.fs/);
    assert.equal(waiting.registry, beforeFs.registry, "a pending consumer registers nothing");
    assert.equal(waiting.effectRows.length, beforeFs.effectRows.length);
    await clickPlugin("fs-local");
    await page.waitFor(() => document.querySelector("#registry").textContent.includes("read, edit, write"), {
      label: "the file tools to register",
      timeout: 20000,
    });
    const satisfied = await snapshot();
    assert.match(satisfied.inactive, /Every mounted plugin has the services it injects/);
    assert.match(satisfied.registry, /4 tools/);
    assert.ok(satisfied.effectRows.length > waiting.effectRows.length, "two plugins' registrations joined the ledger");
    await clickPlugin("tool-fs");
    await clickPlugin("fs-local");
    await waitForCount(initialCount);

    // 9. Reboot: the composition comes back exactly as it booted.
    await page.click("#reset");
    await page.waitFor(
      () => document.querySelector('#plugin-list button[data-plugin="bash-sandbox"]').getAttribute("aria-pressed") === "true",
      { label: "the composition to reboot", timeout: 20000 },
    );
    const rebooted = await snapshot();
    assert.equal(rebooted.effectRows.length, initialCount);
    assert.match(rebooted.swap, /generation 1/, "the loop generation is back to the shipped one");
    assert.match(rebooted.services, /dsh-bash-sandbox/, "and ctx.shell is back on the shipped provider");

    // 10. Every plugin button reflects its own mount state.
    const states = (await snapshot()).pluginStates;
    assert.equal(states.length, PLUGINS.length, "one button per package");
    for (const plugin of PLUGINS) {
      const state = states.find((entry) => entry.id === plugin.id);
      assert.ok(state, `${plugin.id} has a button`);
      assert.equal(state.pressed, plugin.defaultMounted ? "true" : "false", `${plugin.id}'s pressed state`);
    }

    // A phone-width column must not push the page sideways; the ledger's long identifiers are the
    // usual cause.
    await page.emulateViewport({ width: 390, height: 844 });
    const narrow = await page.evaluate(() => ({
      client: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth,
    }));
    assert.equal(narrow.scroll <= narrow.client, true, `no horizontal overflow at 390px (${narrow.scroll} vs ${narrow.client})`);
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
