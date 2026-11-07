// @ts-check
/**
 * The playground's DOM half. Every number and every state word comes out of runtime.js; this file
 * only moves them onto the page. Nothing here holds a count of its own, which is what makes a label
 * that stops tracking its control a bug in one of the two files rather than a stale screenshot.
 *
 * Unmounting is animated because the order IS the claim: registrations come off in reverse
 * registration order. Rows are removed one per tick from the live ledger, and the finished state is
 * rendered from the reducer — the animation never invents a row the model did not unwind.
 */
import {
  DEFAULT_ROUND_CAP,
  EVENT_DOMAIN,
  PLUGINS,
  SERVICE_KEYS,
  hotSwapLoop,
  initialRuntime,
  mount,
  mountFork,
  runGoal,
  runTurn,
  swapShell,
  unmount,
  unmountFork,
} from "./runtime.js";

const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
const number = (value) => value.toLocaleString("en-US");

/** @type {ReturnType<typeof initialRuntime>} */
let state = initialRuntime();
let queue = [];
let running = false;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function clear(element) {
  while (element.firstChild) element.removeChild(element.firstChild);
}

// --------------------------------------------------------------------------------------------
// The ledger
// --------------------------------------------------------------------------------------------

function ledgerRow(row) {
  const item = node("li", "effect");
  item.dataset.seq = String(row.seq);
  item.dataset.plugin = row.plugin;
  item.dataset.scope = row.scope ?? "root";
  const head = node("p", "effect-head");
  head.append(
    node("span", "effect-seq", `#${row.seq}`),
    node("span", "effect-plugin", row.pkg),
    node("span", "effect-label", row.label),
  );
  item.append(head, node("p", "effect-inverse", `unwinds to: ${row.inverse}`));
  return item;
}

function renderLedger(target, rows) {
  clear(target);
  for (const row of rows) target.append(ledgerRow(row));
}

function renderUnwind(title, target, record) {
  if (!record) {
    title.textContent = "Nothing has unwound yet.";
    clear(target);
    return;
  }
  const count = record.rows.length;
  title.textContent = `${record.reason} · ${number(count)} registration${count === 1 ? "" : "s"} unwound, in this order:`;
  renderLedger(target, record.rows);
}

// --------------------------------------------------------------------------------------------
// The panels
// --------------------------------------------------------------------------------------------

function renderStatus() {
  const { counts } = state;
  $("runtime-status").textContent =
    `${number(counts.mounted)} plugins mounted · ${number(counts.active)} active · ` +
    `${number(counts.inactive)} waiting on a service · ${number(counts.registrations)} registrations live`;
  $("effect-count").textContent = number(counts.registrations);
}

function renderServices() {
  const target = $("services");
  clear(target);
  for (const key of SERVICE_KEYS) {
    const provider = state.services[key];
    const term = node("dt", null, `ctx.${key}`);
    const value = node("dd", provider ? null : "muted");
    if (provider) {
      const plugin = PLUGINS.find((entry) => entry.id === provider);
      value.textContent = plugin ? plugin.pkg : provider;
    } else {
      value.textContent = "no provider";
    }
    target.append(term, value);
  }
}

function renderPlugins() {
  const target = $("plugin-list");
  clear(target);
  for (const plugin of PLUGINS) {
    const mounted = state.mounted.includes(plugin.id);
    const active = state.active.includes(plugin.id);
    const waiting = state.inactive.find((entry) => entry.id === plugin.id);
    const button = node("button", "plugin-toggle");
    button.type = "button";
    button.dataset.plugin = plugin.id;
    button.setAttribute("aria-pressed", mounted ? "true" : "false");
    button.append(
      node("span", "plugin-pkg", plugin.pkg),
      node("span", "plugin-state", active ? "active" : waiting ? `waiting for ctx.${waiting.waitingFor.join(", ctx.")}` : "not mounted"),
      node("span", "plugin-role", plugin.role),
    );
    target.append(button);
  }
}

function renderRegistry() {
  const { tools, schemaTokens } = state.registry;
  const names = tools.map((tool) => tool.name);
  $("registry").textContent = tools.length
    ? `${number(tools.length)} tools · ${names.join(", ")} · ${number(schemaTokens)} schema tokens in the request`
    : "no tools registered: ctx.tools is mounted, but no consumer contributed a schema";
}

function renderInactive() {
  const waiting = state.inactive;
  $("inactive").textContent = waiting.length
    ? waiting
        .map((entry) => {
          const plugin = PLUGINS.find((candidate) => candidate.id === entry.id);
          const keys = entry.waitingFor.map((key) => `ctx.${key}`).join(", ");
          return `${plugin.pkg} is mounted and inactive: no active provider for ${keys}. Its registrations are not in the ledger.`;
        })
        .join(" ")
    : "Every mounted plugin has the services it injects.";
}

function renderTurn() {
  const turn = state.turn;
  const target = $("turn-events");
  clear(target);
  if (!turn) {
    $("turn-status").textContent = "No turn has run since this page loaded.";
    return;
  }
  $("turn-status").textContent =
    `${turn.id} · agent/pre-step → ${turn.preStep} · ${number(turn.steps)} step${turn.steps === 1 ? "" : "s"} · ` +
    `${number(turn.requests)} model request${turn.requests === 1 ? "" : "s"} · ` +
    `${number(turn.toolCalls)} tool call${turn.toolCalls === 1 ? "" : "s"} · shell: ${turn.provider ?? "none"}`;
  for (const entry of turn.flow) {
    const item = node("li", `event event-${EVENT_DOMAIN[entry.name] ?? "agent"}`);
    item.append(node("span", "event-name", entry.name), node("span", "event-text", entry.text));
    target.append(item);
  }
}

function renderSession() {
  const log = state.sessionLog;
  $("session-status").textContent = `${number(log.length)} durable events in the log`;
  const target = $("session-tail");
  clear(target);
  for (const entry of log.slice(-4)) {
    const item = node("li", "event event-session");
    item.append(node("span", "event-name", entry.type), node("span", "event-text", entry.text));
    target.append(item);
  }
}

function renderGoal() {
  const goal = state.goal;
  const target = $("goal-status");
  if (!goal) {
    target.textContent = `No goal has run. dsh-goal's default cap is ${number(DEFAULT_ROUND_CAP)} rounds; the control sets a lower one so the block is reachable here.`;
    return;
  }
  if (goal.phase === "inactive") {
    target.textContent = "The goal service or its round driver is not active, so nothing would schedule a round. Mount both and run it again.";
    return;
  }
  target.textContent =
    `phase ${goal.phase} · ${number(goal.rounds)} of ${number(goal.cap)} rounds ran · ${number(goal.steps)} steps owed to the goal · ` +
    `blocker ${goal.blocker.code}: “${goal.blocker.message}”`;
}

function renderSwap() {
  const target = $("swap-status");
  const generation = state.loopGeneration;
  if (!state.swapRecord) {
    target.textContent = `dsh-agent-loop is mounted at generation ${number(generation)}. A module swap unloads the fiber and mounts the new code in the same context.`;
    return;
  }
  const record = state.swapRecord;
  target.textContent =
    `generation ${number(record.to)} · ${number(record.unwound)} registrations unwound and ${number(record.remounted)} re-registered · ` +
    `session log ${number(record.logBefore)} events before the swap, ${number(record.logAfter)} after`;
}

function renderFork() {
  const status = $("fork-status");
  const ledger = $("fork-effects");
  if (!state.fork) {
    status.textContent = state.forkUnwind
      ? `The isolated scope is gone; the root ledger still has ${number(state.counts.registrations)} registrations and ctx.shell still resolves to ${state.services.shell ? PLUGINS.find((p) => p.id === state.services.shell).pkg : "nothing"}.`
      : `No isolated scope is mounted. ctx.shell resolves to ${state.services.shell ? PLUGINS.find((p) => p.id === state.services.shell).pkg : "nothing"} for the whole tree.`;
    clear(ledger);
    return;
  }
  status.textContent = `ctx.isolate("shell") below one child context mounts ${PLUGINS.find((p) => p.id === state.fork.plugin).pkg} there. The root provider is unchanged, and so is the root ledger (${number(state.counts.registrations)} registrations).`;
  renderLedger(ledger, state.fork.effects);
}

function renderAll() {
  renderStatus();
  renderServices();
  renderPlugins();
  renderLedger($("effects"), state.effects);
  renderUnwind($("unwind-title"), $("unwind"), state.unwind);
  renderRegistry();
  renderInactive();
  renderTurn();
  renderSession();
  renderGoal();
  renderSwap();
  renderFork();
  renderUnwind($("fork-unwind-title"), $("fork-unwind"), state.forkUnwind);
}

// --------------------------------------------------------------------------------------------
// Actions
// --------------------------------------------------------------------------------------------

/** Remove the unwound rows one tick at a time, from the live ledger, then render the new state. */
function animate(rows, done) {
  let index = 0;
  const step = () => {
    const row = rows[index];
    if (!row) {
      done();
      return;
    }
    const scope = row.scope === "fork" ? "fork-effects" : "effects";
    const target = $(scope).querySelector(`[data-seq="${row.seq}"]`);
    if (target) target.remove();
    index += 1;
    setTimeout(step, 35);
  };
  step();
}

function drain() {
  if (running) return;
  const next = queue.shift();
  if (!next) return;
  const result = next();
  const rows = result.unwind?.rows ?? [];
  state = result;
  // The record panels are rendered from the finished state; the ledger animation runs over the
  // rows the reducer actually unwound.
  running = true;
  if (rows.length) {
    animate(rows, () => {
      running = false;
      renderAll();
      drain();
    });
  } else {
    running = false;
    renderAll();
    drain();
  }
}

function act(mutator) {
  queue.push(() => mutator());
  drain();
}

// --------------------------------------------------------------------------------------------
// Wiring
// --------------------------------------------------------------------------------------------

$("plugin-list").addEventListener("click", (event) => {
  const button = /** @type {HTMLElement} */ (event.target).closest("button[data-plugin]");
  if (!button) return;
  const id = /** @type {HTMLElement} */ (button).dataset.plugin;
  if (!id) return;
  act(() => (state.mounted.includes(id) ? unmount(state, id) : mount(state, id)));
});

$("swap-shell").addEventListener("click", () => {
  act(() => swapShell(state, state.services.shell === "bash-local" ? "bash-sandbox" : "bash-local"));
});

$("hot-swap-loop").addEventListener("click", () => {
  act(() => {
    const record = {
      to: state.loopGeneration + 1,
      unwound: state.effects.filter((row) => row.plugin === "agent-loop").length,
      remounted: PLUGINS.find((plugin) => plugin.id === "agent-loop").effects.length,
      logBefore: state.sessionLog.length,
    };
    const next = hotSwapLoop(state);
    const remounted = next.effects.filter((row) => row.plugin === "agent-loop").length;
    next.swapRecord = { ...record, remounted, logAfter: next.sessionLog.length };
    return next;
  });
});

$("mount-fork").addEventListener("click", () => act(() => mountFork(state)));
$("unmount-fork").addEventListener("click", () => act(() => unmountFork(state)));

$("run-turn").addEventListener("click", () => {
  const preStep = /** @type {HTMLSelectElement} */ ($("pre-step")).value === "reject" ? "reject" : "enter";
  act(() => runTurn(state, { preStep }));
});

$("goal-cap").addEventListener("input", () => {
  $("goal-cap-value").textContent = $("goal-cap").value;
});

$("run-goal").addEventListener("click", () => {
  const cap = Number(/** @type {HTMLInputElement} */ ($("goal-cap")).value);
  act(() => runGoal(state, { cap }));
});

$("reset").addEventListener("click", () => {
  queue = [];
  state = initialRuntime();
  renderAll();
});

$("goal-cap-value").textContent = $("goal-cap").value;
renderAll();
