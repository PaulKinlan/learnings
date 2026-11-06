// @ts-check
/**
 * The plugin explorer's DOM half. All arithmetic lives in plugins.js; this file reads the controls,
 * asks the model for a graph, and renders it. Every number printed comes out of the returned run,
 * which is what keeps the labels derived rather than typed.
 *
 * The one piece of state that is not a control on the page is the load order of the plugins: it is
 * the order the reader pressed the buttons in, and the order the hooks would dispatch in.
 */
import { PLUGINS, explore } from "./plugins.js";

const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
const number = (value) => value.toLocaleString("en-US");
const plural = (count, one, many = `${one}s`) => `${number(count)} ${count === 1 ? one : many}`;

/** The load order. Adding appends; clearing is the explorer's own restart. */
const loaded = [];
let unregisterAttempted = false;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function readControls() {
  return {
    plugins: [...loaded],
    offered: ["read", "bash", "edit", "write", "lint"].filter(
      (id) => /** @type {HTMLInputElement} */ ($(`explore-tool-${id}`)).checked,
    ),
    guard: $("guard").value,
    deactivateTool: $("deactivate-tool").value,
    deactivateAfter: Number(/** @type {HTMLInputElement} */ ($("deactivate-after")).value),
    forkAfter: $("fork").value,
    cut: Number(/** @type {HTMLInputElement} */ ($("cut")).value),
    unregisterAttempt: unregisterAttempted,
  };
}

/** One entry in a session file. `role` decides the marker; the id is the entry's name in the tree. */
function entryRow(entry, options = {}) {
  const item = node("li", `entry entry-${entry.role}`);
  if (options.abandoned) item.classList.add("entry-abandoned");
  if (options.replaced) item.classList.add("entry-replaced");
  item.append(
    node("span", "entry-id", entry.id),
    node("span", "entry-label", entry.label),
    node("span", "entry-tokens", `${number(entry.tokens)} tok`),
  );
  if (entry.outcome && entry.outcome !== "executed") item.append(node("span", "entry-flag", entry.outcome));
  if (options.note) item.append(node("p", "entry-note", options.note));
  return item;
}

/** A session file, with the compaction cut drawn where it falls on the active branch. */
function fileBlock(run, file, isActive) {
  const block = node("section", "file");
  const heading = node("h4", "", `file ${file.id} · ${file.role}`);
  block.append(heading);
  block.append(
    node(
      "p",
      "small",
      file.parentSession
        ? `header names parentSession ${file.parentSession}; ${plural(file.entries.length, "entry", "entries")}, ${number(file.tokens)} tokens`
        : `no parentSession; ${plural(file.entries.length, "entry", "entries")}, ${number(file.tokens)} tokens`,
    ),
  );

  const list = node("ol", "entries");
  const cut = run.compaction.cutIndex;
  // The note explains the edge, so it goes on the first entry that carries the edge rather than on
  // every one of them: a wall of identical sentences is a wall, not an explanation.
  let notedReplaced = false;
  let notedAbandoned = false;
  for (const [index, entry] of file.entries.entries()) {
    if (isActive && index === cut && run.compaction.summaryTokens > 0) {
      const summary = node("li", "entry entry-summary");
      summary.append(
        node("span", "entry-id", "compaction"),
        node(
          "span",
          "entry-label",
          `summary of ${plural(run.compaction.replaced.length, "entry", "entries")} · firstKeptEntryId = ${run.compaction.firstKeptEntryId}`,
        ),
        node("span", "entry-tokens", `${number(run.compaction.summaryTokens)} tok`),
      );
      list.append(summary);
    }
    const abandoned = run.abandoned.includes(entry);
    const replaced = isActive && index < cut;
    let note = "";
    if (abandoned && !notedAbandoned) {
      notedAbandoned = true;
      note = "abandoned by the fork: still in this file, still readable, still in /tree";
    } else if (replaced && !notedReplaced) {
      notedReplaced = true;
      note = "replaced by the summary when the context is rebuilt; the entry stays in the file";
    }
    list.append(entryRow(entry, { abandoned, replaced, note }));
  }
  block.append(list);
  return block;
}

const METRIC_ROWS = [
  ["Model requests", (run) => number(run.metrics.requests)],
  ["Turns", (run) => number(run.metrics.turns)],
  [
    "Tool calls",
    (run) => `${number(run.metrics.calls)} (${number(run.metrics.executed)} ran, ${number(run.metrics.notOffered)} not offered, ${number(run.metrics.deactivated)} deactivated, ${number(run.metrics.blocked)} blocked)`,
  ],
  ["Tools registered", (run) => `${number(run.registry.registeredCount)} · ${run.registry.registered.join(", ")}`],
  ["Tools active (offered)", (run) => `${number(run.registry.offeredCount)} · ${run.registry.offered.join(", ") || "none"}`],
  [
    "Deactivated mid-run",
    (run) =>
      run.deactivation.tool
        ? `${run.deactivation.tool} after turn ${run.deactivation.afterTurn} · still registered, still in the registry`
        : "nothing",
  ],
  ["Prompt tokens (base + schemas)", (run) => number(run.metrics.promptTokens)],
  ["Context before compaction", (run) => number(run.compaction.contextBefore)],
  ["Context after the summary", (run) => number(run.compaction.contextAfter)],
  ["firstKeptEntryId", (run) => run.compaction.firstKeptEntryId ?? "none"],
  [
    "Entries kept / replaced",
    (run) => `${number(run.compaction.kept.length)} / ${number(run.compaction.replaced.length)}`,
  ],
  ["Entries deleted", (run) => number(run.compaction.deleted)],
  [
    "Abandoned branch (still readable)",
    (run) => `${plural(run.fork.abandonedEntries, "entry", "entries")} · ${number(run.fork.abandonedTokens)} tok`,
  ],
];

function render() {
  const run = explore(readControls());

  // The cut is a position in the active branch, so the range's bounds follow the branch: forking
  // into a longer branch gives the reader more places to put it.
  const cutInput = /** @type {HTMLInputElement} */ ($("cut"));
  cutInput.max = String(run.activeBranch.length - 1);
  cutInput.value = String(run.compaction.cutIndex);
  $("cut-value").textContent = run.compaction.firstKeptEntryId ?? "none";

  $("explorer-status").textContent =
    `${plural(run.metrics.requests, "request")}, ${plural(run.metrics.calls, "tool call")}` +
    `${run.metrics.terminated ? `, stopped by a unanimous terminate vote after turn ${run.metrics.turns}` : ""}` +
    `; ${plural(run.plugins.length, "plugin")} loaded, ${plural(run.abandoned.length, "abandoned entry", "abandoned entries")}, ` +
    `firstKeptEntryId ${run.compaction.firstKeptEntryId} keeps ${plural(run.compaction.kept.length, "entry", "entries")}.`;

  const metrics = $("explorer-metrics");
  metrics.replaceChildren();
  for (const [label, value] of METRIC_ROWS) {
    metrics.append(node("dt", "", label), node("dd", "number", value(run)));
  }

  // The load order, and what each plugin's hook did on this run.
  const order = $("plugin-order");
  order.replaceChildren();
  if (!run.plugins.length) {
    order.append(node("li", "small", "Nothing loaded: the buttons above add an extension to the load order."));
  }
  for (const plugin of run.plugins) {
    const item = node("li");
    item.append(
      node("p", "ev-title", `${plugin.order}. ${plugin.label} — on("${plugin.hook}")`),
      node("p", "ev-note", `fires ${plugin.when}: ${number(plugin.fires)} dispatch${plugin.fires === 1 ? "" : "es"} on this run`),
      node("p", "ev-note", plugin.effect),
    );
    order.append(item);
  }
  for (const plugin of PLUGINS) {
    const button = /** @type {HTMLButtonElement} */ ($(`add-${plugin.id}`));
    button.disabled = loaded.includes(plugin.id);
  }

  const tree = $("tree");
  tree.replaceChildren();
  tree.append(fileBlock(run, run.fileA, !run.fileB));
  if (run.fileB) tree.append(fileBlock(run, run.fileB, true));

  $("branch-line").textContent = run.fileB
    ? `Forked after turn ${run.fork.afterTurn}: a new file with ${plural(run.fork.newFileEntries, "entry", "entries")} and ${number(run.fork.newFileTokens)} tokens, built on the ${plural(run.prefix.length, "entry", "entries")} before the fork point. File A is untouched: it still holds all ${plural(run.fileA.entries.length, "entry", "entries")}, including the ${plural(run.fork.abandonedEntries, "entry", "entries")} the new branch left behind. Entries deleted: ${number(run.fork.deleted)}.`
    : `One file, one branch. The model sees the active branch (${plural(run.activeBranch.length, "entry", "entries")}); a fork would write a second file and leave this one alone.`;

  $("cut-line").textContent =
    `Cut at ${run.compaction.firstKeptEntryId}: ${plural(run.compaction.replaced.length, "entry", "entries")} before it are replaced by a ${number(run.compaction.summaryTokens)}-token summary, and the ${plural(run.compaction.kept.length, "entry", "entries")} from there on (${number(run.compaction.keptTokens)} tokens) are sent as they are. Context ${number(run.compaction.contextBefore)} → ${number(run.compaction.contextAfter)}.` +
    (run.compaction.summaryTokens && run.compaction.contextAfter >= run.compaction.contextBefore
      ? " The summary costs more than the entries it replaces at this cut — which is why compaction fires on a token threshold, not on a hunch."
      : "");

  const unregister = $("unregister-result");
  unregister.hidden = !run.unregister.attempted;
  if (run.unregister.attempted) {
    unregister.textContent =
      `${run.unregister.called} → there is no such API: ${run.unregister.hits} hits in this version's dist, docs and examples. ` +
      `Tools registered before: ${number(run.unregister.registeredBefore)}; after: ${number(run.unregister.registeredAfter)}; active: ${number(run.unregister.activeAfter)}. ` +
      "The supported removal is `setActiveTools` deactivation — the tool stays registered, and the next call to it fails — or deleting the file and reloading the whole runtime.";
  }

  const trace = $("explorer-trace");
  trace.replaceChildren();
  for (const event of run.trace) {
    const item = node("li", `ev ev-${event.kind}`);
    switch (event.kind) {
      case "request":
        item.append(
          node(
            "p",
            "ev-title",
            `request #${event.index} · context ${number(event.contextTokens)}${event.closing ? " · closing" : ""}`,
          ),
        );
        if (event.note) item.append(node("p", "ev-note", event.note));
        break;
      case "tool":
        item.append(
          node(
            "p",
            "ev-title",
            event.outcome === "executed"
              ? `${event.tool} ${event.target} → ${number(event.tokens)} tokens`
              : `${event.tool} ${event.target} · ${event.outcome} · isError`,
          ),
        );
        item.append(node("p", "ev-note", event.note));
        break;
      case "hook":
        item.append(
          node("p", "ev-title", `turn_end · turn ${event.turn}`),
          node(
            "p",
            "ev-note",
            event.active ? event.note : "no turn_end handler loaded, so nothing dispatched",
          ),
        );
        break;
      default:
        item.append(node("p", "ev-title", event.text));
    }
    trace.append(item);
  }

  const notes = $("explorer-notes");
  notes.replaceChildren();
  for (const note of run.notes) notes.append(node("li", "", note));
}

for (const plugin of PLUGINS) {
  const button = $(`add-${plugin.id}`);
  button.addEventListener("click", () => {
    if (!loaded.includes(plugin.id)) loaded.push(plugin.id);
    render();
  });
}

$("plugin-clear").addEventListener("click", () => {
  loaded.length = 0;
  render();
});

$("unregister-attempt").addEventListener("click", () => {
  unregisterAttempted = true;
  render();
});

for (const control of document.querySelectorAll(
  "#guard, #deactivate-tool, #deactivate-after, #fork, #cut, input[type=checkbox]",
)) {
  control.addEventListener("input", render);
  control.addEventListener("change", render);
}

render();
