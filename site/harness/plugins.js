// @ts-check
/**
 * The plugin explorer's model: a deterministic plugin-and-execution graph, in the harness's own
 * vocabulary.
 *
 * NOTHING HERE LOADS AN EXTENSION, CALLS A MODEL OR TOUCHES A SESSION FILE. Every number the
 * explorer shows is computed from the controls plus the constants below, and every state it draws
 * is built from the same scripted run. The mechanics modelled:
 *
 *   - an extension registers hooks at load; a hook fires at a fixed point in the loop, and a plugin
 *     added at load time changes every later dispatch (and no dispatch already in progress);
 *   - a tool that is registered but not *active* still exists in the registry: its next call fails,
 *     and nothing in the API removes it;
 *   - a `tool_call` handler that returns `{ block: true }` stops that call; `terminate: true` is a
 *     vote, and the loop only stops when every finalized result in the batch agrees;
 *   - a session is an append-only tree: a fork writes a *new* file from an earlier entry and leaves
 *     the original file, and its abandoned branch, intact and readable;
 *   - compaction adds a summary entry with `firstKeptEntryId`; everything before that entry is
 *     replaced when the context is rebuilt, and nothing is deleted.
 */

/** The task message the scripted run starts from. */
export const TASK_TOKENS = 120;

/** Prompt assembly: the base prompt, plus one JSON Schema per offered tool. */
export const BASE_PROMPT_TOKENS = 900;

/** One assistant message that carries tool calls, and the closing answer. */
export const TURN_TEXT_TOKENS = 230;
export const ANSWER_TEXT_TOKENS = 320;

/** What a result costs when the call did not run. They differ on purpose: the reader should be able
 *  to tell "not offered" from "deactivated" from "blocked" by the number as well as the label. */
export const NOT_OFFERED_TOKENS = 45;
export const DEACTIVATED_TOKENS = 60;
export const BLOCKED_TOKENS = 75;

/** The `tool_result` handler's annotation, added to every result that came back from a tool. */
export const ANNOTATION_TOKENS = 60;

/** The `context` handler's redaction: dropped from each request, not from the transcript. */
export const REDACTION_TOKENS = 180;

/** The `before_agent_start` handler's stamp: it rewrites the run's system prompt once. */
export const STAMP_TOKENS = 220;

/** The summary that replaces everything before `firstKeptEntryId`. */
export const SUMMARY_TOKENS = { base: 300, ratio: 0.18 };

/** The four built-in tools the explorer's run may use. `schemaTokens` is the cost of carrying the
 *  tool's TypeBox-derived JSON Schema in the prompt — the same price the billing playground charges. */
export const EXPLORER_TOOLS = [
  { id: "read", schemaTokens: 220 },
  { id: "bash", schemaTokens: 260 },
  { id: "edit", schemaTokens: 240 },
  { id: "write", schemaTokens: 200 },
];

/** A tool that does not exist until an extension registers it. */
export const LINT_TOOL = { id: "lint", schemaTokens: 190 };

/** The scripted run: four turns of tool calls, then a closing answer. The tool names are the four
 *  built-ins, so every control below has something real to act on. */
export const TURNS = [
  { calls: [{ tool: "read", target: "remote/lane-host.mjs", tokens: 1400 }] },
  {
    calls: [
      { tool: "bash", target: "npm test -- tests/lane-host.test.js", tokens: 900 },
      { tool: "read", target: "pi/loop.ts", tokens: 1100 },
    ],
  },
  { calls: [{ tool: "edit", target: "remote/lane-host.mjs", tokens: 260 }] },
  { calls: [{ tool: "bash", target: "npm test -- tests/lane-host.test.js", tokens: 900 }] },
];

/** The call the run makes only once `lint-tool` has registered a `lint` tool and the run offers it. */
export const LINT_CALL = { tool: "lint", target: "remote/lane-host.mjs", tokens: 620 };

/** The other branch, taken at the fork point: lint first, then read the compaction code. Two turns
 *  and an answer, whatever turn the fork happened at. */
export const ALT_TURNS = [
  { calls: [{ tool: "bash", target: "npm run lint", tokens: 700 }] },
  { calls: [{ tool: "read", target: "pi/compact.ts", tokens: 1500 }] },
];

/** The plugins a reader can add, in the order the buttons appear. `hook` is the hook name, `when`
 *  says where in the loop it fires, and `fires` in the result counts the dispatches. */
export const PLUGINS = [
  {
    id: "turn-logger",
    label: "turn-logger",
    hook: "turn_end",
    when: "after every turn",
    observes: true,
  },
  {
    id: "context-redactor",
    label: "context-redactor",
    hook: "context",
    when: "before every model request",
    observes: false,
  },
  {
    id: "prompt-stamp",
    label: "prompt-stamp",
    hook: "before_agent_start",
    when: "once per user prompt",
    observes: false,
  },
  {
    id: "result-annotator",
    label: "result-annotator",
    hook: "tool_result",
    when: "on every tool result",
    observes: false,
  },
  {
    id: "lint-tool",
    label: "lint-tool",
    hook: "registerTool",
    when: "when the extension loads",
    observes: false,
  },
];

/** The `tool_call` guards. `terminate` is the vote the harness only honours unanimously. */
export const GUARDS = [
  { id: "none", label: "no guard", blocks: [], terminate: false },
  { id: "edit", label: "block `edit`", blocks: ["edit"], terminate: false },
  { id: "bash", label: "block `bash`", blocks: ["bash"], terminate: false },
  { id: "bash-terminate", label: "block `bash`, and vote to terminate", blocks: ["bash"], terminate: true },
  { id: "edit-terminate", label: "block `edit`, and vote to terminate", blocks: ["edit"], terminate: true },
];

/** The fork points, as the turn the fork happens after. */
export const FORK_POINTS = [
  { id: "none", label: "do not fork", afterTurn: 0 },
  { id: "turn-1", label: "after turn 1", afterTurn: 1 },
  { id: "turn-2", label: "after turn 2", afterTurn: 2 },
  { id: "turn-3", label: "after turn 3", afterTurn: 3 },
];

/** What `unregisterTool('edit')` finds in 0.87.1: nothing, because it is not there. */
export const UNREGISTER_PROBE = { called: "unregisterTool('edit')", exists: false, hits: 0 };

const find = (list, id) => list.find((entry) => entry.id === id);
const count = (value) => value.toLocaleString("en-US");
const sumTokens = (entries) => entries.reduce((total, entry) => total + entry.tokens, 0);
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

/**
 * Run the scripted scenario under one configuration and return the graph, the compaction cut and
 * every number the explorer prints.
 *
 * The options are the controls, resolved:
 *   - `plugins` — ids from `PLUGINS`, in load order (the order the reader added them);
 *   - `offered` — the tools the session offers (construction-time allow/deny);
 *   - `guard` — an id from `GUARDS`;
 *   - `deactivateTool`/`deactivateAfter` — the mid-run `setActiveTools` deactivation;
 *   - `forkAfter` — the turn a fork happens after, or 0;
 *   - `cut` — the index of the entry that becomes `firstKeptEntryId`;
 *   - `unregisterAttempt` — whether the reader pressed the button that cannot work.
 */
export function explore(options = {}) {
  const pluginIds = (Array.isArray(options.plugins) ? options.plugins : []).filter((id) => find(PLUGINS, id));
  const has = (id) => pluginIds.includes(id);

  const registered = [...EXPLORER_TOOLS, ...(has("lint-tool") ? [LINT_TOOL] : [])];
  const requestedOffers = options.offered ?? [...EXPLORER_TOOLS.map((tool) => tool.id), LINT_TOOL.id];
  // Offering a name nothing registered offers nothing: there is no schema to send and no tool to
  // call. The ignored names are kept so the page can say so instead of quietly dropping them.
  const offered = requestedOffers.filter((id) => find(registered, id));
  const ignoredOffers = requestedOffers.filter((id) => !find(registered, id));
  const guard = find(GUARDS, options.guard) ?? GUARDS[0];
  const deactivateTool = registered.some((tool) => tool.id === options.deactivateTool) ? options.deactivateTool : "";
  const deactivateAfter = deactivateTool ? clamp(Number(options.deactivateAfter) || 0, 0, TURNS.length) : 0;

  const stampTokens = has("prompt-stamp") ? STAMP_TOKENS : 0;
  const systemTokens = BASE_PROMPT_TOKENS + offered.reduce((total, id) => total + find(registered, id).schemaTokens, 0);
  const promptTokens = systemTokens + stampTokens;
  const redaction = has("context-redactor") ? REDACTION_TOKENS : 0;
  const annotation = has("result-annotator") ? ANNOTATION_TOKENS : 0;

  /** The calls of one scripted turn: the script's own, plus `lint` once something registers it. */
  const callsFor = (turn) => {
    const calls = turn.calls.map((call) => ({ ...call }));
    if (turn === TURNS[TURNS.length - 1] && registered.some((tool) => tool.id === LINT_TOOL.id) && offered.includes(LINT_TOOL.id)) {
      calls.push({ ...LINT_CALL });
    }
    return calls;
  };

  const entries = [
    { id: "u1", parentId: null, role: /** @type {const} */ ("user"), label: "the task", tokens: TASK_TOKENS },
  ];
  const trace = [];
  const metrics = {
    requests: 0,
    turns: 0,
    calls: 0,
    executed: 0,
    notOffered: 0,
    deactivated: 0,
    blocked: 0,
    terminateVotes: 0,
    terminated: false,
    promptTokens,
    schemaTokens: systemTokens - BASE_PROMPT_TOKENS,
    stampTokens,
    redacted: 0,
    annotated: 0,
    contextBefore: 0,
  };

  let leafId = "u1";
  let requestsOfRun = 0;
  /** The last entry of each turn, which is where a fork at that turn branches from. */
  const lastEntryOfTurn = {};

  for (const [index, turn] of TURNS.entries()) {
    const turnNumber = index + 1;
    const calls = callsFor(turn);
    requestsOfRun += 1;
    metrics.requests += 1;
    const contextTokens = promptTokens + sumTokens(entries) - redaction;
    metrics.redacted += redaction;
    trace.push({
      kind: "request",
      index: turnNumber,
      contextTokens,
      note: redaction ? `context handler dropped ${count(redaction)} tokens of a secret block before sending` : "",
    });

    const assistantId = `a${turnNumber}`;
    entries.push({
      id: assistantId,
      parentId: leafId,
      role: "assistant",
      label: `turn ${turnNumber}: ${calls.map((call) => call.tool).join(" + ")}`,
      tokens: TURN_TEXT_TOKENS,
    });
    leafId = assistantId;
    metrics.turns = turnNumber;

    let votes = 0;
    for (const [position, call] of calls.entries()) {
      metrics.calls += 1;
      const deactivatedHere = deactivateTool === call.tool && turnNumber > deactivateAfter;
      const guardHere = !deactivatedHere && guard.blocks.includes(call.tool);
      const offeredHere = offered.includes(call.tool);
      const resultId = `t${turnNumber}.${position + 1}`;
      let tokens = 0;
      let outcome = /** @type {string} */ ("executed");
      let note = "";

      if (!offeredHere) {
        outcome = "not-offered";
        tokens = NOT_OFFERED_TOKENS;
        metrics.notOffered += 1;
        note = `${call.tool} is not in this session's tool set, so it is not in the schema and nothing ran: isError: true`;
      } else if (deactivatedHere) {
        outcome = "deactivated";
        tokens = DEACTIVATED_TOKENS;
        metrics.deactivated += 1;
        note = `${call.tool} is still registered and still in the registry, but it is no longer active: the call fails — deactivated, not unregistered`;
      } else if (guardHere) {
        outcome = "blocked";
        tokens = BLOCKED_TOKENS;
        metrics.blocked += 1;
        if (guard.terminate) {
          votes += 1;
          metrics.terminateVotes += 1;
        }
        note = `the tool_call handler returned { block: true }${
          guard.terminate ? " and terminate: true (one vote)" : ""
        }: the tool never ran and the model got the reason back`;
      } else {
        tokens = call.tokens + annotation;
        metrics.executed += 1;
        metrics.annotated += annotation;
        note = annotation ? `tool_result handler appended ${count(annotation)} tokens to this result` : "";
      }

      tokens = Math.round(tokens);
      entries.push({
        id: resultId,
        parentId: assistantId,
        role: "tool",
        label: `${call.tool} ${call.target}`,
        tokens,
        outcome,
      });
      leafId = resultId;
      trace.push({ kind: "tool", turn: turnNumber, tool: call.tool, target: call.target, outcome, tokens, note });
    }

    lastEntryOfTurn[turnNumber] = leafId;

    trace.push({
      kind: "hook",
      hook: "turn_end",
      turn: turnNumber,
      active: has("turn-logger"),
      note: has("turn-logger") ? "turn-logger ran; it only observes" : "",
    });

    // `terminate` is a vote, and the loop stops only when every finalized result in the batch
    // agreed. A two-call batch with one vote continues; a one-call batch that votes stops the run.
    if (votes > 0 && votes === calls.length) {
      metrics.terminated = true;
      trace.push({
        kind: "note",
        text: `every one of the ${calls.length} finalized result${calls.length === 1 ? "" : "s"} in this batch asked to terminate, so the loop stops here: no more requests, and no closing answer.`,
      });
      break;
    }
    if (votes > 0) {
      trace.push({
        kind: "note",
        text: `${votes} of ${calls.length} finalized results in this batch asked to terminate. One vote is not a veto: the loop continues.`,
      });
    }
  }

  if (!metrics.terminated) {
    metrics.requests += 1;
    requestsOfRun += 1;
    metrics.redacted += redaction;
    trace.push({
      kind: "request",
      index: requestsOfRun,
      contextTokens: promptTokens + sumTokens(entries) - redaction,
      closing: true,
      note: "the closing request: no tool calls, so the loop ends",
    });
    entries.push({
      id: "answer",
      parentId: leafId,
      role: "answer",
      label: "the closing answer",
      tokens: ANSWER_TEXT_TOKENS,
    });
    leafId = "answer";
    trace.push({ kind: "answer", text: "assistant answered without calling a tool" });
    trace.push({ kind: "end", text: "agent_end — this run is closed" });
  }

  // --- the tree, and the fork ----------------------------------------------------------------
  const forkPoint = find(FORK_POINTS, options.forkAfter) ?? FORK_POINTS[0];
  const forkAfterTurn = forkPoint.afterTurn ? Math.min(forkPoint.afterTurn, metrics.turns) : 0;
  const prefixLength =
    forkAfterTurn && lastEntryOfTurn[forkAfterTurn]
      ? entries.findIndex((entry) => entry.id === lastEntryOfTurn[forkAfterTurn]) + 1
      : 0;
  const prefix = entries.slice(0, prefixLength);

  /** The alternative branch, appended to a copy of the prefix in a new file. */
  const altEntries = () => {
    const branch = [];
    let parent = prefix.length ? prefix[prefix.length - 1].id : null;
    for (const [index, turn] of ALT_TURNS.entries()) {
      branch.push({
        id: `b${index + 1}a`,
        parentId: parent,
        role: /** @type {const} */ ("assistant"),
        label: `turn ${forkAfterTurn + index + 1}: ${turn.calls.map((call) => call.tool).join(" + ")}`,
        tokens: TURN_TEXT_TOKENS,
      });
      parent = `b${index + 1}a`;
      for (const [position, call] of turn.calls.entries()) {
        branch.push({
          id: `b${index + 1}t${position + 1}`,
          parentId: parent,
          role: /** @type {const} */ ("tool"),
          label: `${call.tool} ${call.target}`,
          tokens: call.tokens,
          outcome: "executed",
        });
        parent = `b${index + 1}t${position + 1}`;
      }
    }
    branch.push({
      id: "banswer",
      parentId: parent,
      role: /** @type {const} */ ("answer"),
      label: "the closing answer",
      tokens: ANSWER_TEXT_TOKENS,
    });
    return branch;
  };

  const forked = prefixLength > 0 ? altEntries() : [];
  const fileA = {
    id: "A",
    role: "the session that ran",
    entries,
    leafId,
    tokens: sumTokens(entries),
  };
  const fileB = forked.length
    ? {
        id: "B",
        role: `forked after turn ${forkAfterTurn}`,
        parentSession: "A",
        entries: [...prefix, ...forked],
        leafId: forked[forked.length - 1].id,
        tokens: sumTokens([...prefix, ...forked]),
      }
    : null;

  const activeBranch = fileB ? fileB.entries : fileA.entries;
  // Entries the active branch left behind in file A. The fork wrote a new file; it did not edit
  // this one, so the abandoned branch is still there, in full.
  const abandoned = fileB ? fileA.entries.slice(prefixLength) : [];

  // --- compaction ----------------------------------------------------------------------------
  const cut = clamp(Number(options.cut ?? 2) || 0, 0, activeBranch.length - 1);
  const replaced = activeBranch.slice(0, cut);
  const kept = activeBranch.slice(cut);
  const replacedTokens = sumTokens(replaced);
  const keptTokens = sumTokens(kept);
  const summaryTokens = replaced.length ? Math.round(SUMMARY_TOKENS.base + SUMMARY_TOKENS.ratio * replacedTokens) : 0;
  const contextBefore = promptTokens + sumTokens(activeBranch) - redaction;
  const contextAfter = promptTokens + summaryTokens + keptTokens - redaction;
  const compaction = {
    cutIndex: cut,
    firstKeptEntryId: kept.length ? kept[0].id : null,
    kept,
    replaced,
    keptTokens,
    replacedTokens,
    summaryTokens,
    contextBefore,
    contextAfter,
    entriesBefore: activeBranch.length,
    entriesInFileAfter: activeBranch.length + (replaced.length ? 1 : 0),
    deleted: 0,
  };
  metrics.contextBefore = contextBefore;
  metrics.contextAfter = contextAfter;

  // --- the plugins, now that the run has happened ---------------------------------------------
  const plugins = pluginIds.map((id, order) => {
    const plugin = find(PLUGINS, id);
    const record = { ...plugin, order: order + 1, fires: 0, effect: "" };
    switch (id) {
      case "turn-logger":
        record.fires = metrics.turns;
        record.effect = `${metrics.turns} turn_end dispatch${metrics.turns === 1 ? "" : "es"}, one trace line each; it observes, so nothing else moves`;
        break;
      case "context-redactor":
        record.fires = metrics.requests;
        record.effect = `dropped ${count(REDACTION_TOKENS)} tokens from each of ${metrics.requests} requests: ${count(metrics.redacted)} context tokens never sent`;
        break;
      case "prompt-stamp":
        record.fires = 1;
        record.effect = `rewrote the run's system prompt once, before turn 1: +${count(STAMP_TOKENS)} tokens on every request, ${count(STAMP_TOKENS * metrics.requests)} in total`;
        break;
      case "result-annotator":
        record.fires = metrics.executed;
        record.effect = `appended ${count(ANNOTATION_TOKENS)} tokens to ${metrics.executed} tool result${metrics.executed === 1 ? "" : "s"}: ${count(metrics.annotated)} extra context tokens`;
        break;
      case "lint-tool":
        record.fires = 1;
        record.effect = offered.includes(LINT_TOOL.id)
          ? `registered \`lint\` at load (+${count(LINT_TOOL.schemaTokens)} schema tokens) and the run called it`
          : `registered \`lint\` at load, but this session does not offer it: registered is not the same as active`;
        break;
      default:
        record.effect = "";
    }
    return record;
  });

  const notes = [];
  if (ignoredOffers.length) {
    notes.push(
      `${ignoredOffers.map((id) => `\`${id}\``).join(", ")} offered but nothing registered ${ignoredOffers.length === 1 ? "it" : "them"}: there is no schema to send and no tool to call. Offering a name is not registering a tool.`,
    );
  }
  if (deactivateTool && guard.blocks.includes(deactivateTool)) {
    notes.push(
      `\`${deactivateTool}\` is both deactivated and guarded. The deactivation wins: a call to a tool that is no longer active never reaches the tool_call hooks, so the guard has nothing to block.`,
    );
  }
  if (deactivateTool && metrics.deactivated === 0) {
    notes.push(
      `nothing was deactivated: the run makes no \`${deactivateTool}\` call after turn ${deactivateAfter}, so the control has no call to fail.`,
    );
  }
  notes.push(
    "A tool that is deactivated stays registered: `setActiveTools` changes what is active, and no API removes a tool from the registry. Only a reload of the whole runtime re-reads the extension files.",
  );

  return {
    plugins,
    pluginIds,
    registry: {
      registered: registered.map((tool) => tool.id),
      offered,
      registeredCount: registered.length,
      offeredCount: offered.length,
    },
    fileA,
    fileB,
    prefix,
    activeBranch,
    abandoned,
    compaction,
    metrics,
    trace,
    notes,
    guard: { id: guard.id, terminate: guard.terminate, blocks: guard.blocks },
    deactivation: { tool: deactivateTool, afterTurn: deactivateAfter },
    fork: {
      afterTurn: prefixLength > 0 ? forkAfterTurn : 0,
      requested: forkPoint.afterTurn,
      clamped: forkPoint.afterTurn !== forkAfterTurn,
      prefixEntries: prefix.length,
      prefixTokens: sumTokens(prefix),
      abandonedEntries: abandoned.length,
      abandonedTokens: sumTokens(abandoned),
      newFileEntries: fileB ? fileB.entries.length : 0,
      newFileTokens: fileB ? fileB.tokens : 0,
      deleted: 0,
    },
    unregister: {
      ...UNREGISTER_PROBE,
      attempted: Boolean(options.unregisterAttempt),
      registeredBefore: registered.length,
      registeredAfter: registered.length,
      activeAfter: offered.length,
    },
  };
}
