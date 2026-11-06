// @ts-check
/**
 * A deterministic model of the harness loop's mechanics.
 *
 * NOTHING HERE CALLS A MODEL. Every token count is a constant in this file and every number the
 * playground shows is computed from the controls plus those constants. The point is to make the
 * loop's arithmetic visible — where a request boundary falls, what a batch costs in re-sent
 * context, when compaction fires and what it leaves behind — not to predict what a model would
 * write. The scripted assistant is a fixed scenario with a fixed blocked branch.
 *
 * The mechanics modelled, in the harness's own vocabulary:
 *   - one turn = one assistant response + its tool calls + their results, appended to the
 *     transcript, then another model request if the assistant asked for tools;
 *   - `agent_end` closes one run, `agent_settled` says no automatic work remains;
 *   - compaction fires when `contextTokens > contextWindow - reserveTokens`, keeps
 *     `keepRecentTokens` from the end untouched, and replaces the older turns with a summary;
 *   - a tool that is not in the run's allowlist is not executed: the assistant gets an
 *     `isError: true` result back instead;
 *   - a tool error does not end the run, but an assistant response with no tool calls does.
 */

/** The two demo windows. Both are small on purpose, so a transcript of a few thousand scripted
 *  tokens can cross the compaction line without the reader doing arithmetic about 128k contexts. */
export const DEMO_MODELS = [
  { id: "deepseek-v4-lite", label: "deepseek-v4-lite", contextWindow: 16384 },
  { id: "deepseek-v4-pro", label: "deepseek-v4-pro", contextWindow: 32768 },
];

/** Thinking level buys reasoning tokens. They are billed as output, kept on the assistant message
 *  (`requiresReasoningContentOnAssistantMessages`), and therefore re-sent as input on every later
 *  request in the same run. */
export const THINKING_LEVELS = [
  { id: "off", label: "thinking off", reasoningTokens: 0 },
  { id: "low", label: "thinking low", reasoningTokens: 900 },
  { id: "high", label: "thinking high", reasoningTokens: 2400 },
];

/** The built-in tools the harness ships. The first four are its default allowlist; `grep` and
 *  `find` exist but are not enabled unless a lane asks for them. `promptTokens` is the cost of
 *  carrying the tool's TypeBox-derived JSON Schema in the prompt. */
export const TOOLS = [
  { id: "read", label: "read", promptTokens: 220, defaultOn: true, writes: false },
  { id: "bash", label: "bash", promptTokens: 260, defaultOn: true, writes: false },
  { id: "edit", label: "edit", promptTokens: 240, defaultOn: true, writes: true },
  { id: "write", label: "write", promptTokens: 200, defaultOn: true, writes: true },
  { id: "grep", label: "grep", promptTokens: 180, defaultOn: false, writes: false },
  { id: "find", label: "find", promptTokens: 160, defaultOn: false, writes: false },
];

/** Prompt assembly, in tokens: the base prompt, the lane's role text, the task, and a user message
 *  that arrives mid-run (steering) or after the run has settled (a follow-up). */
export const PROMPT_TOKENS = { system: 900, task: 120, steer: 130 };

/** Per-response token costs: the assistant's own text, the arguments of each tool call, the
 *  closing answers, and the short `isError` result an unexecuted tool call produces. */
export const TURN_TOKENS = {
  assistantText: 140,
  perCallArgs: 90,
  finalAnswer: 320,
  blockedAnswer: 120,
  followUpAnswer: 280,
  rejectedResult: 40,
};

/** The summary that replaces compacted turns: a fixed preamble plus a fraction of what it read. */
export const SUMMARY_TOKENS = { base: 400, ratio: 0.15 };

/** The scripted run. Each step is one assistant response: the calls it carries, and the result
 *  each call returns. A step with no calls is the closing answer that ends the run. */
export const SCRIPT = [
  {
    calls: [
      { tool: "read", target: "remote/lane-host.mjs", resultTokens: 1800 },
      { tool: "grep", target: "agent_settled", resultTokens: 700 },
      { tool: "read", target: "pi/loop.ts", resultTokens: 2200 },
    ],
  },
  {
    calls: [
      { tool: "bash", target: "npm test -- tests/lane-host.test.js", resultTokens: 950 },
      { tool: "read", target: "pi/compact.ts", resultTokens: 1600 },
    ],
  },
  {
    calls: [
      { tool: "edit", target: "remote/lane-host.mjs", resultTokens: 260 },
      { tool: "read", target: "remote/lane-host.mjs", resultTokens: 1400 },
    ],
  },
  {
    calls: [{ tool: "bash", target: "npm test -- tests/lane-host.test.js", resultTokens: 900 }],
  },
  { calls: [] },
];

/** The turn after which a mid-run prompt arrives. Turn 1 is the first assistant response. */
export const PROMPT_ARRIVES_AFTER_TURN = 2;

export const defaultAllowlist = () => TOOLS.filter((tool) => tool.defaultOn).map((tool) => tool.id);

const byId = (list, id, fallbackId) => list.find((entry) => entry.id === id) ?? byId(list, fallbackId) ?? list[0];
const toolById = (id) => TOOLS.find((tool) => tool.id === id);
const total = (messages) => messages.reduce((sum, message) => sum + message.tokens, 0);

/** Clamp to the range and step the control advertises, so a hand-built call and a slider agree. */
function clampStep(value, min, max, step) {
  const clamped = Math.min(Math.max(Number(value) || 0, min), max);
  return Math.round(clamped / step) * step;
}

/**
 * Drop everything outside a `keepRecentTokens` window from the end of the transcript.
 *
 * A summary cannot replace the newest message, so at least one message is always kept — a
 * compaction that emptied the transcript after a tool call would leave the next request with
 * results the assistant never asked for. The system prompt is not in `messages`; the harness's
 * `keepRecentTokens` does not touch it either.
 */
export function compactMessages(messages, keepRecentTokens) {
  let kept = 0;
  let index = messages.length;
  while (index > 0) {
    const next = messages[index - 1];
    if (kept > 0 && kept + next.tokens > keepRecentTokens) break;
    kept += next.tokens;
    index -= 1;
    if (kept >= keepRecentTokens) break;
  }
  const older = messages.slice(0, index);
  const recent = messages.slice(index);
  const summarisedTokens = total(older);
  const summaryTokens = older.length
    ? Math.round(SUMMARY_TOKENS.base + SUMMARY_TOKENS.ratio * summarisedTokens)
    : 0;
  return { older, recent, keptTokens: kept, summarisedTokens, summaryTokens };
}

/**
 * Run the scripted scenario under one configuration.
 *
 * Returns `{ events, metrics, status, threshold }` plus the controls as they were resolved (the
 * clamped reserve, the model, the allowlist), so a caller can print the inputs it actually used:
 *   - `events` is the trace the page renders, in order;
 *   - `metrics` is every number the page shows, each one derived from `options` and the
 *     constants above (no literal in the renderer);
 *   - `status` is `settled`, `blocked` (a write-class tool was not in the allowlist) or `failed`
 *     (an injected 429 with auto-retry off).
 */
export function simulateRun(options = {}) {
  const model = byId(DEMO_MODELS, options.model, "deepseek-v4-lite");
  const thinking = byId(THINKING_LEVELS, options.thinking, "low");
  const allowlist = (options.tools ?? defaultAllowlist()).filter((id) => toolById(id));
  const batchMode = options.batchMode === "sequential" ? "sequential" : "parallel";
  const reserveTokens = clampStep(options.reserveTokens ?? 8192, 0, model.contextWindow - 1024, 512);
  const keepRecentTokens = clampStep(options.keepRecentTokens ?? 4096, 0, model.contextWindow, 256);
  const failureRequest = options.inject429 === "request-2" ? 2 : 0;
  const autoRetry = options.autoRetry !== false;
  const promptMode = options.prompt === "steer" || options.prompt === "follow-up" ? options.prompt : "none";

  const threshold = model.contextWindow - reserveTokens;
  const systemTokens = PROMPT_TOKENS.system + allowlist.reduce((sum, id) => sum + toolById(id).promptTokens, 0);

  const events = [];
  const messages = [];
  const metrics = {
    promptTokens: systemTokens,
    toolDefinitions: allowlist.length,
    requests: 0,
    retries: 0,
    compactions: 0,
    toolCalls: 0,
    executedCalls: 0,
    rejectedCalls: 0,
    billedInputTokens: 0,
    reasoningTokens: 0,
    outputTokens: 0,
    runs: 1,
    peakContext: 0,
  };

  const contextTokens = () => systemTokens + total(messages);
  let requestIndex = 0;
  let turn = 0;
  let blocked = false;
  let failed = false;

  /** The compaction check that runs before every model request. */
  function compactIfNeeded() {
    const before = contextTokens();
    if (before <= threshold) return;
    const { older, recent, keptTokens, summarisedTokens, summaryTokens } = compactMessages(
      messages,
      keepRecentTokens,
    );
    if (!older.length) {
      const reason =
        systemTokens >= threshold
          ? `the system prompt and tool schemas alone (${systemTokens} tokens) are already past the line, and compaction only rewrites messages`
          : keepRecentTokens >= total(messages)
            ? `keepRecentTokens (${keepRecentTokens}) covers the whole transcript, so there is nothing older to summarise`
            : "the newest message alone exceeds the kept window, and a summary cannot replace it";
      events.push({ kind: "compaction-skipped", before, threshold, keepRecentTokens, reason });
      return;
    }
    const summary = { role: "assistant", kind: "summary", tokens: summaryTokens };
    messages.splice(0, messages.length, summary, ...recent);
    metrics.compactions += 1;
    events.push({
      kind: "compaction",
      before,
      after: contextTokens(),
      summarisedTokens,
      summaryTokens,
      keptTokens,
      keptMessages: recent.length,
      keepRecentTokens,
      threshold,
    });
  }

  /**
   * One model request, with the optional injected 429. Returns false when the run has to stop:
   * with auto-retry off, a 429 is the end of the run, not something the loop absorbs.
   */
  function request() {
    compactIfNeeded();
    const context = contextTokens();
    metrics.peakContext = Math.max(metrics.peakContext, context);
    requestIndex += 1;
    if (failureRequest && requestIndex === failureRequest) {
      metrics.requests += 1;
      events.push({
        kind: "request",
        index: requestIndex,
        contextTokens: context,
        threshold,
        status: "429",
        billed: 0,
        note: "refused before any output: the request cost no tokens",
      });
      if (!autoRetry) {
        failed = true;
        events.push({
          kind: "failed",
          index: requestIndex,
          reason: "the provider answered 429 and auto-retry is off, so the run ends here",
        });
        return false;
      }
      metrics.retries += 1;
      events.push({ kind: "retry", attempt: 1, status: 429, backoffMs: 2000 });
    }
    metrics.requests += 1;
    metrics.billedInputTokens += context;
    events.push({
      kind: "request",
      index: requestIndex,
      contextTokens: context,
      threshold,
      window: model.contextWindow,
      status: "ok",
      billed: context,
      reasoningTokens: thinking.reasoningTokens,
    });
    return true;
  }

  /** One assistant response plus the results of the calls it carried. */
  function exchange(calls) {
    if (!request()) return false;
    turn += 1;
    const textTokens = TURN_TOKENS.assistantText + calls.length * TURN_TOKENS.perCallArgs;
    messages.push({ role: "assistant", kind: calls.length ? "tool-call" : "answer", tokens: textTokens + thinking.reasoningTokens });
    metrics.outputTokens += textTokens;
    metrics.reasoningTokens += thinking.reasoningTokens;
    events.push({
      kind: "assistant",
      turn,
      textTokens,
      reasoningTokens: thinking.reasoningTokens,
      calls: calls.map((call) => call.tool),
    });
    for (const call of calls) {
      const tool = toolById(call.tool);
      metrics.toolCalls += 1;
      if (!allowlist.includes(call.tool)) {
        metrics.rejectedCalls += 1;
        messages.push({ role: "tool", kind: "error", tool: call.tool, tokens: TURN_TOKENS.rejectedResult });
        events.push({
          kind: "tool",
          tool: call.tool,
          target: call.target,
          status: "rejected",
          resultTokens: TURN_TOKENS.rejectedResult,
          error: `${call.tool} is not in this run's allowlist, so nothing ran and the assistant got isError: true back`,
        });
        if (tool.writes) blocked = true;
        continue;
      }
      metrics.executedCalls += 1;
      messages.push({ role: "tool", kind: "result", tool: call.tool, tokens: call.resultTokens });
      events.push({
        kind: "tool",
        tool: call.tool,
        target: call.target,
        status: "ok",
        resultTokens: call.resultTokens,
      });
    }
    if (promptMode === "steer" && turn === PROMPT_ARRIVES_AFTER_TURN) {
      messages.push({ role: "user", kind: "steer", tokens: PROMPT_TOKENS.steer });
      events.push({
        kind: "prompt",
        role: "steer",
        tokens: PROMPT_TOKENS.steer,
        note: "queued mid-run and delivered before the next request, inside this run",
      });
    }
    return true;
  }

  /** The closing answer, then `agent_end` and `agent_settled`. */
  function answer(tokens) {
    if (!request()) return;
    turn += 1;
    messages.push({ role: "assistant", kind: "answer", tokens });
    metrics.outputTokens += tokens;
    events.push({ kind: "assistant", turn, textTokens: tokens, reasoningTokens: 0, calls: [] });
    events.push({ kind: "run-end" });
    events.push({ kind: "settled" });
  }

  messages.push({ role: "user", kind: "task", tokens: PROMPT_TOKENS.task });
  events.push({ kind: "prompt", role: "task", tokens: PROMPT_TOKENS.task, note: "the task the run was started with" });

  // The scripted turns. Sequential tool calls mean one call per response, so every extra call
  // also costs another full context re-send — the batching decision, priced.
  for (const step of SCRIPT) {
    const batches = batchMode === "parallel" ? [step.calls] : step.calls.map((call) => [call]);
    for (const calls of batches.length ? batches : [[]]) {
      if (!exchange(calls)) break;
      if (blocked) break;
    }
    if (failed || blocked) break;
  }

  if (!failed) {
    if (blocked) {
      events.push({
        kind: "note",
        text: "a write-class call was refused, so the scripted run takes its blocked branch: no further calls, one short answer",
      });
      answer(TURN_TOKENS.blockedAnswer);
    } else {
      answer(TURN_TOKENS.finalAnswer);
    }
  }

  if (!failed && promptMode === "follow-up") {
    metrics.runs += 1;
    messages.push({ role: "user", kind: "follow-up", tokens: PROMPT_TOKENS.steer });
    events.push({ kind: "run-start", run: metrics.runs });
    events.push({
      kind: "prompt",
      role: "follow-up",
      tokens: PROMPT_TOKENS.steer,
      note: "the run had already settled, so this prompt starts a new run over the same transcript",
    });
    answer(TURN_TOKENS.followUpAnswer);
  }

  const status = failed ? "failed" : blocked ? "blocked" : "settled";
  return { events, metrics, status, threshold, model, thinking, allowlist, batchMode, reserveTokens, keepRecentTokens };
}
