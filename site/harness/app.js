// @ts-check
/**
 * The playground's DOM half. All arithmetic lives in sim.js; this file reads the controls, asks
 * the simulation for a run, and renders it. Nothing here holds a token count of its own: every
 * number printed comes out of the returned run, which is what makes the page's labels derived
 * rather than hardcoded.
 */
import { DEMO_MODELS, THINKING_LEVELS, TOOLS, PROMPT_TOKENS, simulateRun } from "./sim.js";

const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
const number = (value) => value.toLocaleString("en-US");
const percent = (part, whole) => `${Math.min(100, (part / whole) * 100)}%`;

/** The node factory. Text goes in through textContent, never as markup. */
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function readControls() {
  return {
    model: $("model").value,
    thinking: $("thinking").value,
    tools: TOOLS.map((tool) => tool.id).filter((id) => /** @type {HTMLInputElement} */ ($(`tool-${id}`)).checked),
    batchMode: $("batch").value,
    inject429: $("failure").value,
    autoRetry: $("autoretry").value === "on",
    prompt: $("prompt").value,
    reserveTokens: Number(/** @type {HTMLInputElement} */ ($("reserve")).value),
    keepRecentTokens: Number(/** @type {HTMLInputElement} */ ($("keep")).value),
  };
}

/**
 * The two ranges are bounded by the selected model's window: a reserve larger than the window is
 * not a setting, it is a bug. Moving the window re-clamps them and the outputs follow.
 */
function syncRangeBounds() {
  const model = DEMO_MODELS.find((entry) => entry.id === $("model").value) ?? DEMO_MODELS[0];
  for (const [id, max] of [["reserve", model.contextWindow - 1024], ["keep", model.contextWindow]]) {
    const input = /** @type {HTMLInputElement} */ ($(id));
    input.max = String(max);
    if (Number(input.value) > max) input.value = String(max);
    $(`${id}-value`).textContent = input.value;
  }
}

function bar(contextTokens, window, threshold) {
  const wrap = node("div", "ctx-bar");
  wrap.setAttribute("role", "img");
  wrap.setAttribute(
    "aria-label",
    `context ${number(contextTokens)} of ${number(window)} tokens; compaction line at ${number(threshold)}`,
  );
  const fill = node("div", "ctx-fill");
  fill.style.width = percent(contextTokens, window);
  const mark = node("div", "ctx-mark");
  mark.style.left = percent(threshold, window);
  wrap.append(fill, mark);
  return wrap;
}

/** One trace entry. Each kind says what the harness did, in the harness's own vocabulary. */
function entry(event, run) {
  const item = node("li", `ev ev-${event.kind}`);
  switch (event.kind) {
    case "prompt":
      item.append(
        node("p", "ev-title", `user message · ${event.role} · ${number(event.tokens)} tokens`),
        node("p", "ev-note", event.note),
      );
      break;
    case "compaction":
      item.append(
        node("p", "ev-title", "compaction"),
        node(
          "p",
          "ev-line",
          `context ${number(event.before)} → ${number(event.after)} tokens · summarised ${number(event.summarisedTokens)} into ${number(event.summaryTokens)} · kept the last ${event.keptMessages} messages, ${number(event.keptTokens)} tokens of the ${number(event.keepRecentTokens)} keepRecentTokens window`,
        ),
      );
      break;
    case "compaction-skipped":
      item.append(
        node("p", "ev-title", "compaction skipped"),
        node("p", "ev-line", `context ${number(event.before)} is past the line at ${number(event.threshold)}, but ${event.reason}`),
      );
      break;
    case "request": {
      const windowTokens = event.window ?? run.model.contextWindow;
      const over = event.contextTokens > windowTokens;
      if (over) item.classList.add("ev-over");
      item.append(
        node(
          "p",
          "ev-title",
          event.status === "429"
            ? `request #${event.index} · HTTP 429`
            : `request #${event.index} · context ${number(event.contextTokens)} / ${number(windowTokens)}`,
        ),
        bar(event.contextTokens, windowTokens, event.threshold),
      );
      if (over) {
        item.append(
          node(
            "p",
            "ev-warn",
            `over the window by ${number(event.contextTokens - windowTokens)} tokens: nothing was left to compact (see the skipped note above), and the bar stops at 100% — a real request this size would be refused or truncated by the provider.`,
          ),
        );
      }
      if (event.note) item.append(node("p", "ev-note", event.note));
      if (event.status === "ok") {
        item.append(
          node(
            "p",
            "ev-note",
            `billed ${number(event.billed)} input tokens · reasoning ${number(event.reasoningTokens)} · compaction line ${number(event.threshold)}`,
          ),
        );
      }
      break;
    }
    case "retry":
      item.append(
        node("p", "ev-title", `retry · attempt ${event.attempt} after HTTP ${event.status}`),
        node("p", "ev-note", `waiting ${number(event.backoffMs)} ms, then re-sending the same context`),
      );
      break;
    case "assistant":
      item.append(
        node(
          "p",
          "ev-title",
          event.calls.length
            ? `assistant · turn ${event.turn} · ${event.calls.length} call${event.calls.length === 1 ? "" : "s"}`
            : `assistant · turn ${event.turn} · closing answer`,
        ),
        node(
          "p",
          "ev-note",
          `${number(event.textTokens)} text tokens${event.reasoningTokens ? ` + ${number(event.reasoningTokens)} reasoning tokens, kept on the message` : ", no reasoning"}`,
        ),
      );
      break;
    case "tool":
      item.append(
        node(
          "p",
          "ev-title",
          event.status === "ok"
            ? `${event.tool} ${event.target} → ${number(event.resultTokens)} tokens`
            : `${event.tool} ${event.target} · isError`,
        ),
      );
      if (event.error) item.append(node("p", "ev-line", event.error));
      break;
    case "note":
      item.append(node("p", "ev-note", event.text));
      break;
    case "run-end":
      item.append(node("p", "ev-title", "agent_end — this run is closed"));
      break;
    case "settled":
      item.append(node("p", "ev-title", "agent_settled — no automatic work remains"));
      break;
    case "run-start":
      item.append(node("p", "ev-title", `run ${event.run} starts over the same transcript`));
      break;
    case "failed":
      item.append(node("p", "ev-title", "run failed"), node("p", "ev-line", event.reason));
      break;
    default:
      item.append(node("p", "ev-title", event.kind));
  }
  return item;
}

const METRIC_ROWS = [
  ["Model requests", (run) => number(run.metrics.requests)],
  ["Tool calls executed", (run) => number(run.metrics.executedCalls)],
  ["Tool calls refused", (run) => number(run.metrics.rejectedCalls)],
  ["Retries after a 429", (run) => number(run.metrics.retries)],
  ["Compactions", (run) => number(run.metrics.compactions)],
  ["Runs", (run) => number(run.metrics.runs)],
  [
    "Prompt tokens (system + tool schemas)",
    (run) => `${number(run.metrics.promptTokens)} · ${run.metrics.toolDefinitions} tool schema${run.metrics.toolDefinitions === 1 ? "" : "s"}`,
  ],
  ["Billed input tokens", (run) => number(run.metrics.billedInputTokens)],
  ["Reasoning tokens", (run) => number(run.metrics.reasoningTokens)],
  ["Output tokens", (run) => number(run.metrics.outputTokens)],
  ["Peak context", (run) => `${number(run.metrics.peakContext)} of ${number(run.model.contextWindow)}`],
];

function render() {
  syncRangeBounds();
  const run = simulateRun(readControls());

  const metrics = $("metrics");
  metrics.replaceChildren();
  for (const [label, value] of METRIC_ROWS) {
    metrics.append(node("dt", "", label), node("dd", "number", value(run)));
  }

  $("status").textContent = {
    settled: `settled — ${number(run.metrics.requests)} model requests, ${number(run.metrics.compactions)} compactions, ${number(run.metrics.rejectedCalls)} refused ${run.metrics.rejectedCalls === 1 ? "call" : "calls"}, ${number(run.metrics.billedInputTokens)} billed input tokens.`,
    blocked: `blocked — a write-class call was not in the allowlist, so the scripted run took its blocked branch after ${number(run.metrics.requests)} requests.`,
    failed: `failed — a 429 with auto-retry off ended the run after ${number(run.metrics.requests)} requests. Everything after it never happened.`,
  }[run.status];

  $("compaction-line").textContent =
    `Compaction fires above ${number(run.threshold)} tokens: a ${number(run.model.contextWindow)}-token window minus reserveTokens ${number(run.reserveTokens)}. It leaves the last keepRecentTokens ${number(run.keepRecentTokens)} untouched.`;

  const trace = $("trace");
  trace.replaceChildren();
  for (const event of run.events) trace.append(entry(event, run));

  $("sim-inputs").textContent =
    `The simulation's own constants: base prompt ${number(PROMPT_TOKENS.system)} tokens, task ${number(PROMPT_TOKENS.task)}, ` +
    `a mid-run prompt ${number(PROMPT_TOKENS.steer)}; thinking ` +
    `${THINKING_LEVELS.map((level) => `${level.id} = ${number(level.reasoningTokens)} reasoning tokens per response`).join(", ")}; ` +
    `tool schemas ${TOOLS.map((tool) => `${tool.id} ${number(tool.promptTokens)}`).join(", ")}.`;
}

for (const control of document.querySelectorAll("#model, #thinking, #batch, #failure, #autoretry, #prompt, #reserve, #keep, input[type=checkbox]")) {
  control.addEventListener("input", render);
  control.addEventListener("change", render);
}

render();
