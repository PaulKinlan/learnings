// @ts-check
/**
 * A deterministic model of the dsh + Cordis runtime this page explains, and nothing else.
 *
 * Every rule in here is taken from the vendored Cordis source or the dsh docs, and the page names
 * the rule next to the control that shows it:
 *
 *   - a mounted plugin is a fiber; its registrations are effects (vendor/cordis/src/fiber.ts);
 *   - a fiber's effects are disposed in reverse registration order (fiber.ts `dispose()`);
 *   - a fiber with unmet `inject` entries never activates; when a service appears or disappears the
 *     fiber's dependency epoch changes and it is (re)loaded or unloaded (fiber.ts `_refresh`,
 *     `_setEpoch`);
 *   - an unloading provider waits for its dependents' effects to finish first (fiber.ts `_unload`).
 *
 * What is modelled away, deliberately: config schemas, the loader and patch layers, real module
 * loading, real time, and every token count (those are illustrative — the page says so).
 */

/**
 * @typedef {object} EffectSpec
 * @property {string} label    the registration, in the vocabulary the docs use
 * @property {string} inverse  the disposer, described by what it takes back
 */

/**
 * @typedef {object} PluginSpec
 * @property {string} id
 * @property {string} pkg      the real npm package name
 * @property {string} role     core | seam | consumer | driver | observer
 * @property {string[]} claims ctx keys this plugin provides
 * @property {string[]} injects ctx keys this plugin waits for
 * @property {{name: string, tokens: number}[]} [tools] model-facing tools it registers
 * @property {EffectSpec[]} effects
 * @property {boolean} [defaultMounted]
 * @property {string} note
 */

/** Token sizes for tool schemas. Illustrative: the clone's schemas are not measured here. */
const SCHEMA_TOKENS = { bash: 300, read: 190, edit: 220, write: 200 };

/** @type {PluginSpec[]} */
export const PLUGINS = [
  {
    id: "session",
    pkg: "@deepseek-ai/dsh-session",
    role: "core",
    claims: ["sessions"],
    injects: [],
    effects: [
      { label: 'ctx.provide("sessions")', inverse: "unregister ctx.sessions" },
      { label: 'ctx.on("session/event", appendToFeed)', inverse: "remove the session/event listener" },
    ],
    defaultMounted: true,
    note: "The append-only SessionEvent log. Model-visible means logged, so this is the state that outlives a swap.",
  },
  {
    id: "system-prompt",
    pkg: "@deepseek-ai/dsh-system-prompt",
    role: "core",
    claims: ["systemPrompt"],
    injects: [],
    effects: [
      { label: 'ctx.provide("systemPrompt")', inverse: "unregister ctx.systemPrompt" },
      { label: "ctx.effect(register a prompt section)", inverse: "remove every section it added" },
    ],
    defaultMounted: true,
    note: "Prompt-section and tool-schema assembly: the place tool schemas are counted into the request.",
  },
  {
    id: "tools",
    pkg: "@deepseek-ai/dsh-tools",
    role: "core",
    claims: ["tools"],
    injects: [],
    effects: [
      { label: 'ctx.provide("tools")', inverse: "unregister ctx.tools" },
      { label: 'ctx.on("tools/pre-execute", policy)', inverse: "remove the pre-execute policy listener" },
      { label: 'ctx.on("tools/post-execute", finalize)', inverse: "remove the post-execute listener" },
    ],
    defaultMounted: true,
    note: "The scoped tool registry and the guarded execution pipeline; per-agent restrictions narrow the visible set.",
  },
  {
    id: "llm",
    pkg: "@deepseek-ai/dsh-llm",
    role: "core",
    claims: ["llm"],
    injects: [],
    effects: [
      { label: 'ctx.provide("llm")', inverse: "unregister ctx.llm" },
      { label: 'ctx.on("llm/stream", stream)', inverse: "remove the stream listener" },
    ],
    defaultMounted: true,
    note: "The provider-neutral call service: adapters register here, the loop calls here.",
  },
  {
    id: "llm-deepseek",
    pkg: "@deepseek-ai/dsh-llm-deepseek",
    role: "seam",
    claims: [],
    injects: ["llm"],
    effects: [
      { label: "ctx.effect(register the DeepSeek adapter on ctx.llm)", inverse: "unregister the adapter" },
    ],
    defaultMounted: true,
    note: "One adapter behind the ctx.llm seam. Nothing above it imports it by name.",
  },
  {
    id: "agent-loop",
    pkg: "@deepseek-ai/dsh-agent-loop",
    role: "core",
    claims: ["agentLoop"],
    injects: ["tools", "sessions", "systemPrompt", "llm"],
    effects: [
      { label: 'ctx.provide("agentLoop")', inverse: "unregister ctx.agentLoop" },
      { label: 'ctx.on("agent/pre-step", admit)', inverse: "remove the pre-step listener" },
      { label: 'ctx.on("agent/turn-stopping", decide)', inverse: "remove the turn-stopping listener" },
    ],
    defaultMounted: true,
    note: "The default driver behind ctx.agentLoop: one model request plus its tools is a step, zero or more steps is a turn.",
  },
  {
    id: "bash-local",
    pkg: "@deepseek-ai/dsh-bash-local",
    role: "seam",
    claims: ["shell"],
    injects: [],
    effects: [
      { label: 'ctx.provide("shell")', inverse: "unregister the shell provider" },
      { label: "ctx.effect(spawn a fresh bash -c per command)", inverse: "stop accepting commands" },
    ],
    note: "The unconfined POSIX executor: a fresh non-login `bash -c` per command, with the harness process's own authority. dsh-base does not mount it; swapping to it is how a deployment drops confinement.",
  },
  {
    id: "bash-sandbox",
    pkg: "@deepseek-ai/dsh-bash-sandbox",
    role: "seam",
    claims: ["shell"],
    injects: ["sandbox"],
    effects: [
      { label: 'ctx.provide("shell")', inverse: "unregister the shell provider" },
      { label: "ctx.effect(confine every command via ctx.sandbox)", inverse: "stop accepting commands" },
    ],
    defaultMounted: true,
    note: "What dsh-base mounts behind ctx.shell on POSIX: the same seam, confined, consuming ctx.sandbox instead of the harness's own authority.",
  },
  {
    id: "sandbox",
    pkg: "@deepseek-ai/dsh-sandbox-local",
    role: "seam",
    claims: ["sandbox"],
    injects: [],
    effects: [
      { label: 'ctx.provide("sandbox")', inverse: "unregister the sandbox backend" },
      { label: "ctx.effect(bwrap / landlock / Seatbelt / Windows ACL runner)", inverse: "tear the runner down" },
    ],
    defaultMounted: true,
    note: "A confinement backend (bwrap, landlock, Seatbelt, Windows ACL). File access only: network and process visibility are outside its guarantee.",
  },
  {
    id: "tool-bash",
    pkg: "@deepseek-ai/dsh-tool-bash",
    role: "consumer",
    claims: [],
    injects: ["tools", "shell", "systemPrompt"],
    tools: [{ name: "bash", tokens: SCHEMA_TOKENS.bash }],
    effects: [
      { label: 'ctx.tools.register("bash")', inverse: "unregister the bash tool and drop its schema" },
      { label: 'ctx.on("tools/pre-execute", bashPolicy)', inverse: "remove the bash policy listener" },
      { label: "ctx.effect(add the bash prompt section)", inverse: "remove the bash prompt section" },
    ],
    defaultMounted: true,
    note: "The model-facing tool. It names the seam (ctx.shell), never a provider: swapping the provider leaves this plugin's code untouched.",
  },
  {
    id: "fs-local",
    pkg: "@deepseek-ai/dsh-fs-local",
    role: "seam",
    claims: ["fs"],
    injects: [],
    effects: [
      { label: 'ctx.provide("fs")', inverse: "unregister the filesystem provider" },
    ],
    note: "The filesystem seam's local provider. Mounting it is what makes dsh-tool-fs activate.",
  },
  {
    id: "tool-fs",
    pkg: "@deepseek-ai/dsh-tool-fs",
    role: "consumer",
    claims: [],
    injects: ["tools", "fs", "systemPrompt"],
    tools: [
      { name: "read", tokens: SCHEMA_TOKENS.read },
      { name: "edit", tokens: SCHEMA_TOKENS.edit },
      { name: "write", tokens: SCHEMA_TOKENS.write },
    ],
    effects: [
      { label: "ctx.tools.register(read, edit, write)", inverse: "unregister three tools and their schemas" },
      { label: "ctx.effect(add the file prompt section)", inverse: "remove the file prompt section" },
    ],
    note: "A second consumer of the same shape. Mount it before dsh-fs-local and it waits: that wait is a coeffect.",
  },
  {
    id: "compaction",
    pkg: "@deepseek-ai/dsh-compaction-basic",
    role: "driver",
    claims: [],
    injects: ["llm"],
    effects: [
      { label: 'ctx.on("agent/request", pressure)', inverse: "remove the pressure listener" },
      { label: "ctx.effect(register the summarizer)", inverse: "unregister the compaction backend" },
    ],
    note: "The shipped condensation backend: one extra model request, and the shadowed content stays in the log.",
  },
  {
    id: "approval",
    pkg: "@deepseek-ai/dsh-user-approval",
    role: "driver",
    claims: ["approval"],
    injects: [],
    effects: [
      { label: 'ctx.provide("approval")', inverse: "unregister ctx.approval" },
      { label: 'ctx.on("tools/pre-execute", ask)', inverse: "remove the approval listener" },
    ],
    defaultMounted: true,
    note: "The ask-before-risky-actions policy, mounted from a config row. dsh-base sets policy: ask unless DSH_PERMISSION_MODE is danger-full-access.",
  },
  {
    id: "permission-presets",
    pkg: "@deepseek-ai/dsh-permission-presets",
    role: "driver",
    claims: ["permissionPresets"],
    injects: [],
    effects: [
      { label: 'ctx.provide("permissionPresets")', inverse: "unregister ctx.permissionPresets" },
      { label: "ctx.effect(publish read-only / workspace-write / danger-full-access)", inverse: "withdraw the preset table" },
    ],
    defaultMounted: true,
    note: "The named permission presets. The shipped table pairs each mode with a sandbox mode and an approval policy.",
  },
  {
    id: "goal",
    pkg: "@deepseek-ai/dsh-goal",
    role: "core",
    claims: ["goals"],
    injects: [],
    effects: [
      { label: 'ctx.provide("goals")', inverse: "unregister ctx.goals" },
      { label: 'ctx.on("agent/turn-stopping", arm)', inverse: "remove the continuation listener" },
    ],
    defaultMounted: true,
    note: "One durable completion objective per session. It stores goal state and schedules nothing.",
  },
  {
    id: "goal-round-driver",
    pkg: "@deepseek-ai/dsh-goal-round-driver",
    role: "driver",
    claims: [],
    injects: ["goals"],
    effects: [
      { label: 'ctx.on("agent/status", continueWhileIdle)', inverse: "remove the idle listener" },
      { label: "ctx.effect(hold a round allowance)", inverse: "release the round allowance" },
    ],
    defaultMounted: true,
    note: "Turns an active goal into rounds: each round is another turn, and the goal's cap bounds them.",
  },
  {
    id: "inspector",
    pkg: "@deepseek-ai/dsh-experimental-inspector",
    role: "observer",
    claims: ["inspector"],
    injects: [],
    effects: [
      { label: 'ctx.provide("inspector")', inverse: "unregister ctx.inspector" },
      { label: 'ctx.on("agent/status", snapshotTree)', inverse: "remove the tree listener" },
    ],
    note: "The observability-for-free case: it reads the context tree and changes no product behaviour.",
  },
];

const BY_ID = new Map(PLUGINS.map((plugin) => [plugin.id, plugin]));

/** The default round cap in dsh-goal, for the page to quote rather than invent. */
export const DEFAULT_ROUND_CAP = 256;

/** Root-context service keys, in the order the page's table lists them. */
export const SERVICE_KEYS = ["sessions", "systemPrompt", "tools", "llm", "agentLoop", "shell", "sandbox", "approval", "permissionPresets", "goals", "fs", "inspector"];

/** ctx keys that must resolve to exactly one provider at a time. */
const EXCLUSIVE_KEYS = new Set(["shell"]);

const clone = (value) => JSON.parse(JSON.stringify(value));

/**
 * Mount a plugin in the root context.
 * @param {ReturnType<typeof initialRuntime>} state
 * @param {string} id
 */
export function mount(state, id) {
  if (!BY_ID.has(id)) throw new Error(`unknown plugin ${id}`);
  const next = clone(state);
  if (next.mounted.includes(id)) return derive(next);
  const spec = BY_ID.get(id);
  const rows = [];
  // A second provider for an exclusive key replaces the first, the way a profile patch replaces a
  // row by id rather than stacking a second one beside it.
  for (const key of spec.claims) {
    if (!EXCLUSIVE_KEYS.has(key)) continue;
    const current = providersOf(next.mounted)[key];
    if (current && current !== id) rows.push(...removePlugin(next, current, "replaced as the exclusive provider"));
  }
  next.mounted.push(id);
  // A mount with nothing replaced unwinds nothing, so the ledger of inverses is cleared rather
  // than left showing the previous action's rows.
  next.unwind = rows.length ? { reason: `mount ${spec.pkg}`, rows } : null;
  next.lastAction = { kind: "mount", id };
  return derive(next);
}

/**
 * Unmount a plugin from the root context. Its dependents lose a coeffect and deactivate first, then
 * the plugin's own effects unwind. Within one fiber the unwind is reverse registration order.
 * @param {ReturnType<typeof initialRuntime>} state
 * @param {string} id
 */
export function unmount(state, id) {
  if (!BY_ID.has(id)) throw new Error(`unknown plugin ${id}`);
  const next = clone(state);
  if (!next.mounted.includes(id)) return derive(next);
  const rows = removePlugin(next, id, "unmounted");
  next.unwind = rows.length ? { reason: `unmount ${BY_ID.get(id).pkg}`, rows } : null;
  next.lastAction = { kind: "unmount", id, deactivated: next.lastDeactivated ?? [] };
  return derive(next);
}

/**
 * Swap the `ctx.shell` provider. The consumer is not edited: it loses the service, unloads, sees
 * the new provider, and reloads — which is why its registrations end up at the end of the ledger.
 * @param {ReturnType<typeof initialRuntime>} state
 * @param {"bash-local" | "bash-sandbox"} id
 */
export function swapShell(state, id) {
  if (!["bash-local", "bash-sandbox"].includes(id)) throw new Error(`not a shell provider: ${id}`);
  let next = clone(state);
  const current = providersOf(next.mounted).shell;
  if (current === id) return derive(next);
  const rows = current ? removePlugin(next, current, "unmounted: ctx.shell provider replaced") : [];
  next.mounted.push(id);
  next.unwind = { reason: `swap ctx.shell to ${BY_ID.get(id).pkg}`, rows };
  next.lastAction = { kind: "swap", id, deactivated: next.lastDeactivated ?? [] };
  return derive(next);
}

/** The shell provider the root context currently resolves, if any. */
export function shellProvider(state) {
  const provider = providersOf(state.mounted).shell;
  return provider ? BY_ID.get(provider) : null;
}

/**
 * Hot-swap the agent-loop module: generation 1 unloads (every effect unwinds) and generation 2
 * mounts in the same context. Only the registrations move; the session log is owned by dsh-session.
 * @param {ReturnType<typeof initialRuntime>} state
 */
export function hotSwapLoop(state) {
  const next = clone(state);
  const rows = removePlugin(next, "agent-loop", "unloaded for the module swap");
  next.mounted.push("agent-loop");
  next.loopGeneration = next.loopGeneration + 1;
  next.unwind = { reason: `hot-swap ${BY_ID.get("agent-loop").pkg} to generation ${next.loopGeneration}`, rows };
  next.lastAction = { kind: "hot-swap", id: "agent-loop", deactivated: next.lastDeactivated ?? [] };
  return derive(next);
}

/**
 * Mount a sandbox provider inside an isolated `shell` scope. The root provider keeps every
 * registration: `ctx.isolate("shell")` shadows one service name below the child context.
 * @param {ReturnType<typeof initialRuntime>} state
 */
export function mountFork(state) {
  const next = clone(state);
  if (next.fork) return derive(next);
  const spec = BY_ID.get("bash-sandbox");
  const rows = spec.effects.map((effect, index) => ({
    seq: next.nextSeq++,
    plugin: spec.id,
    pkg: spec.pkg,
    label: effect.label,
    inverse: effect.inverse,
    scope: "fork",
    reason: index === 0 ? 'mounted inside ctx.isolate("shell")' : undefined,
  }));
  next.fork = { plugin: "bash-sandbox", effects: rows };
  next.forkUnwind = null;
  next.lastAction = { kind: "fork-mount" };
  return derive(next);
}

/**
 * Unmount the isolated scope only. The root ledger is untouched.
 * @param {ReturnType<typeof initialRuntime>} state
 */
export function unmountFork(state) {
  const next = clone(state);
  if (!next.fork) return derive(next);
  next.forkUnwind = { reason: "unmount the isolated shell scope", rows: [...next.fork.effects].reverse() };
  next.fork = null;
  next.lastAction = { kind: "fork-unmount" };
  return derive(next);
}

/**
 * Run one turn.
 *
 * `agent/pre-step` decides whether the claimed input enters the step. A rejected (or rewritten
 * empty) first claim closes a durable turn with no step — the turn events are logged either way.
 * @param {ReturnType<typeof initialRuntime>} state
 * @param {{preStep?: "enter" | "reject"}} [options]
 */
export function runTurn(state, options = {}) {
  const preStep = options.preStep ?? "enter";
  const next = clone(state);
  const provider = shellProvider(next);
  const turnId = `turn-${next.turnCount + 1}`;
  const log = (type, text, extra = {}) => {
    next.sessionLog.push({ seq: next.sessionLog.length + 1, type, turn: turnId, text, ...extra });
  };
  const flow = [];
  const step = (name, text) => flow.push({ name, text });

  log("turn/start", `${turnId} opens before its first input is claimed`);
  step("turn/start", `${turnId} opens`);
  log("agent/pre-step", preStep === "reject" ? "listener rejects the claimed message" : "listener enters the claimed message");
  step("agent/pre-step", preStep === "reject" ? "waterfall → reject" : "waterfall → enter(messages)");

  if (preStep === "reject") {
    log("turn/end", `${turnId} closes: durable turn, zero steps, nothing owed`);
    step("turn/end", "turn closes with no step");
    next.turnCount += 1;
    next.turn = { id: turnId, preStep, steps: 0, requests: 0, toolCalls: 0, provider: provider?.pkg ?? null, flow };
    return derive(next);
  }

  log("step/start", "step 1 of the turn");
  step("step/start", "step 1 starts");
  log("agent/request", "prepareCall() binds the adapter before the prompt is committed");
  step("agent/request", "waterfall → prepareCall()");
  log("llm/stream", "one provider attempt on ctx.llm");
  step("llm/stream", "waterfall → the adapter streams");
  log("assistant/message", "the reply carries one tool call");
  step("assistant/message", "assistant/message committed");
  log("tool/call", `bash → ${provider ? provider.pkg : "no shell provider"}`);
  step("tool/call", "tools/pre-execute → tools/execute → tools/post-execute");
  log("tool/result", provider ? `returned by ${provider.pkg}` : "no provider: the call cannot run");
  step("tool/result", "tool/result committed");
  log("step/end", "the step closes");
  step("step/end", "step ends");
  log("agent/turn-stopping", "serial listener: nothing is owed, so the turn stops");
  step("agent/turn-stopping", "serial, no next()");
  log("turn/end", `${turnId} closes: 1 step`);
  step("turn/end", "turn closes");

  next.turnCount += 1;
  next.turn = {
    id: turnId,
    preStep,
    steps: 1,
    requests: 1,
    toolCalls: 1,
    provider: provider?.pkg ?? null,
    flow,
  };
  return derive(next);
}

/**
 * Run a goal until its round cap blocks it. Each round is one turn in the same session, which is
 * the composition the page describes: goal ⊃ rounds ⊃ turns ⊃ steps.
 * @param {ReturnType<typeof initialRuntime>} state
 * @param {{cap?: number}} [options]
 */
export function runGoal(state, options = {}) {
  const cap = Math.max(1, Math.round(options.cap ?? 3));
  let next = clone(state);
  const driverActive = next.active.includes("goal-round-driver") && next.active.includes("goal");
  if (!driverActive) {
    next.goal = { phase: "inactive", rounds: 0, cap, steps: 0, blocker: null, driverActive };
    return derive(next);
  }
  const rounds = cap;
  let steps = 0;
  for (let round = 0; round < rounds; round += 1) {
    next = runTurn(next, { preStep: "enter" });
    steps += next.turn.steps;
  }
  next.goal = {
    phase: "blocked",
    rounds,
    cap,
    steps,
    driverActive,
    blocker: { code: "round-limit", message: "Goal round limit reached." },
  };
  next.turnCount = next.turnCount;
  return derive(next);
}

/** Reset to the boot composition. */
export function initialRuntime() {
  const state = {
    mounted: PLUGINS.filter((plugin) => plugin.defaultMounted).map((plugin) => plugin.id),
    detached: [],
    fork: null,
    forkUnwind: null,
    loopGeneration: 1,
    sessionLog: [
      { seq: 1, type: "session/created", turn: null, text: "session s-1 enters the SessionStore" },
      { seq: 2, type: "session/event", turn: null, text: "the durable feed opens; nothing has been asked yet" },
    ],
    turnCount: 0,
    turn: null,
    goal: null,
    unwind: null,
    lastAction: null,
    activationOrder: [],
    activationSeq: {},
    nextSeq: 1,
    effects: [],
  };
  return derive(state);
}

/**
 * Recompute everything observable from the mounted set: providers, activation, the effect ledger,
 * the tool registry and the counters.
 */
function derive(input) {
  const state = input;
  const providers = providersOf(state.mounted);
  const activeOrder = activationOf(state.mounted);

  // A plugin that stayed active keeps its sequence numbers; one that deactivated and came back
  // registers afresh, which is what makes a reactivation visible in the ledger.
  const active = new Set(activeOrder);
  state.activationOrder = state.activationOrder.filter((id) => active.has(id));
  for (const id of activeOrder) if (!state.activationOrder.includes(id)) state.activationOrder.push(id);

  const effects = [];
  for (const id of state.activationOrder) {
    const spec = BY_ID.get(id);
    const base = state.activationSeq[id] ?? (state.activationSeq[id] = state.nextSeq);
    spec.effects.forEach((effect, index) => {
      const seq = base + index;
      if (seq >= state.nextSeq) state.nextSeq = seq + 1;
      effects.push({ seq, plugin: id, pkg: spec.pkg, label: effect.label, inverse: effect.inverse, scope: "root" });
    });
  }
  effects.sort((a, b) => a.seq - b.seq);
  state.effects = effects;
  for (const id of Object.keys(state.activationSeq)) if (!active.has(id)) delete state.activationSeq[id];

  state.active = activeOrder;
  state.inactive = state.mounted
    .filter((id) => !active.has(id))
    .map((id) => ({ id, waitingFor: BY_ID.get(id).injects.filter((key) => !providers[key]) }));
  state.services = providers;
  state.registry = registryOf(activeOrder);
  state.counts = {
    mounted: state.mounted.length,
    active: activeOrder.length,
    inactive: state.inactive.length,
    registrations: effects.length,
    detached: state.detached.length,
  };
  return state;
}

/**
 * Remove `id` from the mounted set and unwind, in order: the dependents that lose a coeffect (each
 * one's own effects in reverse registration order), then the plugin's own effects in reverse
 * registration order. Cordis guarantees the two halves separately: reverse order within a fiber,
 * and every dependent unloaded before the provider's effects are disposed.
 *
 * @returns the unwound rows, in the order they came off
 */
function removePlugin(state, id, reason) {
  const before = state.active;
  const remaining = state.mounted.filter((other) => other !== id);
  const after = activationOf(remaining);
  const deactivated = before.filter((other) => other !== id && !after.includes(other)).reverse();
  const rows = [];
  for (const other of [...deactivated, id]) {
    rows.push(...takeEffects(state, other, other === id ? reason : `deactivated: lost ${missingKeys(state, remaining, other)}`));
    // A reactivation registers afresh, so the ledger cannot reuse the old sequence numbers.
    delete state.activationSeq[other];
    state.activationOrder = state.activationOrder.filter((entry) => entry !== other);
  }
  state.mounted = remaining;
  state.detached.push(...deactivated, id);
  state.lastDeactivated = deactivated;
  return rows;
}

/** The names of the ctx keys `id` injects that the remaining mounted set no longer provides. */
function missingKeys(state, remaining, id) {
  const providers = providersOf(remaining);
  return BY_ID.get(id).injects.filter((key) => !providers[key]).join(", ") || "a service";
}

/** Ordered active plugin ids for a mounted set: providers before their dependents. */
function activationOf(mounted) {
  const providers = providersOf(mounted);
  const active = new Set();
  const order = [];
  let progress = true;
  while (progress) {
    progress = false;
    for (const id of mounted) {
      if (active.has(id)) continue;
      const spec = BY_ID.get(id);
      if (!spec.injects.every((key) => providers[key] && active.has(providers[key]))) continue;
      active.add(id);
      order.push(id);
      progress = true;
    }
  }
  return order;
}

/** ctx key → plugin id, for a mounted set. */
function providersOf(mounted) {
  const providers = {};
  for (const id of mounted) {
    for (const key of BY_ID.get(id).claims) providers[key] = id;
  }
  return providers;
}

/** The tools an active plugin set registers, with their schema cost. */
function registryOf(active) {
  const tools = [];
  for (const id of active) {
    for (const tool of BY_ID.get(id).tools ?? []) tools.push({ ...tool, from: id });
  }
  return { tools, schemaTokens: tools.reduce((total, tool) => total + tool.tokens, 0) };
}

/** The ledger rows for one plugin, removed and returned in reverse registration order. */
function takeEffects(state, id, reason) {
  const rows = state.effects.filter((effect) => effect.plugin === id);
  state.effects = state.effects.filter((effect) => effect.plugin !== id);
  return [...rows].reverse().map((row) => ({ ...row, reason }));
}

/** The domain each event belongs to, for the page's event list. */
export const EVENT_DOMAIN = {
  "session/created": "session",
  "session/event": "session",
  "turn/start": "session",
  "turn/end": "session",
  "step/start": "session",
  "step/end": "session",
  "agent/pre-step": "agent",
  "agent/request": "agent",
  "agent/turn-stopping": "agent",
  "llm/stream": "agent",
  "assistant/message": "session",
  "tool/call": "capability",
  "tool/result": "session",
};
