// site/neural-networks/backpropagation.js — the two widgets on backpropagation.html.
// All arithmetic comes from math.js: singleNeuronChainRule() for the one-neuron panel, and
// buildTinyNetwork() + backwardTrace() for the step-through. tests/nn-chapters.test.js checks that
// the step-by-step trace gives exactly Value.backward()'s gradients and that a training step helps.
import {
  backwardTrace,
  buildTinyNetwork,
  mulberry32,
  numericalDerivative,
  singleNeuronChainRule,
  TINY_NET_DEFAULTS,
  trainTinyNetworkStep,
} from "./math.js";

const $ = (id) => document.getElementById(id);
const fmt = (v) =>
  Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-4) ? v.toExponential(2) : v.toFixed(4);
const LR = 0.5;

function cellRow(cells, className = "") {
  const tr = document.createElement("tr");
  if (className) tr.className = className;
  cells.forEach((text, i) => {
    const cell = document.createElement(i === 0 ? "th" : "td");
    if (i === 0) cell.scope = "row";
    cell.textContent = text;
    tr.append(cell);
  });
  return tr;
}

// ── One neuron: the chain rule is three numbers multiplied together ──────────────────────

const NEURON = ["w", "b", "x", "y"];

function renderNeuron() {
  const v = Object.fromEntries(NEURON.map((k) => [k, Number($(`bp-${k}`).value)]));
  for (const k of NEURON) $(`bp-${k}-out`).textContent = v[k].toFixed(2);
  const c = singleNeuronChainRule(v);
  const nudged = numericalDerivative((w) => singleNeuronChainRule({ ...v, w }).loss, v.w);
  $("bp-neuron-rows").replaceChildren(
    ...[
      ["z = w·x + b", c.z],
      ["a = σ(z)", c.a],
      ["L = (a − y)²", c.loss],
      ["∂L/∂a = 2(a − y)", c.dL_da],
      ["∂a/∂z = a(1 − a)", c.da_dz],
      ["∂z/∂w = x", c.dz_dw],
      ["∂L/∂w, the three multiplied", c.dL_dw],
      ["∂L/∂w, by nudging w", nudged],
    ].map(([label, value]) => cellRow([label, fmt(value)])),
  );
  $("bp-neuron-status").textContent =
    `∂L/∂w = ${fmt(c.dL_da)} × ${fmt(c.da_dz)} × ${fmt(c.dz_dw)} = ${fmt(c.dL_dw)}. ` +
    `Nudging w directly gives ${fmt(nudged)}.`;
}

for (const k of NEURON) $(`bp-${k}`).addEventListener("input", renderNeuron);
renderNeuron();

// ── A 2-2-1 network, one node at a time ──────────────────────────────────────────────────

let config = structuredClone(TINY_NET_DEFAULTS);
let seed = 1;
let net;
let trace;
let ops;
let names;
let forwardShown = 0;
let backwardShown = 0;

function build() {
  net = buildTinyNetwork(config);
  trace = backwardTrace(net.loss);
  ops = trace.order.filter((v) => v.children.length);
  names = new Map();
  for (const [k, node] of Object.entries(net.params)) names.set(node, k);
  names.set(net.nodes.x1, "x1");
  names.set(net.nodes.x2, "x2");
  let t = 0;
  for (const v of trace.order) {
    if (names.has(v)) continue;
    // The only unnamed leaf is the constant target y; unnamed operations become t1, t2, ...
    names.set(v, v.children.length ? v.label || `t${++t}` : "y");
  }
}

function expression(v) {
  const [a, b] = v.children.map((c) => names.get(c));
  if (v.op === "^2") return `${a}²`;
  if (v.children.length === 2) return `${a} ${v.op} ${b}`;
  return `${v.op}(${a})`;
}

/** Gradients after the first k backward steps: final for visited nodes, partial for the rest. */
function gradientsAfter(k) {
  const g = new Map([[net.loss, 1]]);
  for (const step of trace.steps.slice(0, k)) {
    for (const c of step.contributions) g.set(c.child, (g.get(c.child) ?? 0) + c.delta);
  }
  return g;
}

function update(message) {
  const g = gradientsAfter(backwardShown);
  const visited = new Set(trace.steps.slice(0, backwardShown).map((s) => s.node));
  const current = backwardShown
    ? trace.steps[backwardShown - 1].node
    : forwardShown
    ? ops[forwardShown - 1]
    : null;
  $("bp-ops").replaceChildren(
    ...ops.map((v, i) =>
      cellRow(
        [
          names.get(v),
          expression(v),
          i < forwardShown ? fmt(v.data) : "—",
          visited.has(v) ? fmt(g.get(v)) : "—",
        ],
        v === current ? "is-current" : "",
      )
    ),
  );

  const done = backwardShown === trace.steps.length;
  let worst = 0;
  const leaves = [...Object.entries(net.params), ["x1", net.nodes.x1], ["x2", net.nodes.x2]];
  $("bp-leaves").replaceChildren(
    ...leaves.map(([k, node]) => {
      let nudged = "";
      if (done && k in net.params) {
        const d = numericalDerivative(
          (val) =>
            buildTinyNetwork({ ...config, params: { ...config.params, [k]: val } }).loss.data,
          config.params[k],
        );
        worst = Math.max(worst, Math.abs(d - g.get(node)));
        nudged = fmt(d);
      }
      return cellRow([k, fmt(node.data), backwardShown ? fmt(g.get(node) ?? 0) : "—", nudged]);
    }),
  );

  const forwardDone = forwardShown === ops.length;
  $("bp-loss").textContent = forwardDone ? fmt(net.loss.data) : "—";
  $("bp-forward").disabled = forwardDone;
  $("bp-backward").disabled = !forwardDone || done;
  $("bp-progress").textContent =
    `forward ${forwardShown} of ${ops.length} · backward ${backwardShown} of ${trace.steps.length}`;
  const check = $("bp-check");
  check.dataset.maxDiff = done ? String(worst) : "";
  check.textContent = !done
    ? ""
    : worst < 1e-6
    ? `All 9 weight gradients from backprop match a finite difference (largest gap ${
      worst.toExponential(1)
    }).`
    : `Backprop and finite differences disagree by up to ${worst.toExponential(1)}.`;
  if (message) $("bp-status").textContent = message;
}

$("bp-forward").addEventListener("click", () => {
  if (forwardShown >= ops.length) return;
  forwardShown++;
  const v = ops[forwardShown - 1];
  const tail = forwardShown === ops.length ? " The forward pass is done; now go backwards." : "";
  update(`${names.get(v)} = ${expression(v)} = ${fmt(v.data)}.${tail}`);
});

$("bp-backward").addEventListener("click", () => {
  if (forwardShown < ops.length || backwardShown >= trace.steps.length) return;
  backwardShown++;
  const s = trace.steps[backwardShown - 1];
  const pushes = s.contributions
    .map((c) => `∂L/∂${names.get(c.child)} += ${fmt(s.grad)} × ${fmt(c.local)}`)
    .join("; ");
  update(`∂L/∂${names.get(s.node)} = ${fmt(s.grad)}. ${pushes}.`);
});

$("bp-all").addEventListener("click", () => {
  forwardShown = ops.length;
  backwardShown = trace.steps.length;
  update(`Both passes run: loss ${fmt(net.loss.data)}, and every gradient is filled in.`);
});

$("bp-reset").addEventListener("click", () => {
  forwardShown = 0;
  backwardShown = 0;
  update("Reset. Press Forward to compute the first node.");
});

$("bp-train").addEventListener("click", () => {
  const r = trainTinyNetworkStep(config, LR);
  config = r.config;
  build();
  forwardShown = ops.length;
  backwardShown = trace.steps.length;
  update(
    `One step of gradient descent with η = ${LR}: the loss went from ${fmt(r.lossBefore)} to ${
      fmt(r.lossAfter)
    }.`,
  );
});

$("bp-random").addEventListener("click", () => {
  const rng = mulberry32(++seed);
  config = {
    ...config,
    params: Object.fromEntries(
      Object.keys(config.params).map((k) => [k, Math.round((rng() * 2 - 1) * 100) / 100]),
    ),
  };
  build();
  forwardShown = 0;
  backwardShown = 0;
  update("New random weights. Press Forward to start.");
});

build();
update("Press Forward to compute the network one node at a time, then Backward.");
