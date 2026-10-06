// site/neural-networks/gradient-descent.js — the loss-landscape race on gradient-descent.html.
// Every path is computed by optimizePath() in math.js, and tests/nn-chapters.test.js checks the
// outcomes this page describes: who converges, who diverges, and who escapes the local minimum.
import { OPTIMIZERS, RACE_LEARNING_RATES, SURFACES, optimizePath } from "./math.js";

const $ = (id) => document.getElementById(id);
const map = $("gd-map");
const chart = $("gd-chart");
const ctx = map.getContext("2d");
const cctx = chart.getContext("2d");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const ORDER = Object.keys(OPTIMIZERS);
// These match the .sw-* swatches in chapter.css.
const COLORS = {
  sgd: "#d1495b",
  momentum: "#f2a541",
  adagrad: "#2f9e44",
  rmsprop: "#7b2cbf",
  adam: "#e83e8c",
  adamw: "#111827",
  alt: "#8a5a00",
};

const ui = {
  surface: $("gd-surface"),
  optimizer: $("gd-optimizer"),
  lr: $("gd-lr"),
  lrOut: $("gd-lr-out"),
  momentum: $("gd-momentum"),
  momentumOut: $("gd-momentum-out"),
  steps: $("gd-steps"),
  stepsOut: $("gd-steps-out"),
  startX: $("gd-start-x"),
  startY: $("gd-start-y"),
  startOut: $("gd-start-out"),
  status: $("gd-status"),
  results: $("gd-results"),
};

const state = { surface: "bowl", start: [...SURFACES.bowl.start], runs: [] };
const backgrounds = new Map();
let animation = 0;

const surface = () => SURFACES[state.surface];
// A preset (the race rate for the chosen optimiser) is kept exactly: the log-scale slider only
// has 0.01 steps, so 0.15 would otherwise come back as 10^-0.82 = 0.151. Moving the slider
// switches to the slider's value.
let exactRate = null;
const learningRate = () => exactRate ?? 10 ** Number(ui.lr.value);
const num = (v) =>
  Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)
    ? v.toExponential(1)
    : String(Number(v.toPrecision(3)));

function toPx(x, y) {
  const d = surface().domain;
  return [
    ((x - d.x[0]) / (d.x[1] - d.x[0])) * map.width,
    (1 - (y - d.y[0]) / (d.y[1] - d.y[0])) * map.height,
  ];
}

function fromPx(px, py) {
  const d = surface().domain;
  return [
    d.x[0] + (px / map.width) * (d.x[1] - d.x[0]),
    d.y[0] + (1 - py / map.height) * (d.y[1] - d.y[0]),
  ];
}

/** The loss surface as a heat map on a log scale, with thin contour bands. Cached per surface. */
function background() {
  if (backgrounds.has(state.surface)) return backgrounds.get(state.surface);
  const s = surface();
  const { width: w, height: h } = map;
  const values = new Float64Array(w * h);
  let lo = Infinity;
  let hi = -Infinity;
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const [x, y] = fromPx(px + 0.5, py + 0.5);
      const v = s.f(x, y);
      values[py * w + px] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const img = ctx.createImageData(w, h);
  const span = Math.log1p(hi - lo) || 1;
  for (let i = 0; i < values.length; i++) {
    const t = Math.log1p(values[i] - lo) / span; // 0 at the lowest point, 1 at the highest
    const band = t * 14;
    const edge = Math.abs(band - Math.round(band)) < 0.06 ? 0.82 : 1;
    img.data[i * 4] = (12 + 225 * t) * edge;
    img.data[i * 4 + 1] = (52 + 190 * t) * edge;
    img.data[i * 4 + 2] = (110 + 140 * t) * edge;
    img.data[i * 4 + 3] = 255;
  }
  backgrounds.set(state.surface, img);
  return img;
}

function ring(x, y, r, fill, stroke) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function visiblePoints(run, progress) {
  return run.path.slice(0, Math.max(1, Math.ceil(run.path.length * progress)));
}

function drawMap(progress) {
  ctx.putImageData(background(), 0, 0);
  const s = surface();
  ctx.lineJoin = "round";
  for (const run of state.runs) {
    const pts = visiblePoints(run, progress);
    for (const [width, color] of [[5, "rgba(255,255,255,0.85)"], [2.2, run.color]]) {
      ctx.beginPath();
      pts.forEach((p, i) => {
        const [px, py] = toPx(p.x, p.y);
        if (i) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
      });
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
    const last = pts.at(-1);
    ring(...toPx(last.x, last.y), 4.5, run.color, "#ffffff");
  }
  ring(...toPx(...s.minimum), 7, "#ffffff", "#0b1f33");
  ring(...toPx(...state.start), 6, "#ffffff", "#000000");
}

/** Loss above the true minimum against step, on a log scale. */
function drawChart(progress) {
  const { width: w, height: h } = chart;
  cctx.fillStyle = "#ffffff";
  cctx.fillRect(0, 0, w, h);
  const s = surface();
  const fmin = s.f(...s.minimum);
  const gap = (v) => Math.log10(Math.max(v - fmin, 1e-12));
  const steps = Math.max(1, ...state.runs.map((r) => r.path.length - 1));
  const all = state.runs.flatMap((r) => r.path.map((p) => gap(p.loss)));
  const top = Math.min(8, Math.ceil(Math.max(1, ...all)));
  const bottom = Math.max(-12, Math.floor(Math.min(top - 1, ...all)));
  const pad = { l: 48, r: 10, t: 10, b: 24 };
  const X = (i) => pad.l + (i / steps) * (w - pad.l - pad.r);
  const Y = (v) =>
    pad.t + ((top - Math.min(top, Math.max(bottom, v))) / (top - bottom)) * (h - pad.t - pad.b);
  cctx.lineWidth = 1;
  cctx.strokeStyle = "#bacbd9";
  cctx.fillStyle = "#445e73";
  cctx.font = "12px system-ui, sans-serif";
  const stride = Math.max(1, Math.ceil((top - bottom) / 6));
  for (let e = bottom; e <= top; e += stride) {
    cctx.beginPath();
    cctx.moveTo(pad.l, Y(e));
    cctx.lineTo(w - pad.r, Y(e));
    cctx.stroke();
    cctx.fillText(`1e${e}`, 4, Y(e) + 4);
  }
  cctx.fillText(`step 0 → ${steps}`, pad.l, h - 6);
  for (const run of state.runs) {
    const pts = visiblePoints(run, progress);
    cctx.beginPath();
    pts.forEach((p, i) => {
      if (i) cctx.lineTo(X(i), Y(gap(p.loss)));
      else cctx.moveTo(X(i), Y(gap(p.loss)));
    });
    cctx.lineWidth = 2;
    cctx.strokeStyle = run.color;
    cctx.stroke();
  }
}

function outcome(run) {
  if (run.diverged) return "Diverged: each step overshot further until the numbers blew up";
  const s = surface();
  const last = run.path.at(-1);
  const g = last.loss - s.f(...s.minimum);
  const dist = Math.hypot(last.x - s.minimum[0], last.y - s.minimum[1]);
  if (dist < 0.02 && g < 1e-3) return "Reached the minimum";
  if (s.localMinimum && Math.hypot(last.x - s.localMinimum[0], last.y - s.localMinimum[1]) < 0.1) {
    return "Stuck in the shallower well";
  }
  if (Math.hypot(...s.grad(last.x, last.y)) < 1e-3) return "Stuck at a flat point";
  if (last.loss > run.path[0].loss) return "Getting worse";
  return "Still on its way";
}

function renderTable() {
  const s = surface();
  const fmin = s.f(...s.minimum);
  ui.results.replaceChildren(
    ...state.runs.map((run) => {
      const last = run.path.at(-1);
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      const swatch = document.createElement("span");
      swatch.className = `swatch sw-${run.swatch}`;
      name.append(swatch, run.label);
      const cells = [
        num(run.lr),
        run.diverged ? "—" : num(Math.max(last.loss - fmin, 0)),
        run.diverged ? "—" : `(${last.x.toFixed(2)}, ${last.y.toFixed(2)})`,
        outcome(run),
      ];
      tr.append(
        name,
        ...cells.map((text) => {
          const td = document.createElement("td");
          td.textContent = text;
          return td;
        }),
      );
      return tr;
    }),
  );
}

function describe() {
  const s = surface();
  map.setAttribute(
    "aria-label",
    `Contour map of the ${s.name}. Darker is lower loss. The white ring is the minimum at (${
      s.minimum.join(", ")
    }); the black-ringed dot is the start at (${state.start.map((v) => v.toFixed(2)).join(", ")}). ${
      state.runs.length
    } path${state.runs.length === 1 ? "" : "s"} drawn; the table below lists where each ends.`,
  );
}

function animate() {
  cancelAnimationFrame(animation);
  if (reduceMotion) {
    drawMap(1);
    drawChart(1);
    return;
  }
  const t0 = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - t0) / 900);
    drawMap(p);
    drawChart(p);
    if (p < 1) animation = requestAnimationFrame(tick);
  };
  animation = requestAnimationFrame(tick);
}

function update(runs, message) {
  state.runs = runs;
  renderTable();
  describe();
  animate();
  ui.status.textContent = message ??
    runs.map((r) => `${r.label}: ${outcome(r).split(":")[0].toLowerCase()}`).join(" · ");
}

function runOne(optimizer, rate, { momentum, steps, label, colorKey } = {}) {
  const { path, diverged } = optimizePath(state.surface, optimizer, {
    lr: rate,
    momentum: momentum ?? Number(ui.momentum.value),
    steps: steps ?? Number(ui.steps.value),
    start: [...state.start],
  });
  const key = colorKey ?? optimizer;
  return { optimizer, lr: rate, path, diverged, swatch: key, color: COLORS[key], label: label ?? OPTIMIZERS[optimizer].name };
}

function race() {
  return ORDER.map((o) => runOne(o, RACE_LEARNING_RATES[state.surface][o], { momentum: 0.9 }));
}

function syncControls() {
  const v = learningRate();
  ui.lrOut.textContent = num(v);
  ui.lr.setAttribute("aria-valuetext", `learning rate ${num(v)}`);
  ui.momentumOut.textContent = ui.momentum.value;
  ui.momentum.disabled = ui.optimizer.value !== "momentum";
  ui.stepsOut.textContent = ui.steps.value;
}

function setLearningRate(v) {
  exactRate = v;
  ui.lr.value = Math.log10(v).toFixed(2);
  syncControls();
}

function syncStart() {
  const d = surface().domain;
  for (const [input, [lo, hi], v] of [[ui.startX, d.x, state.start[0]], [ui.startY, d.y, state.start[1]]]) {
    input.min = String(lo);
    input.max = String(hi);
    input.step = String((hi - lo) / 200);
    input.value = String(v);
  }
  ui.startOut.textContent = `(${state.start[0].toFixed(2)}, ${state.start[1].toFixed(2)})`;
}

function changeSurface() {
  state.surface = ui.surface.value;
  state.start = [...surface().start];
  syncStart();
  setLearningRate(RACE_LEARNING_RATES[state.surface][ui.optimizer.value]);
  update(race(), `${surface().name}: ${surface().note} All six optimisers raced from the default start.`);
}

ui.surface.addEventListener("change", changeSurface);
ui.optimizer.addEventListener("change", () => setLearningRate(RACE_LEARNING_RATES[state.surface][ui.optimizer.value]));
ui.lr.addEventListener("input", () => {
  exactRate = null;
  syncControls();
});
for (const input of [ui.momentum, ui.steps]) input.addEventListener("input", syncControls);
for (const input of [ui.startX, ui.startY]) {
  input.addEventListener("input", () => {
    state.start = [Number(ui.startX.value), Number(ui.startY.value)];
    ui.startOut.textContent = `(${state.start[0].toFixed(2)}, ${state.start[1].toFixed(2)})`;
    describe();
    drawMap(1);
  });
}

$("gd-run").addEventListener("click", () => {
  const opt = ui.optimizer.value;
  update([...state.runs.filter((r) => r.optimizer !== opt), runOne(opt, learningRate())]);
});
$("gd-race").addEventListener("click", () => update(race()));
$("gd-clear").addEventListener("click", () => update([], "Paths cleared."));
$("gd-compare").addEventListener("click", () => {
  ui.surface.value = "bowl";
  state.surface = "bowl";
  state.start = [...SURFACES.bowl.start];
  syncStart();
  update([
    runOne("sgd", 0.19, { steps: 300, label: "SGD, η = 0.19" }),
    runOne("sgd", 0.21, { steps: 300, label: "SGD, η = 0.21", colorKey: "alt" }),
  ]);
});
map.addEventListener("pointerdown", (e) => {
  const rect = map.getBoundingClientRect();
  state.start = fromPx(
    ((e.clientX - rect.left) / rect.width) * map.width,
    ((e.clientY - rect.top) / rect.height) * map.height,
  );
  syncStart();
  $("gd-run").click();
});

syncStart();
setLearningRate(RACE_LEARNING_RATES.bowl[ui.optimizer.value]);
update(race(), "All six optimisers raced down the elongated bowl.");
