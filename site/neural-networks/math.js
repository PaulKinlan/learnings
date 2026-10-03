// site/neural-networks/math.js
// The maths behind every interactive widget in the chapter pages, in one place.
//
// Pure functions only: no DOM, no randomness without a seed, no network. Every export is
// exercised by tests/nn-chapters.test.js, so a number a page shows comes from code that a
// test has checked against an independent result (a finite difference, a closed form, or a
// known property such as "softmax sums to 1").

// ── Deterministic randomness ─────────────────────────────────────────────────────────────

/** Small seeded PRNG (mulberry32). Same seed, same sequence, on every machine. */
export function mulberry32(seed = 1) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Activation functions and their derivatives ───────────────────────────────────────────

export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const GELU_K = Math.sqrt(2 / Math.PI);

/**
 * Each entry: f(x) and its derivative d(x). GELU uses the tanh approximation from
 * Hendrycks & Gimpel (2016), the form GPT-2 shipped, because JavaScript has no Math.erf.
 */
export const ACTIVATIONS = {
  step: {
    name: "Step (Heaviside)",
    era: "1943 · McCulloch & Pitts",
    f: (x) => (x >= 0 ? 1 : 0),
    d: () => 0,
  },
  sigmoid: {
    name: "Sigmoid",
    era: "1980s · backprop era",
    f: sigmoid,
    d: (x) => {
      const s = sigmoid(x);
      return s * (1 - s);
    },
  },
  tanh: {
    name: "Tanh",
    era: "1980s–1990s",
    f: Math.tanh,
    d: (x) => 1 - Math.tanh(x) ** 2,
  },
  relu: {
    name: "ReLU",
    era: "2010–2012 · Nair & Hinton, AlexNet",
    f: (x) => (x > 0 ? x : 0),
    d: (x) => (x > 0 ? 1 : 0),
  },
  leaky_relu: {
    name: "Leaky ReLU (α = 0.01)",
    era: "2013 · Maas et al.",
    f: (x) => (x > 0 ? x : 0.01 * x),
    d: (x) => (x > 0 ? 1 : 0.01),
  },
  gelu: {
    name: "GELU (tanh approximation)",
    era: "2016 · Hendrycks & Gimpel; BERT, GPT",
    f: (x) => 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x ** 3))),
    d: (x) => {
      const t = Math.tanh(GELU_K * (x + 0.044715 * x ** 3));
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_K * (1 + 3 * 0.044715 * x * x);
    },
  },
  silu: {
    name: "SiLU / Swish",
    era: "2017 · Ramachandran et al.",
    f: (x) => x * sigmoid(x),
    d: (x) => {
      const s = sigmoid(x);
      return s * (1 + x * (1 - s));
    },
  },
};

/** Central finite difference: the independent check every analytic derivative is tested against. */
export function numericalDerivative(f, x, h = 1e-5) {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/**
 * A chain of `layers` scalar neurons, h_l = act(w · h_{l-1}). Returns, per layer, the
 * pre-activation z, the output h, and the gradient that reaches that layer's INPUT from the
 * final output: ∂h_L/∂h_{l-1} = Π_{j=l..L} w · act'(z_j). This is the vanishing-gradient
 * product made visible: with sigmoid, act' ≤ 0.25, so ten layers shrink it below 0.25¹⁰.
 */
export function gradientThroughDepth(activation, { layers = 10, weight = 1, input = 0.5 } = {}) {
  const act = ACTIVATIONS[activation];
  if (!act) throw new Error(`unknown activation ${activation}`);
  const zs = [];
  const hs = [];
  let h = input;
  for (let l = 0; l < layers; l++) {
    const z = weight * h;
    h = act.f(z);
    zs.push(z);
    hs.push(h);
  }
  const grads = new Array(layers);
  let g = 1;
  for (let l = layers - 1; l >= 0; l--) {
    g *= weight * act.d(zs[l]);
    grads[l] = g;
  }
  return zs.map((z, l) => ({ layer: l + 1, z, h: hs[l], grad: grads[l] }));
}

// ── The perceptron ───────────────────────────────────────────────────────────────────────

export const PERCEPTRON_DATASETS = {
  and: [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 0 },
    { x: [1, 0], y: 0 },
    { x: [1, 1], y: 1 },
  ],
  or: [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 1 },
  ],
  nand: [
    { x: [0, 0], y: 1 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ],
  xor: [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ],
};

/** Rosenblatt's unit: fire (1) when the weighted sum crosses the threshold, else 0. */
export function perceptronOutput(w, b, x) {
  return w[0] * x[0] + w[1] * x[1] + b > 0 ? 1 : 0;
}

/** One application of the perceptron learning rule: w ← w + η (y − ŷ) x, b ← b + η (y − ŷ). */
export function perceptronUpdate(w, b, point, lr = 1) {
  const yhat = perceptronOutput(w, b, point.x);
  const err = point.y - yhat;
  return {
    w: [w[0] + lr * err * point.x[0], w[1] + lr * err * point.x[1]],
    b: b + lr * err,
    yhat,
    err,
  };
}

/**
 * Cycle through the data until an epoch makes no mistakes. Novikoff (1962) proved this ends
 * for any linearly separable data set; for XOR it never does, which is Minsky & Papert's point.
 */
export function trainPerceptron(points, { lr = 1, maxEpochs = 100, w = [0, 0], b = 0 } = {}) {
  let weights = [...w];
  let bias = b;
  const errorsPerEpoch = [];
  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    let errors = 0;
    for (const p of points) {
      const next = perceptronUpdate(weights, bias, p, lr);
      if (next.err !== 0) errors++;
      weights = next.w;
      bias = next.b;
    }
    errorsPerEpoch.push(errors);
    if (errors === 0) {
      return { converged: true, epochs: epoch + 1, w: weights, b: bias, errorsPerEpoch };
    }
  }
  return { converged: false, epochs: maxEpochs, w: weights, b: bias, errorsPerEpoch };
}

/** XOR with one hidden layer of hand-set threshold units: XOR = AND(OR, NAND). */
export function xorTwoLayer(x) {
  const or = perceptronOutput([1, 1], -0.5, x);
  const nand = perceptronOutput([-1, -1], 1.5, x);
  return { or, nand, out: perceptronOutput([1, 1], -1.5, [or, nand]) };
}

// ── Losses, softmax and temperature ──────────────────────────────────────────────────────

/** Softmax with temperature T. Subtracting the max first keeps exp() from overflowing. */
export function softmax(logits, temperature = 1) {
  if (!(temperature > 0)) throw new Error("temperature must be > 0");
  const scaled = logits.map((z) => z / temperature);
  const m = Math.max(...scaled);
  const e = scaled.map((z) => Math.exp(z - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function entropy(p) {
  return -p.reduce((acc, pi) => (pi > 0 ? acc + pi * Math.log(pi) : acc), 0);
}

export function crossEntropy(probs, target) {
  return -Math.log(probs[target]);
}

export function mse(pred, target) {
  let s = 0;
  for (let i = 0; i < pred.length; i++) s += (pred[i] - target[i]) ** 2;
  return s / pred.length;
}

export function oneHot(index, n) {
  return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
}

/** ∂CE/∂logits for softmax(z / T) with a one-hot target: (p − y) / T. */
export function softmaxCrossEntropyGrad(logits, target, temperature = 1) {
  const p = softmax(logits, temperature);
  return p.map((pi, i) => (pi - (i === target ? 1 : 0)) / temperature);
}

/**
 * Expected calibration error over equal-width confidence bins: the gap between how sure a
 * model says it is and how often it is right, weighted by how many predictions fall in each bin.
 */
export function expectedCalibrationError(predictions, bins = 10) {
  const buckets = Array.from({ length: bins }, () => ({ n: 0, conf: 0, correct: 0 }));
  for (const { confidence, correct } of predictions) {
    const i = Math.min(bins - 1, Math.floor(confidence * bins));
    buckets[i].n++;
    buckets[i].conf += confidence;
    buckets[i].correct += correct ? 1 : 0;
  }
  const total = predictions.length || 1;
  return buckets.reduce(
    (acc, b) => (b.n ? acc + (b.n / total) * Math.abs(b.conf / b.n - b.correct / b.n) : acc),
    0,
  );
}

// ── Gradient descent on 2-D loss surfaces ────────────────────────────────────────────────

export const SURFACES = {
  bowl: {
    name: "Elongated bowl",
    note: "Convex, but 10× steeper in y than in x (condition number 10).",
    domain: { x: [-3, 3], y: [-3, 3] },
    start: [-2.5, 2.5],
    minimum: [0, 0],
    f: (x, y) => 0.5 * x * x + 5 * y * y,
    grad: (x, y) => [x, 10 * y],
  },
  rosenbrock: {
    name: "Rosenbrock valley",
    note: "A long curved valley; the minimum at (1, 1) is easy to find and slow to reach.",
    domain: { x: [-2, 2], y: [-1, 3] },
    start: [-1.5, 2.2],
    minimum: [1, 1],
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    grad: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
  },
  doubleWell: {
    name: "Tilted double well",
    note: "Non-convex: a shallow local minimum on the right, the global one on the left.",
    domain: { x: [-2, 2], y: [-2, 2] },
    start: [1.8, 1.5],
    minimum: [-1.036, 0],
    localMinimum: [0.9601, 0],
    f: (x, y) => (x * x - 1) ** 2 + 0.3 * x + 0.5 * y * y,
    grad: (x, y) => [4 * x * (x * x - 1) + 0.3, y],
  },
};

export const OPTIMIZERS = {
  sgd: { name: "SGD", year: "1847 (Cauchy) · 1951 (Robbins–Monro)" },
  momentum: { name: "Momentum", year: "1964 · Polyak" },
  adagrad: { name: "AdaGrad", year: "2011 · Duchi, Hazan & Singer" },
  rmsprop: { name: "RMSProp", year: "2012 · Hinton (lecture 6e)" },
  adam: { name: "Adam", year: "2014 · Kingma & Ba" },
  adamw: { name: "AdamW", year: "2017 · Loshchilov & Hutter" },
};

/**
 * Learning rates for the 300-step race on each surface, picked by a grid search. Momentum's
 * rates are about 10× smaller than SGD's because with μ = 0.9 its effective step is η / (1 − μ).
 * tests/nn-chapters.test.js checks every outcome gradient-descent.html describes for these rates.
 */
export const RACE_LEARNING_RATES = {
  bowl: { sgd: 0.15, momentum: 0.01, adagrad: 0.5, rmsprop: 0.01, adam: 0.05, adamw: 0.05 },
  rosenbrock: { sgd: 0.002, momentum: 0.002, adagrad: 0.5, rmsprop: 0.1, adam: 0.05, adamw: 0.05 },
  doubleWell: { sgd: 0.05, momentum: 0.01, adagrad: 0.3, rmsprop: 0.02, adam: 0.05, adamw: 0.05 },
};

/** Returns step(p, g) → next p. State (velocity, moment estimates, step count) lives in the closure. */
export function makeOptimizer(
  name,
  {
    lr = 0.01,
    momentum = 0.9,
    rho = 0.9,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
    weightDecay = 0.01,
  } = {},
) {
  const n = 2;
  const v = new Array(n).fill(0);
  const s = new Array(n).fill(0);
  const m = new Array(n).fill(0);
  let t = 0;
  const adam = (p, g, decoupled) => {
    t++;
    return p.map((pi, i) => {
      m[i] = beta1 * m[i] + (1 - beta1) * g[i];
      s[i] = beta2 * s[i] + (1 - beta2) * g[i] * g[i];
      const mHat = m[i] / (1 - beta1 ** t);
      const sHat = s[i] / (1 - beta2 ** t);
      const next = pi - (lr * mHat) / (Math.sqrt(sHat) + eps);
      return decoupled ? next - lr * weightDecay * pi : next;
    });
  };
  switch (name) {
    case "sgd":
      return (p, g) => p.map((pi, i) => pi - lr * g[i]);
    case "momentum":
      return (p, g) =>
        p.map((pi, i) => {
          v[i] = momentum * v[i] - lr * g[i];
          return pi + v[i];
        });
    case "adagrad":
      return (p, g) =>
        p.map((pi, i) => {
          s[i] += g[i] * g[i];
          return pi - (lr * g[i]) / (Math.sqrt(s[i]) + eps);
        });
    case "rmsprop":
      return (p, g) =>
        p.map((pi, i) => {
          s[i] = rho * s[i] + (1 - rho) * g[i] * g[i];
          return pi - (lr * g[i]) / (Math.sqrt(s[i]) + eps);
        });
    case "adam":
      return (p, g) => adam(p, g, false);
    case "adamw":
      return (p, g) => adam(p, g, true);
    default:
      throw new Error(`unknown optimizer ${name}`);
  }
}

/** Run an optimizer on a surface. Stops early (diverged: true) if the point blows up. */
export function optimizePath(surfaceKey, optimizerName, { start, steps = 200, ...opts } = {}) {
  const surface = SURFACES[surfaceKey];
  if (!surface) throw new Error(`unknown surface ${surfaceKey}`);
  const step = makeOptimizer(optimizerName, opts);
  let p = [...(start ?? surface.start)];
  const path = [{ x: p[0], y: p[1], loss: surface.f(p[0], p[1]) }];
  for (let i = 0; i < steps; i++) {
    p = step(p, surface.grad(p[0], p[1]));
    const loss = surface.f(p[0], p[1]);
    if (!Number.isFinite(loss) || Math.abs(p[0]) > 1e6 || Math.abs(p[1]) > 1e6) {
      return { path, diverged: true };
    }
    path.push({ x: p[0], y: p[1], loss });
  }
  return { path, diverged: false };
}

// ── Reverse-mode automatic differentiation (backpropagation) ─────────────────────────────

let valueIds = 0;

/**
 * A scalar in a computation graph. Each operation records its inputs and the LOCAL
 * derivative of its output with respect to each input; backward() then applies the chain
 * rule in reverse topological order. A value used twice gets both contributions added,
 * which is the multivariable chain rule.
 */
export class Value {
  constructor(data, { children = [], localGrads = [], op = "", label = "" } = {}) {
    this.id = valueIds++;
    this.data = data;
    this.grad = 0;
    this.children = children;
    this.localGrads = localGrads;
    this.op = op;
    this.label = label;
  }

  static of(x) {
    return x instanceof Value ? x : new Value(x, { label: String(x) });
  }

  add(other) {
    const o = Value.of(other);
    return new Value(this.data + o.data, { children: [this, o], localGrads: [1, 1], op: "+" });
  }

  sub(other) {
    const o = Value.of(other);
    return new Value(this.data - o.data, { children: [this, o], localGrads: [1, -1], op: "−" });
  }

  mul(other) {
    const o = Value.of(other);
    return new Value(this.data * o.data, {
      children: [this, o],
      localGrads: [o.data, this.data],
      op: "×",
    });
  }

  pow(n) {
    return new Value(this.data ** n, {
      children: [this],
      localGrads: [n * this.data ** (n - 1)],
      op: `^${n}`,
    });
  }

  tanh() {
    const t = Math.tanh(this.data);
    return new Value(t, { children: [this], localGrads: [1 - t * t], op: "tanh" });
  }

  sigmoid() {
    const s = sigmoid(this.data);
    return new Value(s, { children: [this], localGrads: [s * (1 - s)], op: "σ" });
  }

  relu() {
    return new Value(this.data > 0 ? this.data : 0, {
      children: [this],
      localGrads: [this.data > 0 ? 1 : 0],
      op: "ReLU",
    });
  }

  named(label) {
    this.label = label;
    return this;
  }

  /** Every node this value depends on, inputs before outputs. */
  topo() {
    const order = [];
    const seen = new Set();
    const visit = (v) => {
      if (seen.has(v)) return;
      seen.add(v);
      for (const c of v.children) visit(c);
      order.push(v);
    };
    visit(this);
    return order;
  }

  /** Seed ∂out/∂out = 1, then push gradients from each node to its inputs, last node first. */
  backward() {
    const order = this.topo();
    for (const v of order) v.grad = 0;
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) {
      const v = order[i];
      v.children.forEach((c, k) => {
        c.grad += v.grad * v.localGrads[k];
      });
    }
    return order;
  }
}

export const TINY_NET_DEFAULTS = {
  inputs: [1, -2],
  target: 1,
  params: { w11: 0.5, w12: -0.3, b1: 0.1, w21: -0.4, w22: 0.8, b2: 0.0, v1: 1.2, v2: -0.7, c: 0.2 },
};

/**
 * A 2-2-1 network: two tanh hidden units, a sigmoid output, squared-error loss.
 * Returns the loss Value and named handles to every intermediate, so a page can lay the
 * graph out and step through the forward and backward passes node by node.
 */
export function buildTinyNetwork({ inputs, target, params } = TINY_NET_DEFAULTS) {
  const p = {};
  for (const [k, val] of Object.entries(params)) p[k] = new Value(val, { label: k });
  const x1 = new Value(inputs[0], { label: "x1" });
  const x2 = new Value(inputs[1], { label: "x2" });
  const z1 = p.w11.mul(x1).add(p.w12.mul(x2)).add(p.b1).named("z1");
  const h1 = z1.tanh().named("h1");
  const z2 = p.w21.mul(x1).add(p.w22.mul(x2)).add(p.b2).named("z2");
  const h2 = z2.tanh().named("h2");
  const z3 = p.v1.mul(h1).add(p.v2.mul(h2)).add(p.c).named("z3");
  const yhat = z3.sigmoid().named("ŷ");
  const loss = yhat.sub(target).pow(2).named("L");
  return { loss, params: p, nodes: { x1, x2, z1, h1, z2, h2, z3, yhat }, target };
}

/**
 * The chain rule for one sigmoid neuron with squared error, written out factor by factor:
 * L = (σ(w x + b) − y)², so ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w = 2(a − y) · a(1 − a) · x.
 */
export function singleNeuronChainRule({ w, b, x, y }) {
  const z = w * x + b;
  const a = sigmoid(z);
  const dL_da = 2 * (a - y);
  const da_dz = a * (1 - a);
  const dz_dw = x;
  return { z, a, loss: (a - y) ** 2, dL_da, da_dz, dz_dw, dL_dw: dL_da * da_dz * dz_dw, dL_db: dL_da * da_dz };
}

// ── Convolution, pooling and the CNN forward pass ────────────────────────────────────────

export const KERNEL_PRESETS = {
  sobelHorizontal: {
    name: "Sobel — horizontal edges",
    note: "Responds where brightness changes from top to bottom.",
    k: [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
  },
  sobelVertical: {
    name: "Sobel — vertical edges",
    note: "Responds where brightness changes from left to right.",
    k: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
  },
  sharpen: {
    name: "Sharpen",
    note: "Boosts the centre pixel against its four neighbours.",
    k: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  blur: {
    name: "Blur (Gaussian 3×3)",
    note: "A weighted average: [1 2 1; 2 4 2; 1 2 1] / 16.",
    k: [
      [1 / 16, 2 / 16, 1 / 16],
      [2 / 16, 4 / 16, 2 / 16],
      [1 / 16, 2 / 16, 1 / 16],
    ],
  },
  outline: {
    name: "Outline (Laplacian)",
    note: "Zero on flat regions, large wherever the centre differs from its surroundings.",
    k: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1],
    ],
  },
  identity: {
    name: "Identity",
    note: "Copies the input: a sanity check that the machinery works.",
    k: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
  },
};

/** Output size along one axis: ⌊(W − K + 2P) / S⌋ + 1. */
export function conv2dOutputSize(size, kernel, padding, stride) {
  return Math.floor((size - kernel + 2 * padding) / stride) + 1;
}

/** Surround a 2-D grid with `p` rings of zeros. */
export function zeroPad(input, p) {
  if (p === 0) return input.map((row) => [...row]);
  const w = input[0].length + 2 * p;
  const out = [];
  for (let i = 0; i < p; i++) out.push(new Array(w).fill(0));
  for (const row of input) out.push([...new Array(p).fill(0), ...row, ...new Array(p).fill(0)]);
  for (let i = 0; i < p; i++) out.push(new Array(w).fill(0));
  return out;
}

/**
 * Single-channel 2-D convolution as deep-learning libraries compute it (strictly a
 * cross-correlation: the kernel is not flipped). Returns the output map and, for each output
 * cell, the receptive field position, the nine element-wise products and their sum — exactly
 * what the sliding-kernel visualiser shows one step at a time.
 */
export function conv2d(input, kernel, { stride = 1, padding = 0, bias = 0 } = {}) {
  const K = kernel.length;
  const padded = zeroPad(input, padding);
  const outH = conv2dOutputSize(input.length, K, padding, stride);
  const outW = conv2dOutputSize(input[0].length, K, padding, stride);
  if (outH < 1 || outW < 1) throw new Error("kernel larger than padded input");
  const output = [];
  const steps = [];
  for (let oy = 0; oy < outH; oy++) {
    const row = [];
    for (let ox = 0; ox < outW; ox++) {
      const iy = oy * stride;
      const ix = ox * stride;
      const products = [];
      let sum = bias;
      for (let ky = 0; ky < K; ky++) {
        for (let kx = 0; kx < K; kx++) {
          const v = padded[iy + ky][ix + kx];
          const prod = v * kernel[ky][kx];
          products.push({ ky, kx, value: v, weight: kernel[ky][kx], product: prod });
          sum += prod;
        }
      }
      row.push(sum);
      steps.push({ oy, ox, iy, ix, products, sum });
    }
    output.push(row);
  }
  return { output, steps, padded, outH, outW };
}

/** Multi-channel convolution: input [C][H][W], filters [F][C][K][K], bias [F] → [F][H'][W']. */
export function conv2dMulti(input, filters, bias, { stride = 1, padding = 0 } = {}) {
  return filters.map((filter, f) => {
    let acc = null;
    for (let c = 0; c < input.length; c++) {
      const { output } = conv2d(input[c], filter[c], { stride, padding });
      acc = acc ? acc.map((row, y) => row.map((v, x) => v + output[y][x])) : output;
    }
    return acc.map((row) => row.map((v) => v + (bias?.[f] ?? 0)));
  });
}

export function relu2d(map) {
  return map.map((row) => row.map((v) => (v > 0 ? v : 0)));
}

export function maxPool2d(map, size = 2, stride = 2) {
  const outH = Math.floor((map.length - size) / stride) + 1;
  const outW = Math.floor((map[0].length - size) / stride) + 1;
  const out = [];
  for (let y = 0; y < outH; y++) {
    const row = [];
    for (let x = 0; x < outW; x++) {
      let m = -Infinity;
      for (let dy = 0; dy < size; dy++) {
        for (let dx = 0; dx < size; dx++) m = Math.max(m, map[y * stride + dy][x * stride + dx]);
      }
      row.push(m);
    }
    out.push(row);
  }
  return out;
}

// ── Small dense linear algebra (for attention and normalisation demos) ───────────────────

export function matmul(A, B) {
  const n = A.length;
  const k = B.length;
  const m = B[0].length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      for (let j = 0; j < m; j++) out[i][j] += a * B[p][j];
    }
  }
  return out;
}

export function transpose(A) {
  return A[0].map((_, j) => A.map((row) => row[j]));
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * softmax(Q Kᵀ / √d_k) V, row by row. With `causal`, position i may only attend to j ≤ i
 * (the mask GPT-style decoders use). Returns the raw scores, the attention weights and output.
 */
export function scaledDotProductAttention(Q, K, V, { causal = false } = {}) {
  const dk = K[0].length;
  const raw = matmul(Q, transpose(K)).map((row) => row.map((v) => v / Math.sqrt(dk)));
  const scores = raw.map((row, i) => row.map((v, j) => (causal && j > i ? -Infinity : v)));
  const weights = scores.map((row) => {
    const m = Math.max(...row);
    const e = row.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  });
  return { scores, weights, output: matmul(weights, V) };
}

// ── Modern components: normalisation, RoPE, MoE, FlashAttention, KV cache ────────────────

export function layerNorm(x, { gamma, beta, eps = 1e-5 } = {}) {
  const n = x.length;
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const inv = 1 / Math.sqrt(variance + eps);
  return x.map((v, i) => (v - mean) * inv * (gamma?.[i] ?? 1) + (beta?.[i] ?? 0));
}

/** RMSNorm (Zhang & Sennrich, 2019): rescale by the root-mean-square, no mean subtraction, no bias. */
export function rmsNorm(x, { gamma, eps = 1e-5 } = {}) {
  const rms = Math.sqrt(x.reduce((a, b) => a + b * b, 0) / x.length + eps);
  return x.map((v, i) => (v / rms) * (gamma?.[i] ?? 1));
}

/**
 * Rotary position embedding (Su et al., 2021). Each pair (x_2i, x_2i+1) is rotated by angle
 * pos · θ_i with θ_i = base^(−2i/d). Rotations compose, so ⟨RoPE(q, m), RoPE(k, n)⟩ depends
 * only on m − n: attention sees relative position without any position vector being added.
 */
export function rope(vec, pos, base = 10000) {
  const d = vec.length;
  if (d % 2 !== 0) throw new Error("RoPE needs an even dimension");
  const out = new Array(d);
  for (let i = 0; i < d / 2; i++) {
    const theta = pos * base ** ((-2 * i) / d);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const a = vec[2 * i];
    const b = vec[2 * i + 1];
    out[2 * i] = a * c - b * s;
    out[2 * i + 1] = a * s + b * c;
  }
  return out;
}

/** Top-k mixture-of-experts routing, Mixtral style: softmax over the k selected router logits. */
export function moeRoute(logits, k = 2) {
  const ranked = logits.map((z, i) => ({ i, z })).sort((a, b) => b.z - a.z).slice(0, k);
  const w = softmax(ranked.map((r) => r.z));
  return ranked.map((r, j) => ({ expert: r.i, logit: r.z, weight: w[j] }));
}

/**
 * Order-of-magnitude HBM traffic (in elements) from Dao et al. (2022), Theorem 2:
 * standard attention Θ(N·d + N²), FlashAttention Θ(N²·d² / M) for SRAM of M elements.
 * Constants are dropped, so these are model numbers for comparison, not measurements.
 */
export function attentionHbmTraffic({ seqLen, headDim, sramElements }) {
  const standard = seqLen * headDim + seqLen * seqLen;
  const flash = (seqLen * seqLen * headDim * headDim) / sramElements;
  return { standard, flash, ratio: standard / flash };
}

/** Bytes held by a KV cache: 2 (keys and values) × layers × KV heads × head dim × tokens × batch × bytes. */
export function kvCacheBytes({ layers, kvHeads, headDim, seqLen, batch = 1, bytesPerElement = 2 }) {
  return 2 * layers * kvHeads * headDim * seqLen * batch * bytesPerElement;
}

// ── Quantisation (the LiteRT chapter) ────────────────────────────────────────────────────

/** Affine (asymmetric) int8: real ≈ (q − zeroPoint) × scale, q ∈ [−128, 127]. */
export function quantizeAffineInt8(values) {
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const scale = (max - min) / 255 || 1;
  // zeroPoint = qmin − round(min / scale): the integer that real 0.0 maps to, exactly.
  const zeroPoint = -128 - Math.round(min / scale);
  const q = values.map((v) => Math.max(-128, Math.min(127, Math.round(v / scale) + zeroPoint)));
  const dequant = q.map((qi) => (qi - zeroPoint) * scale);
  const maxError = Math.max(...values.map((v, i) => Math.abs(v - dequant[i])));
  return { scale, zeroPoint, q, dequant, maxError };
}

/** Symmetric int8 (zero point fixed at 0), the usual choice for weights: q ∈ [−127, 127]. */
export function quantizeSymmetricInt8(values) {
  const maxAbs = Math.max(...values.map(Math.abs));
  const scale = maxAbs / 127 || 1;
  const q = values.map((v) => Math.max(-127, Math.min(127, Math.round(v / scale))));
  const dequant = q.map((qi) => qi * scale);
  const maxError = Math.max(...values.map((v, i) => Math.abs(v - dequant[i])));
  return { scale, zeroPoint: 0, q, dequant, maxError };
}
