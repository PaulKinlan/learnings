// tests/nn-chapters.test.js — the maths behind the neural-network chapter pages.
// Every assertion compares the code with something independent of it: a finite difference,
// a value worked out by hand in the comment, or a property that must hold (sums to 1, etc.).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ACTIVATIONS, numericalDerivative, gradientThroughDepth,
  PERCEPTRON_DATASETS, perceptronOutput, trainPerceptron, xorTwoLayer,
  softmax, entropy, crossEntropy, softmaxCrossEntropyGrad, expectedCalibrationError, mse,
  SURFACES, OPTIMIZERS, RACE_LEARNING_RATES, makeOptimizer, optimizePath,
  Value, buildTinyNetwork, TINY_NET_DEFAULTS, singleNeuronChainRule, backwardTrace, trainTinyNetworkStep,
  conv2d, conv2dOutputSize, conv2dMulti, zeroPad, maxPool2d, relu2d, KERNEL_PRESETS,
  scaledDotProductAttention, matmul, transpose, layerNorm, rmsNorm, rope, dot, moeRoute,
  attentionHbmTraffic, kvCacheBytes, quantizeAffineInt8, quantizeSymmetricInt8, mulberry32,
} from '../site/neural-networks/math.js';

const close = (a, b, tol = 1e-6, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} expected ${b}, got ${a}`);

test('every activation derivative matches a central finite difference', () => {
  const xs = [-3, -1.2, -0.4, 0.3, 0.9, 2.5];
  for (const [key, act] of Object.entries(ACTIVATIONS)) {
    if (key === 'step') continue; // derivative is 0 everywhere it exists
    for (const x of xs) close(act.d(x), numericalDerivative(act.f, x), 1e-5, `${key}'(${x})`);
  }
});

test('sigmoid saturates and its derivative never exceeds 0.25', () => {
  for (let x = -10; x <= 10; x += 0.01) assert.ok(ACTIVATIONS.sigmoid.d(x) <= 0.25 + 1e-12);
  close(ACTIVATIONS.sigmoid.d(0), 0.25, 1e-12);
});

test('gradient through 10 sigmoid layers vanishes; through 10 ReLU layers it survives', () => {
  const sig = gradientThroughDepth('sigmoid', { layers: 10, weight: 1, input: 0.5 });
  assert.equal(sig.length, 10);
  assert.ok(sig[0].grad > 0 && sig[0].grad < 0.25 ** 10 + 1e-15, `sigmoid layer-1 grad ${sig[0].grad}`);
  // each layer can only shrink it: |grad_l| ≤ 0.25 · |grad_{l+1}|
  for (let l = 0; l < 9; l++) assert.ok(sig[l].grad <= 0.25 * sig[l + 1].grad + 1e-18);
  const relu = gradientThroughDepth('relu', { layers: 10, weight: 1, input: 0.5 });
  for (const row of relu) assert.equal(row.grad, 1);
});

test('perceptron learns AND, OR and NAND, and every learned rule classifies all four points', () => {
  for (const key of ['and', 'or', 'nand']) {
    const r = trainPerceptron(PERCEPTRON_DATASETS[key], { maxEpochs: 100 });
    assert.ok(r.converged, `${key} should converge`);
    for (const p of PERCEPTRON_DATASETS[key]) assert.equal(perceptronOutput(r.w, r.b, p.x), p.y);
  }
});

test('perceptron never converges on XOR (not linearly separable), but two layers solve it', () => {
  const r = trainPerceptron(PERCEPTRON_DATASETS.xor, { maxEpochs: 1000 });
  assert.equal(r.converged, false);
  assert.ok(r.errorsPerEpoch.every((e) => e > 0));
  for (const p of PERCEPTRON_DATASETS.xor) assert.equal(xorTwoLayer(p.x).out, p.y);
});

test('softmax: sums to 1, shift-invariant, temperature sharpens and flattens', () => {
  const z = [2, 1, 0.1];
  close(softmax(z).reduce((a, b) => a + b, 0), 1, 1e-12);
  softmax(z).forEach((p, i) => close(p, softmax(z.map((v) => v + 100))[i], 1e-12));
  assert.ok(softmax(z, 0.05)[0] > 0.999);
  softmax(z, 1000).forEach((p) => close(p, 1 / 3, 1e-3));
  assert.ok(entropy(softmax(z, 0.5)) < entropy(softmax(z, 1)) && entropy(softmax(z, 1)) < entropy(softmax(z, 2)));
  assert.throws(() => softmax(z, 0));
});

test('softmax cross-entropy gradient (p − y) / T matches finite differences', () => {
  const z = [1.5, -0.3, 0.8, 0.1];
  for (const T of [0.5, 1, 2]) {
    const g = softmaxCrossEntropyGrad(z, 2, T);
    z.forEach((_, i) => {
      const f = (v) => { const zz = [...z]; zz[i] = v; return crossEntropy(softmax(zz, T), 2); };
      close(g[i], numericalDerivative(f, z[i]), 1e-6, `T=${T} dz${i}`);
    });
  }
});

test('MSE and calibration error behave as defined', () => {
  close(mse([1, 2, 3], [1, 2, 5]), 4 / 3, 1e-12);
  // 80%-confident predictions that are right 80% of the time: perfectly calibrated
  const calibrated = Array.from({ length: 10 }, (_, i) => ({ confidence: 0.85, correct: i < 8 })).map((p) => ({ ...p, confidence: 0.8 }));
  close(expectedCalibrationError(calibrated), 0, 1e-12);
  // 99% confident, right half the time: ECE = |0.99 − 0.5| = 0.49
  const overconfident = Array.from({ length: 10 }, (_, i) => ({ confidence: 0.99, correct: i % 2 === 0 }));
  close(expectedCalibrationError(overconfident), 0.49, 1e-12);
});

test('every loss surface gradient matches finite differences, and the stated minima are minima', () => {
  for (const [key, s] of Object.entries(SURFACES)) {
    for (const [x, y] of [[0.3, -0.7], [-1.1, 1.4], [0.9, 0.2]]) {
      const [gx, gy] = s.grad(x, y);
      close(gx, numericalDerivative((v) => s.f(v, y), x), 1e-4, `${key} ∂x`);
      close(gy, numericalDerivative((v) => s.f(x, v), y), 1e-4, `${key} ∂y`);
    }
    const [mx, my] = s.minimum;
    const [gx, gy] = s.grad(mx, my);
    assert.ok(Math.hypot(gx, gy) < 0.02, `${key} gradient at minimum ${gx},${gy}`);
  }
  const dw = SURFACES.doubleWell;
  assert.ok(dw.f(-1.036, 0) < dw.f(0.963, 0), 'left well is the global minimum');
});

test('SGD on the bowl converges below the stability limit and diverges above it (lr < 2/λmax = 0.2)', () => {
  const stable = optimizePath('bowl', 'sgd', { lr: 0.15, steps: 300 });
  assert.equal(stable.diverged, false);
  assert.ok(stable.path.at(-1).loss < 1e-6);
  const unstable = optimizePath('bowl', 'sgd', { lr: 0.21, steps: 300 });
  assert.ok(unstable.diverged || unstable.path.at(-1).loss > unstable.path[0].loss);
});

test('every optimizer reduces the loss on the bowl and on Rosenbrock', () => {
  const lrs = { sgd: 0.001, momentum: 0.0005, adagrad: 0.3, rmsprop: 0.01, adam: 0.02, adamw: 0.02 };
  for (const [name, lr] of Object.entries(lrs)) {
    for (const surface of ['bowl', 'rosenbrock']) {
      const r = optimizePath(surface, name, { lr, steps: 400 });
      assert.equal(r.diverged, false, `${name} on ${surface} diverged`);
      assert.ok(r.path.at(-1).loss < r.path[0].loss, `${name} on ${surface} did not descend`);
    }
  }
});

test('plain SGD from the default start settles in the local (right-hand) well of the double well', () => {
  const r = optimizePath('doubleWell', 'sgd', { lr: 0.05, steps: 500 });
  assert.ok(r.path.at(-1).x > 0.9 && r.path.at(-1).x < 1.0, `ended at x=${r.path.at(-1).x}`);
});

test('the gradient-descent race shows what the chapter says it shows', () => {
  const race = (surface, opt) => optimizePath(surface, opt, { lr: RACE_LEARNING_RATES[surface][opt], steps: 300 });
  const gap = (surface, r) => r.path.at(-1).loss - SURFACES[surface].f(...SURFACES[surface].minimum);
  // Bowl: all six reach the minimum; SGD zig-zags across the steep direction on the way.
  for (const opt of Object.keys(OPTIMIZERS)) {
    const r = race('bowl', opt);
    assert.equal(r.diverged, false, `bowl ${opt}`);
    assert.ok(gap('bowl', r) < 1e-6, `bowl ${opt} gap ${gap('bowl', r)}`);
  }
  const sgdBowl = race('bowl', 'sgd').path;
  assert.ok(Math.sign(sgdBowl[1].y) !== Math.sign(sgdBowl[2].y), 'SGD zig-zags in y on the bowl');
  // Rosenbrock: SGD and momentum get closest; Adam and AdamW crawl; RMSProp lags; AdaGrad stalls.
  for (const opt of ['sgd', 'momentum']) assert.ok(gap('rosenbrock', race('rosenbrock', opt)) < 0.01, opt);
  for (const opt of ['adam', 'adamw']) {
    const g = gap('rosenbrock', race('rosenbrock', opt));
    assert.ok(g > 0.01 && g < 1, `${opt} gap ${g}`);
  }
  const ada = gap('rosenbrock', race('rosenbrock', 'adagrad'));
  const rms = gap('rosenbrock', race('rosenbrock', 'rmsprop'));
  assert.ok(ada > 1 && rms > 1 && rms < ada, `adagrad ${ada}, rmsprop ${rms}`);
  assert.ok(optimizePath('rosenbrock', 'sgd', { lr: 0.005, steps: 300 }).diverged, 'SGD at 0.005 blows up in the valley');
  // Double well: only momentum crosses the hump into the deeper well.
  for (const opt of Object.keys(OPTIMIZERS)) {
    const x = race('doubleWell', opt).path.at(-1).x;
    if (opt === 'momentum') assert.ok(x < -0.9, `momentum ended at x=${x}`);
    else assert.ok(x > 0.9, `${opt} ended at x=${x}`);
  }
  // The stability demo: on the bowl λmax = 10, so the limit is η < 0.2.
  assert.ok(optimizePath('bowl', 'sgd', { lr: 0.19, steps: 300 }).path.at(-1).loss < 1e-6);
  assert.ok(optimizePath('bowl', 'sgd', { lr: 0.21, steps: 300 }).diverged);
});

test('double well has a shallower local minimum; Rosenbrock is ~2,500x more curved across than along at (1, 1)', () => {
  const dw = SURFACES.doubleWell;
  const [lx, ly] = dw.localMinimum;
  assert.ok(Math.hypot(...dw.grad(lx, ly)) < 1e-3);
  assert.ok(dw.f(lx, ly) > dw.f(...dw.minimum));
  // Hessian at (1, 1) is [[802, -400], [-400, 200]]: eigenvalues ~1001.6 and ~0.4
  const h = 1e-4, g = SURFACES.rosenbrock.grad;
  const hxx = (g(1 + h, 1)[0] - g(1 - h, 1)[0]) / (2 * h);
  const hyy = (g(1, 1 + h)[1] - g(1, 1 - h)[1]) / (2 * h);
  const hxy = (g(1, 1 + h)[0] - g(1, 1 - h)[0]) / (2 * h);
  const mean = (hxx + hyy) / 2, r = Math.hypot((hxx - hyy) / 2, hxy);
  const ratio = (mean + r) / (mean - r);
  assert.ok(ratio > 2400 && ratio < 2600, `curvature ratio ${ratio}`);
  close(2 / (mean + r), 0.002, 1e-4);
});

test('AdamW decays weights even with zero gradient; Adam does not (decoupled weight decay)', () => {
  const adamw = makeOptimizer('adamw', { lr: 0.1, weightDecay: 0.5 });
  const adam = makeOptimizer('adam', { lr: 0.1 });
  let p = [2, -4];
  let q = [2, -4];
  for (let i = 0; i < 3; i++) { p = adamw(p, [0, 0]); q = adam(q, [0, 0]); }
  close(p[0], 2 * 0.95 ** 3, 1e-12); close(p[1], -4 * 0.95 ** 3, 1e-12);
  assert.deepEqual(q, [2, -4]);
});

test('backprop: a value used twice receives both gradient contributions', () => {
  const x = new Value(3);
  const y = x.mul(x).add(x); // y = x² + x, dy/dx = 2x + 1 = 7
  y.backward();
  close(x.grad, 7, 1e-12);
});

test('backprop through the 2-2-1 network matches finite differences for every parameter', () => {
  const { loss, params } = buildTinyNetwork(TINY_NET_DEFAULTS);
  loss.backward();
  const lossWith = (key, v) => {
    const cfg = structuredClone(TINY_NET_DEFAULTS);
    cfg.params[key] = v;
    return buildTinyNetwork(cfg).loss.data;
  };
  for (const [key, node] of Object.entries(params)) {
    close(node.grad, numericalDerivative((v) => lossWith(key, v), TINY_NET_DEFAULTS.params[key]), 1e-7, key);
  }
  assert.ok(loss.topo().length > 20, 'graph contains every intermediate');
});

test('single-neuron chain rule: the three factors multiply to the autodiff gradient', () => {
  const cfg = { w: 0.7, b: -0.2, x: 1.5, y: 1 };
  const c = singleNeuronChainRule(cfg);
  const w = new Value(cfg.w), b = new Value(cfg.b);
  const L = w.mul(cfg.x).add(b).sigmoid().sub(cfg.y).pow(2);
  L.backward();
  close(c.dL_dw, w.grad, 1e-12); close(c.dL_db, b.grad, 1e-12);
  close(c.dL_dw, c.dL_da * c.da_dz * c.dz_dw, 1e-15);
  close(c.dL_dw, numericalDerivative((v) => singleNeuronChainRule({ ...cfg, w: v }).loss, cfg.w), 1e-8);
});

test('conv2d: hand-worked example, output size formula, padding and stride', () => {
  const img = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  // Sobel vertical: −1+3 −8+12 −7+9 = 8
  const r = conv2d(img, KERNEL_PRESETS.sobelVertical.k);
  assert.deepEqual(r.output, [[8]]);
  assert.equal(r.steps.length, 1);
  assert.equal(r.steps[0].products.length, 9);
  close(r.steps[0].products.reduce((a, p) => a + p.product, 0), 8, 1e-12);
  assert.equal(conv2dOutputSize(8, 3, 0, 1), 6);
  assert.equal(conv2dOutputSize(8, 3, 1, 1), 8);
  assert.equal(conv2dOutputSize(8, 3, 1, 2), 4);
  assert.equal(conv2dOutputSize(8, 3, 0, 2), 3);
  const rng = mulberry32(7);
  const big = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => Math.round(rng() * 9)));
  for (const [s, p] of [[1, 0], [1, 1], [2, 0], [2, 1]]) {
    const out = conv2d(big, KERNEL_PRESETS.sharpen.k, { stride: s, padding: p });
    assert.equal(out.outH, conv2dOutputSize(8, 3, p, s));
    assert.equal(out.output[0].length, conv2dOutputSize(8, 3, p, s));
  }
  // identity kernel with 'same' padding reproduces the input exactly
  assert.deepEqual(conv2d(big, KERNEL_PRESETS.identity.k, { padding: 1 }).output, big);
  assert.equal(zeroPad([[1]], 1).flat().reduce((a, b) => a + b, 0), 1);
});

test('multi-channel convolution sums the per-channel results; pooling and ReLU', () => {
  const a = [[1, 0, 2], [0, 1, 0], [2, 0, 1]];
  const b = [[0, 1, 0], [1, 0, 1], [0, 1, 0]];
  const k1 = KERNEL_PRESETS.outline.k, k2 = KERNEL_PRESETS.identity.k;
  const multi = conv2dMulti([a, b], [[k1, k2]], [0.5]);
  const expected = conv2d(a, k1).output[0][0] + conv2d(b, k2).output[0][0] + 0.5;
  close(multi[0][0][0], expected, 1e-12);
  assert.deepEqual(maxPool2d([[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 2, 2], [3, 0, 1, 9]]), [[4, 5], [3, 9]]);
  assert.deepEqual(relu2d([[-1, 2], [0, -3]]), [[0, 2], [0, 0]]);
});

test('attention: rows sum to 1, causal mask, output = weights × V, scaling by √d_k', () => {
  const Q = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]];
  const K = [[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1]];
  const V = [[1, 2], [3, 4], [5, 6]];
  const r = scaledDotProductAttention(Q, K, V);
  r.weights.forEach((row) => close(row.reduce((a, b) => a + b, 0), 1, 1e-12));
  close(r.scores[0][2], dot(Q[0], K[2]) / 2, 1e-12); // √4 = 2
  const expectedOut = matmul(r.weights, V);
  r.output.forEach((row, i) => row.forEach((v, j) => close(v, expectedOut[i][j], 1e-12)));
  const c = scaledDotProductAttention(Q, K, V, { causal: true });
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) assert.equal(c.weights[i][j], 0);
  assert.equal(c.weights[0][0], 1);
});

test('LayerNorm centres and scales; RMSNorm only rescales', () => {
  const x = [2, 4, 6, 8, 10];
  const ln = layerNorm(x);
  close(ln.reduce((a, b) => a + b, 0) / 5, 0, 1e-9);
  close(ln.reduce((a, b) => a + b * b, 0) / 5, 1, 1e-4);
  const rn = rmsNorm(x);
  close(Math.sqrt(rn.reduce((a, b) => a + b * b, 0) / 5), 1, 1e-4);
  assert.ok(rn.every((v) => v > 0), 'RMSNorm keeps the sign pattern (no mean subtraction)');
});

test('RoPE preserves length and makes q·k depend only on the position difference', () => {
  const q = [0.3, -1.2, 0.8, 0.5, -0.1, 0.9];
  const k = [1.1, 0.4, -0.6, 0.2, 0.7, -0.3];
  close(Math.hypot(...rope(q, 17)), Math.hypot(...q), 1e-12);
  const base = dot(rope(q, 5), rope(k, 2));
  for (const shift of [1, 10, 250]) close(dot(rope(q, 5 + shift), rope(k, 2 + shift)), base, 1e-9);
  assert.ok(Math.abs(dot(rope(q, 5), rope(k, 4)) - base) > 1e-3, 'a different offset gives a different score');
});

test('MoE top-k routing picks the k largest logits with weights summing to 1', () => {
  const r = moeRoute([0.1, 2.0, -1.0, 1.5, 0.3, 0.0, -0.5, 0.9], 2);
  assert.deepEqual(r.map((e) => e.expert), [1, 3]);
  close(r.reduce((a, e) => a + e.weight, 0), 1, 1e-12);
  assert.ok(r[0].weight > r[1].weight);
});

test('FlashAttention IO model and KV-cache arithmetic', () => {
  const t = attentionHbmTraffic({ seqLen: 4096, headDim: 64, sramElements: 100_000 });
  assert.ok(t.flash < t.standard && t.ratio > 10);
  // Llama-2-7B shape, fp16, 4096 tokens: 2 × 32 × 32 × 128 × 4096 × 2 bytes = 2 GiB
  assert.equal(kvCacheBytes({ layers: 32, kvHeads: 32, headDim: 128, seqLen: 4096 }), 2 * 1024 ** 3);
  // grouped-query attention with 8 KV heads: a quarter of that
  assert.equal(kvCacheBytes({ layers: 32, kvHeads: 8, headDim: 128, seqLen: 4096 }), 512 * 1024 ** 2);
});

test('int8 quantisation: zero is exact, error is at most half a step', () => {
  const values = [-1.37, -0.5, 0, 0.21, 0.73, 1.9, 2.44];
  const a = quantizeAffineInt8(values);
  assert.equal(a.dequant[2], 0);
  assert.ok(a.q.every((q) => q >= -128 && q <= 127 && Number.isInteger(q)));
  assert.ok(a.maxError <= a.scale * 0.5 + 1e-9, `affine maxError ${a.maxError} scale ${a.scale}`);
  const s = quantizeSymmetricInt8(values);
  assert.equal(s.zeroPoint, 0);
  assert.ok(s.q.every((q) => q >= -127 && q <= 127));
  assert.ok(s.maxError <= s.scale * 0.5 + 1e-9);
});

// ── backpropagation.html ────────────────────────────────────────────────────────────────

test('the step-by-step backward trace gives exactly the gradients Value.backward() gives', () => {
  const { loss } = buildTinyNetwork(TINY_NET_DEFAULTS);
  const { steps, grads, order } = backwardTrace(loss);
  loss.backward();
  for (const v of order) close(grads.get(v), v.grad, 1e-12, v.label || v.op);
  assert.equal(steps[0].node, loss);
  assert.equal(steps[0].grad, 1);
  assert.equal(steps.length, order.filter((v) => v.children.length).length);
  // A node is visited only after every node that uses it: nothing pushes into a visited node.
  const visited = new Set();
  for (const s of steps) {
    for (const c of s.contributions) assert.ok(!visited.has(c.child), 'gradient pushed into an already-visited node');
    visited.add(s.node);
  }
});

test('one gradient step on the 2-2-1 network lowers the loss, and twenty keep lowering it', () => {
  const first = trainTinyNetworkStep(TINY_NET_DEFAULTS, 0.5);
  assert.ok(first.lossAfter < first.lossBefore, `${first.lossBefore} -> ${first.lossAfter}`);
  let cfg = TINY_NET_DEFAULTS;
  let last = first.lossBefore;
  for (let i = 0; i < 20; i++) {
    const s = trainTinyNetworkStep(cfg, 0.5);
    assert.ok(s.lossAfter < last, `step ${i}: ${last} -> ${s.lossAfter}`);
    last = s.lossAfter;
    cfg = s.config;
  }
  assert.ok(last < first.lossBefore * 0.7, `after 20 steps ${last}`);
});

test('matrix backprop: for Y = XW, dL/dW = Xᵀ G and dL/dX = G Wᵀ (checked by finite differences)', () => {
  const X = [[1, -2, 0.5], [0.3, 0.8, -1.1]];
  const W = [[0.2, -0.4], [1.5, 0.1], [-0.7, 0.9]];
  const G = [[0.6, -1.2], [2.0, 0.4]]; // the upstream gradient dL/dY, for L = Σ G ⊙ Y
  const L = (Xm, Wm) => matmul(Xm, Wm).reduce((a, row, i) => a + row.reduce((b, y, j) => b + G[i][j] * y, 0), 0);
  const dW = matmul(transpose(X), G);
  const dX = matmul(G, transpose(W));
  W.forEach((row, i) => row.forEach((_, j) => {
    const f = (v) => { const Wm = W.map((r) => [...r]); Wm[i][j] = v; return L(X, Wm); };
    close(dW[i][j], numericalDerivative(f, W[i][j]), 1e-6, `dW[${i}][${j}]`);
  }));
  X.forEach((row, i) => row.forEach((_, j) => {
    const f = (v) => { const Xm = X.map((r) => [...r]); Xm[i][j] = v; return L(Xm, W); };
    close(dX[i][j], numericalDerivative(f, X[i][j]), 1e-6, `dX[${i}][${j}]`);
  }));
});
