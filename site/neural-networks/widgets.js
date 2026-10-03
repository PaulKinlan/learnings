// Shared, independently scoped playgrounds for the hub and chapter pages.
import {
  PERCEPTRON_DATASETS, perceptronOutput, perceptronUpdate, SURFACES, optimizePath,
  TINY_NET_DEFAULTS, buildTinyNetwork, KERNEL_PRESETS, conv2d, maxPool2d,
  softmax, entropy, crossEntropy, mse, oneHot, scaledDotProductAttention, dot,
  ACTIVATIONS, gradientThroughDepth, rmsNorm, rope, moeRoute, attentionHbmTraffic,
  kvCacheBytes, quantizeAffineInt8,
} from './math.js';

const fmt = n => Number.isFinite(n) ? (Math.abs(n) < 0.0001 && n !== 0 ? n.toExponential(3) : Number(n.toFixed(4)).toString()) : String(n);
const select = (key, label, options) => `<label>${label}<select data-control="${key}">${options.map(([v, text]) => `<option value="${v}">${text}</option>`).join('')}</select></label>`;
const slider = (key, label, min, max, value, step = 0.1) => `<label>${label} <output data-value="${key}">${value}</output><input data-control="${key}" type="range" min="${min}" max="${max}" value="${value}" step="${step}"></label>`;
const button = (key, text) => `<button type="button" data-action="${key}">${text}</button>`;
const canvas = label => `<canvas width="560" height="340" role="img" aria-label="${label}">${label}. Numerical results are below.</canvas>`;
function setup(el, html) {
  el.classList.add('widget', 'playground');
  el.innerHTML = html;
  const get = key => el.querySelector(`[data-control="${key}"]`);
  const val = key => Number(get(key).value);
  const action = (key, fn) => el.querySelector(`[data-action="${key}"]`).addEventListener('click', fn);
  const status = text => { el.querySelector('[data-status]').textContent = text; };
  el.addEventListener('input', e => {
    const out = el.querySelector(`[data-value="${e.target.dataset.control}"]`);
    if (out) out.textContent = e.target.value;
  });
  return { get, val, action, status };
}
const readout = '<p class="readout" data-status role="status"></p>';
function plot(c, domain = [-2, 2, -2, 2]) {
  const ctx = c.getContext('2d');
  const [xmin, xmax, ymin, ymax] = domain;
  const X = x => 35 + (x - xmin) / (xmax - xmin) * (c.width - 55);
  const Y = y => c.height - 30 - (y - ymin) / (ymax - ymin) * (c.height - 50);
  ctx.fillStyle = '#0b0f19'; ctx.fillRect(0, 0, c.width, c.height);
  ctx.strokeStyle = '#64748b'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(xmin), Y(0)); ctx.lineTo(X(xmax), Y(0));
  ctx.moveTo(X(0), Y(ymin)); ctx.lineTo(X(0), Y(ymax)); ctx.stroke();
  ctx.fillStyle = '#f8fafc'; ctx.font = '14px system-ui';
  ctx.fillText(String(xmin), X(xmin), c.height - 8); ctx.fillText(String(xmax), X(xmax) - 10, c.height - 8);
  ctx.fillText(String(ymax), 3, Y(ymax) + 5); ctx.fillText(String(ymin), 3, Y(ymin));
  return { ctx, X, Y };
}
function line(ctx, points, color) {
  ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
  points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
}
function perceptron(el) {
  const ui = setup(el, `<h3>Move a decision boundary</h3><p>Try AND, then XOR. Blue circles are class 1; red squares are class 0. White rings mark mistakes. The line is w₁x₁ + w₂x₂ + b = 0.</p><div class="controls">${select('preset', 'Logic task', Object.keys(PERCEPTRON_DATASETS).map(k => [k, k.toUpperCase()]))}${slider('w1', 'Weight w₁', -4, 4, 1)}${slider('w2', 'Weight w₂', -4, 4, 1)}${slider('b', 'Bias b', -4, 4, -1.5)}</div>${canvas('Perceptron plane: horizontal x₁, vertical x₂')}<div class="actions">${button('step', 'Train Step')}${button('reset', 'Reset weights')}</div>${readout}<p data-proof></p>`);
  let step = 0;
  function draw(message = '') {
    const data = PERCEPTRON_DATASETS[ui.get('preset').value];
    const w = [ui.val('w1'), ui.val('w2')], b = ui.val('b');
    const { ctx, X, Y } = plot(el.querySelector('canvas'), [-0.5, 1.5, -0.5, 1.5]);
    ctx.save(); ctx.beginPath(); ctx.rect(35, 20, 505, 290); ctx.clip();
    if (w[1] !== 0) line(ctx, [-0.5, 1.5].map(x => [X(x), Y(-(w[0] * x + b) / w[1])]), '#38bdf8');
    else if (w[0] !== 0) line(ctx, [[X(-b / w[0]), Y(-0.5)], [X(-b / w[0]), Y(1.5)]], '#38bdf8');
    ctx.restore();
    let correct = 0;
    const predictions = data.map(p => {
      const right = perceptronOutput(w, b, p.x) === p.y; correct += right;
      ctx.fillStyle = p.y ? '#38bdf8' : '#fb7185';
      ctx.beginPath();
      if (p.y) ctx.arc(X(p.x[0]), Y(p.x[1]), 10, 0, Math.PI * 2);
      else ctx.rect(X(p.x[0]) - 9, Y(p.x[1]) - 9, 18, 18);
      ctx.fill();
      if (!right) { ctx.strokeStyle = '#f8fafc'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(p.x[0]), Y(p.x[1]), 15, 0, Math.PI * 2); ctx.stroke(); }
      ctx.fillStyle = '#f8fafc'; ctx.fillText(`(${p.x}) → ${p.y}`, X(p.x[0]) - 30, Y(p.x[1]) + 32);
      return `(${p.x}) predicted ${perceptronOutput(w, b, p.x)}, target ${p.y}`;
    });
    ui.status(`${correct}/4 correct. Boundary: ${fmt(w[0])}x₁ + ${fmt(w[1])}x₂ + ${fmt(b)} = 0. ${message} ${predictions.join('; ')}.${w.every(v => v === 0) ? ' Both weights are zero: there is no separating line.' : ''}`);
    el.querySelector('[data-proof]').textContent = ui.get('preset').value === 'xor'
      ? 'XOR is impossible for one line: (0,0) requires b ≤ 0 and (1,1) requires w₁+w₂+b ≤ 0. But the two positive corners require w₁+b > 0 and w₂+b > 0, hence w₁+w₂+2b > 0 — a contradiction. Training keeps making mistakes, not progress to 4/4.'
      : 'Train Step visits one point at a time: w ← w + 0.2(target − prediction)x; b ← b + 0.2(target − prediction).';
  }
  function reset() { step = 0; ['w1', 'w2', 'b'].forEach((k, i) => { ui.get(k).value = [1, 1, -1.5][i]; ui.get(k).dispatchEvent(new Event('input', { bubbles: true })); }); draw(); }
  ui.action('step', () => {
    const data = PERCEPTRON_DATASETS[ui.get('preset').value], p = data[step++ % 4];
    const next = perceptronUpdate([ui.val('w1'), ui.val('w2')], ui.val('b'), p, 0.2);
    ['w1', 'w2', 'b'].forEach((k, i) => { ui.get(k).value = [...next.w, next.b][i]; el.querySelector(`[data-value="${k}"]`).textContent = fmt(ui.val(k)); });
    draw(`Step ${step}: point (${p.x}), error ${next.err}.`);
  });
  ui.action('reset', reset); ui.get('preset').addEventListener('change', reset);
  el.addEventListener('input', () => draw()); draw();
}
function landscape(el) {
  const ui = setup(el, `<h3>Drop a ball on the loss</h3><p>Click the map or use the start sliders. Darker contours are lower. Run traces 160 updates; a large learning rate can diverge.</p><div class="controls">${select('surface', 'Landscape', [['bowl', 'Convex bowl'], ['doubleWell', 'Double well']])}${select('optimizer', 'Optimizer', [['sgd', 'SGD'], ['momentum', 'Momentum'], ['adam', 'Adam']])}${slider('lr', 'Learning rate η', 0.001, 0.5, 0.05, 0.001)}${slider('x', 'Start x', -2, 2, 1.8)}${slider('y', 'Start y', -2, 2, 1.5)}</div>${canvas('Loss contour map; start coordinates adjustable with sliders')}<div class="actions">${button('run', 'Run')}${button('stop', 'Stop')}${button('reset', 'Reset trajectory')}</div>${readout}`);
  const c = el.querySelector('canvas'); let frame = 0, path = [], shown = 0, diverged = false;
  // The contour pixels are constant while only the trajectory changes.
  const maps = new Map();
  function draw() {
    const s = SURFACES[ui.get('surface').value];
    const { ctx, X, Y } = plot(c);
    if (!maps.has(s)) {
      for (let py = 20; py < 310; py += 3) for (let px = 35; px < 540; px += 3) {
        const z = Math.log1p(s.f((px - 35) / 505 * 4 - 2, (310 - py) / 290 * 4 - 2) + 1);
        const band = Math.floor(z * 12);
        ctx.fillStyle = `hsl(205 65% ${15 + band % 2 * 6 + z * 8}%)`; ctx.fillRect(px, py, 3, 3);
      }
      maps.set(s, ctx.getImageData(0, 0, c.width, c.height));
    } else ctx.putImageData(maps.get(s), 0, 0);
    const pts = path.slice(0, shown);
    ctx.save(); ctx.beginPath(); ctx.rect(35, 20, 505, 290); ctx.clip();
    line(ctx, pts.map(p => [X(p.x), Y(p.y)]), '#f8fafc');
    const p = pts.at(-1) ?? { x: ui.val('x'), y: ui.val('y'), loss: s.f(ui.val('x'), ui.val('y')) };
    ctx.fillStyle = '#fbbf24'; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 7, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
    ui.status(`Step ${Math.max(0, shown - 1)}: (${fmt(p.x)}, ${fmt(p.y)}), loss ${fmt(p.loss)}. ${shown === path.length && diverged ? 'Diverged: lower the learning rate.' : s.note}`);
  }
  function reset() { cancelAnimationFrame(frame); path = []; shown = 0; diverged = false; draw(); }
  ui.action('run', () => {
    cancelAnimationFrame(frame);
    ({ path, diverged } = optimizePath(ui.get('surface').value, ui.get('optimizer').value, { start: [ui.val('x'), ui.val('y')], lr: ui.val('lr'), steps: 160 }));
    shown = 1; const start = performance.now();
    const tick = now => { shown = matchMedia('(prefers-reduced-motion: reduce)').matches ? path.length : Math.min(path.length, 1 + Math.floor((now - start) / 18)); draw(); if (shown < path.length) frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
  });
  ui.action('stop', () => cancelAnimationFrame(frame)); ui.action('reset', reset);
  el.addEventListener('input', reset); el.addEventListener('change', reset);
  c.addEventListener('pointerdown', e => {
    const r = c.getBoundingClientRect();
    ui.get('x').value = Math.max(-2, Math.min(2, ((e.clientX - r.left) / r.width * 560 - 35) / 505 * 4 - 2));
    ui.get('y').value = Math.max(-2, Math.min(2, (310 - (e.clientY - r.top) / r.height * 340) / 290 * 4 - 2));
    ['x', 'y'].forEach(k => ui.get(k).dispatchEvent(new Event('input', { bubbles: true })));
  });
  draw();
}
function backprop(el) {
  const ui = setup(el, `<h3>Trace a 2 → 2 → 1 network</h3><p>Two tanh hidden neurons feed a sigmoid output; L = (ŷ − target)². Step Forward reveals one operation. After the full pass, Step Backward accumulates one node’s chain-rule contributions. Editing any input invalidates both passes.</p><div class="controls">${slider('x1', 'Input x₁', -3, 3, 1)}${slider('x2', 'Input x₂', -3, 3, -2)}${slider('target', 'Target', 0, 1, 1)}</div><details><summary>Edit the nine weights and biases</summary><div class="controls">${Object.entries(TINY_NET_DEFAULTS.params).map(([k, v]) => slider(k, k, -2, 2, v)).join('')}</div></details><div class="actions">${button('forward', 'Step Forward')}${button('backward', 'Step Backward')}${button('reset', 'Reset passes')}</div>${readout}<ol class="computation-graph" data-graph></ol>`);
  let net, order, forward = 0, backward = 0;
  const graph = el.querySelector('[data-graph]');
  let messages = new Map();
  function name(v) { return v.label || `op${order.indexOf(v) + 1} (${v.op})`; }
  function render(active = null) {
    graph.replaceChildren(...order.map((v, i) => {
      const li = document.createElement('li'); li.className = v === active ? 'graph-node active-node' : 'graph-node';
      const h = document.createElement('strong'); h.textContent = `${name(v)} = ${i < forward ? fmt(v.data) : '?'}`;
      const p = document.createElement('p'); p.textContent = v.children.length ? `[${v.children.map(name).join(', ')}] → ${name(v)} via ${v.op}` : 'Input / parameter';
      const grad = document.createElement('p'); grad.textContent = `∂L/∂${name(v)} = ${backward ? fmt(v.grad) : '?'}${messages.has(v) ? `; ${messages.get(v)}` : ''}`;
      li.append(h, p, grad); return li;
    }));
    el.querySelector('[data-action="forward"]').disabled = forward === order.length;
    el.querySelector('[data-action="backward"]').disabled = forward !== order.length || backward === order.length;
  }
  function reset() {
    net = buildTinyNetwork({ inputs: [ui.val('x1'), ui.val('x2')], target: ui.val('target'), params: Object.fromEntries(Object.keys(TINY_NET_DEFAULTS.params).map(k => [k, ui.val(k)])) });
    order = net.loss.topo(); forward = 0; backward = 0; messages = new Map();
    render(); ui.status(`Ready: ${order.length} nodes. Forward pass first; weights stay fixed while tracing derivatives.`);
  }
  ui.action('forward', () => { const v = order[forward++]; render(v); ui.status(`Forward ${forward}/${order.length}: ${name(v)} = ${fmt(v.data)}.${forward === order.length ? ' Loss ready. Now step backward.' : ''}`); });
  ui.action('backward', () => {
    if (!backward) net.loss.grad = 1;
    const v = order[order.length - 1 - backward++];
    const equations = [];
    v.children.forEach((child, k) => {
      const term = v.grad * v.localGrads[k]; child.grad += term;
      const equation = `∂L/∂${name(child)} += ${fmt(v.grad)} × ${fmt(v.localGrads[k])} = ${fmt(term)}`;
      messages.set(child, equation); equations.push(equation);
    });
    render(v); ui.status(`Backward ${backward}/${order.length}: ${equations.join('; ') || `${name(v)} is a leaf; gradient ${fmt(v.grad)}`}.${backward === order.length ? ' All parameter gradients complete.' : ''}`);
  });
  ui.action('reset', reset); el.addEventListener('input', reset); reset();
}
function convolution(el) {
  const ui = setup(el, `<h3>Slide a 3 × 3 kernel</h3><p>Toggle input pixels, choose a filter, then step across the padded image. Outlined cells are the receptive field. Blank output cells have not been visited. This is cross-correlation, as used by CNN libraries.</p><div class="controls">${select('kernel', 'Filter', Object.entries(KERNEL_PRESETS).map(([k, v]) => [k, v.name]))}${select('stride', 'Stride', [[1, '1'], [2, '2']])}${select('padding', 'Zero padding', [[0, '0'], [1, '1']])}</div><div class="kernel-layout"><div><h4>8 × 8 input (click to edit)</h4><div class="pixel-grid" data-image></div></div><div><h4>Kernel</h4><pre data-kernel></pre><h4>Output feature map</h4><div class="table-scroll" data-output></div></div></div><div class="actions">${button('step', 'Slide one step')}${button('run', 'Run convolution')}${button('reset', 'Reset scan')}</div>${readout}<details><summary>Max pooling: 2 × 2, stride 2</summary><p>Maximum of each non-overlapping output block, computed from the complete feature map (including unvisited cells).</p><div class="table-scroll" data-pool></div></details><details><summary>Runnable pure-JavaScript conv2d</summary><p>This is the actual implementation used above. Copy the three functions into a console, then run conv2d([[1,2,3],[4,5,6],[7,8,9]], [[0,0,0],[0,1,0],[0,0,0]]).output. Run convolution executes it on the edited image.</p><pre data-code></pre></details>`);
  let image = Array.from({ length: 8 }, (_, y) => Array.from({ length: 8 }, (_, x) => x > 3 || y === 2 ? 1 : 0));
  let result, cursor = -1, timer = 0;
  const grid = el.querySelector('[data-image]');
  function table(values, visible = Infinity) {
    const t = document.createElement('table');
    values.forEach((row, y) => { const tr = t.insertRow(); row.forEach((v, x) => { tr.insertCell().textContent = y * row.length + x <= visible ? fmt(v) : '·'; }); }); return t;
  }
  function draw() {
    const s = result.steps[cursor];
    [...grid.children].forEach((b, i) => {
      const y = Math.floor(i / 8), x = i % 8; b.textContent = image[y][x]; b.setAttribute('aria-pressed', String(Boolean(image[y][x])));
      b.classList.toggle('lit-pixel', Boolean(image[y][x]));
      b.classList.toggle('receptive', Boolean(s && y + ui.val('padding') >= s.iy && y + ui.val('padding') < s.iy + 3 && x + ui.val('padding') >= s.ix && x + ui.val('padding') < s.ix + 3));
    });
    el.querySelector('[data-output]').replaceChildren(table(result.output, cursor));
    el.querySelector('[data-pool]').replaceChildren(table(maxPool2d(result.output)));
    el.querySelector('[data-kernel]').textContent = KERNEL_PRESETS[ui.get('kernel').value].k.map(r => r.map(fmt).join('\t')).join('\n');
    ui.status(s ? `Output [${s.oy},${s.ox}] = ${s.products.map(p => `${fmt(p.value)}×${fmt(p.weight)}`).join(' + ')} = ${fmt(s.sum)}. ${cursor + 1}/${result.steps.length} cells.` : `${result.outH} × ${result.outW} output. Press Slide one step. ${KERNEL_PRESETS[ui.get('kernel').value].note}`);
    el.querySelector('[data-action="step"]').disabled = cursor === result.steps.length - 1;
  }
  function reset() { clearTimeout(timer); cursor = -1; result = conv2d(image, KERNEL_PRESETS[ui.get('kernel').value].k, { stride: ui.val('stride'), padding: ui.val('padding') }); draw(); }
  image.flat().forEach((_, i) => {
    const b = document.createElement('button'); b.type = 'button'; b.setAttribute('aria-label', `Input row ${Math.floor(i / 8) + 1}, column ${i % 8 + 1}`);
    b.addEventListener('click', () => { image[Math.floor(i / 8)][i % 8] ^= 1; reset(); }); grid.append(b);
  });
  ui.action('step', () => { clearTimeout(timer); if (cursor < result.steps.length - 1) cursor++; draw(); });
  ui.action('run', () => { reset(); const tick = () => { cursor = matchMedia('(prefers-reduced-motion: reduce)').matches ? result.steps.length - 1 : cursor + 1; draw(); if (cursor < result.steps.length - 1) timer = setTimeout(tick, 90); }; tick(); });
  ui.action('reset', reset); el.addEventListener('change', reset);
  import('./math.js').then(({ zeroPad, conv2dOutputSize }) => { el.querySelector('[data-code]').textContent = [zeroPad, conv2dOutputSize, conv2d].map(f => f.toString()).join('\n\n'); });
  reset();
}
function probabilities(el) {
  const ui = setup(el, `<h3>Logits → probabilities</h3><p>Lower temperature sharpens unequal logits; higher temperature flattens them. Tied largest logits stay tied, even as τ approaches zero.</p><div class="controls">${[2, 1, 0, -1].map((v, i) => slider(`z${i}`, `Class ${i + 1} logit`, -5, 5, v)).join('')}${slider('tau', 'Temperature τ', 0.1, 5, 1)}${select('target', 'True class', [0, 1, 2, 3].map(i => [i, `Class ${i + 1}`]))}</div><div data-bars></div>${readout}`);
  function draw() {
    const p = softmax([0, 1, 2, 3].map(i => ui.val(`z${i}`)), ui.val('tau'));
    el.querySelector('[data-bars]').replaceChildren(...p.map((v, i) => {
      const row = document.createElement('div'); row.className = 'probability-row';
      const label = document.createElement('span'); label.textContent = `Class ${i + 1}`;
      const meter = document.createElement('meter'); meter.min = 0; meter.max = 1; meter.value = v; meter.setAttribute('aria-label', `Class ${i + 1} probability`);
      const text = document.createElement('output'); text.textContent = `${(100 * v).toFixed(2)}%`; row.append(label, meter, text); return row;
    }));
    const brier = 4 * mse(p, oneHot(ui.val('target'), 4));
    ui.status(`Sum = ${fmt(p.reduce((a, b) => a + b, 0))}. Entropy = ${fmt(entropy(p))} nats. Cross-entropy = ${fmt(crossEntropy(p, ui.val('target')))} nats. MSE (mean over 4 classes) = ${fmt(brier / 4)}. Multiclass Brier (sum over classes) = ${fmt(brier)}. One example cannot establish calibration.`);
  }
  el.addEventListener('input', draw); el.addEventListener('change', draw); draw();
}
function attention(el) {
  const tokens = ['The', 'animal', "didn't", 'cross', 'the', 'street'];
  // Deliberately tiny, hand-authored vectors; not learned language-model attention.
  const Q = [[1, 0], [1, 2], [-1, 1], [0, 2], [1, 0], [2, 1]];
  const K = [[1, 0], [2, 1], [-1, 1], [0, 1], [1, 0], [1, 2]];
  const V = [[0, 1], [2, 0], [-1, 0], [0, 2], [0, 1], [1, 3]];
  const ui = setup(el, `<h3>Who attends to whom?</h3><p>Hand-authored 2D Q, K and V vectors, not a trained model. Each row sums to 1. Select a query token; hover, focus or click a cell for its exact dot product and probability. Column numbers distinguish “The” from “the”.</p><div class="controls">${select('token', 'Query token', tokens.map((t, i) => [i, `${i + 1}: ${t}`]))}${select('mask', 'Attention mask', [['full', 'Bidirectional'], ['causal', 'Causal (past and self only)']])}</div><pre data-vectors></pre><div class="table-scroll"><table class="attention-table" data-heatmap></table></div>${readout}`);
  function draw() {
    const a = scaledDotProductAttention(Q, K, V, { causal: ui.get('mask').value === 'causal' });
    const selected = ui.val('token');
    el.querySelector('[data-vectors]').textContent = tokens.map((t, i) => `${i + 1} ${t}: Q=[${Q[i]}] K=[${K[i]}] V=[${V[i]}]`).join('\n') + `\nSelected output = Σⱼ attention[${selected + 1},j] Vⱼ = [${a.output[selected].map(fmt)}]`;
    const table = el.querySelector('[data-heatmap]'); table.replaceChildren();
    const head = table.createTHead().insertRow();
    ['Q \\ K', ...tokens.map((t, i) => `${i + 1} ${t}`)].forEach(t => { const th = document.createElement('th'); th.scope = 'col'; th.textContent = t; head.append(th); });
    const body = table.createTBody();
    const describe = (i, j) => ui.status(`${tokens[i]} (${i + 1}) → ${tokens[j]} (${j + 1}): Q·K = ${dot(Q[i], K[j])}; /√2 = ${fmt(dot(Q[i], K[j]) / Math.sqrt(2))}; ${a.scores[i][j] === -Infinity ? 'masked to −∞; ' : ''}softmax weight = ${fmt(a.weights[i][j])}. V = [${V[j]}]; weighted contribution = [${V[j].map(v => fmt(v * a.weights[i][j]))}].`);
    tokens.forEach((t, i) => {
      const tr = body.insertRow(); tr.classList.toggle('selected-query', i === selected);
      const th = document.createElement('th'); th.scope = 'row'; th.textContent = `${i + 1} ${t}`; tr.append(th);
      tokens.forEach((_, j) => {
        const td = tr.insertCell(), b = document.createElement('button'); b.type = 'button'; b.className = `heat-${Math.min(4, Math.floor(a.weights[i][j] * 5))}`;
        b.textContent = a.weights[i][j].toFixed(3); b.setAttribute('aria-label', `Query ${i + 1} ${t}, key ${j + 1} ${tokens[j]}: ${b.textContent}`);
        ['pointerenter', 'focus', 'click'].forEach(event => b.addEventListener(event, () => describe(i, j))); td.append(b);
      });
    }); describe(selected, selected);
  }
  el.addEventListener('change', draw); draw();
}
function activations(el) {
  const ui = setup(el, `<h3>Functions, derivatives and ten layers</h3><p>Blue solid: f(x). Yellow dashed: f′(x). The step derivative is zero away from zero and undefined at zero; ReLU’s derivative at zero uses the implementation convention 0.</p><div class="controls">${select('activation', 'Activation', Object.entries(ACTIVATIONS).map(([k, a]) => [k, a.name]))}${slider('input', 'Input', -3, 3, 0.5)}${slider('weight', 'Weight in all 10 layers', -2, 2, 1)}</div>${canvas('Activation and derivative over x from −4 to 4')}<div class="table-scroll"><table><thead><tr><th>Layer</th><th>Preactivation z</th><th>Activation</th><th>Gradient reaching input</th></tr></thead><tbody data-depth></tbody></table></div>${readout}`);
  ui.get('activation').value = 'sigmoid';
  function draw() {
    const a = ACTIVATIONS[ui.get('activation').value], { ctx, X, Y } = plot(el.querySelector('canvas'), [-4, 4, -2, 4]);
    for (const [fn, color, dash] of [[a.f, '#38bdf8', []], [a.d, '#fbbf24', [5, 5]]]) {
      ctx.setLineDash(dash); line(ctx, Array.from({ length: 401 }, (_, i) => { const x = i / 50 - 4; return [X(x), Y(fn(x))]; }), color);
    }
    ctx.setLineDash([]);
    const rows = gradientThroughDepth(ui.get('activation').value, { layers: 10, weight: ui.val('weight'), input: ui.val('input') });
    el.querySelector('[data-depth]').replaceChildren(...rows.map(r => { const tr = document.createElement('tr'); [r.layer, r.z, r.h, r.grad].forEach(v => { const td = document.createElement('td'); td.textContent = fmt(v); tr.append(td); }); return tr; }));
    ui.status(`f(${ui.val('input')}) = ${fmt(a.f(ui.val('input')))}; f′ = ${ui.get('activation').value === 'step' && ui.val('input') === 0 ? 'undefined' : fmt(a.d(ui.val('input')))}. Final-output gradient reaching the first input: ${fmt(rows[0].grad)}. Each layer multiplies by w f′(z).`);
  }
  el.addEventListener('input', draw); el.addEventListener('change', draw); draw();
}
function modern(el) {
  const ui = setup(el, `<h3>Inspect the modern stack</h3><p>Small numerical examples, not a model benchmark. Change a position, vector scale, expert score or context length.</p><div class="controls">${slider('scale', 'Vector scale', 0.1, 5, 1)}${slider('position', 'RoPE position', 0, 30, 0, 1)}${slider('expert', 'Expert 0 router logit', -4, 4, 1)}${slider('tokens', 'Cached tokens', 128, 8192, 1024, 128)}${select('heads', 'KV heads (8 query heads)', [[8, '8: MHA'], [2, '2: GQA'], [1, '1: MQA']])}${slider('sram', 'SRAM tile capacity (elements)', 4096, 65536, 16384, 4096)}</div><pre data-modern></pre>${readout}`);
  function draw() {
    const v = [1, 2, -1, 0.5].map(x => x * ui.val('scale'));
    const traffic = attentionHbmTraffic({ seqLen: ui.val('tokens'), headDim: 64, sramElements: ui.val('sram') });
    el.querySelector('[data-modern]').textContent = `Input: [${v.map(fmt)}]\nRMSNorm: [${rmsNorm(v).map(fmt)}]\nRoPE: [${rope(v, ui.val('position')).map(fmt)}]\nMoE top 2 of logits [${ui.val('expert')}, 2, 0, -1]: ${moeRoute([ui.val('expert'), 2, 0, -1]).map(r => `expert ${r.expert}: ${fmt(r.weight)}`).join('; ')}\nHBM traffic model: standard ${fmt(traffic.standard)} vs tiled ${fmt(traffic.flash)} elements (asymptotic, NOT measured bytes or speed).`;
    ui.status(`KV-cache = ${fmt(kvCacheBytes({ layers: 12, kvHeads: ui.val('heads'), headDim: 64, seqLen: ui.val('tokens') }) / 1048576)} MiB: 2 × 12 layers × ${ui.val('heads')} KV heads × 64 dimensions × ${ui.val('tokens')} tokens × 2 bytes. Batch size 1, FP16. GQA shares K/V heads, not query heads.`);
  }
  el.addEventListener('input', draw); el.addEventListener('change', draw); draw();
}
function quantization(el) {
  const ui = setup(el, `<h3>Float → int8 → float</h3><p>A real affine int8 quantizer in JavaScript, not a LiteRT model invocation. The largest magnitude affects precision for every value in this tensor.</p><div class="controls">${slider('outlier', 'Largest value', 1, 100, 4, 1)}</div><pre data-quant></pre>${readout}`);
  function draw() {
    const values = [-1, -0.1, 0, 0.1, 1, ui.val('outlier')], q = quantizeAffineInt8(values);
    el.querySelector('[data-quant]').textContent = `Real:    ${values.map(fmt).join(', ')}\nInt8:    ${q.q.join(', ')}\nDecoded: ${q.dequant.map(fmt).join(', ')}`;
    ui.status(`Scale ${fmt(q.scale)}, zero point ${q.zeroPoint}, maximum absolute error ${fmt(q.maxError)}. Decoded = (integer − zero point) × scale.`);
  }
  el.addEventListener('input', draw); draw();
}
function backend(el) {
  const kind = el.dataset.backend ?? 'javascript';
  const ui = setup(el, `<h3>Run a real matrix multiply</h3><p>A = [1, 2; 3, 4], B = [5, 6; 7, 8]. Expected C = [19, 22; 43, 50]. This executes ${kind === 'webgpu' ? 'a WebGPU compute shader, if this browser supports it' : kind === 'webassembly' ? 'the existing scalar WebAssembly f32 kernel (not SIMD)' : 'JavaScript Float32Array loops'}; it does not load a trained model.</p><div class="controls">${slider('scale', 'Scale A', 0.5, 3, 1, 0.5)}</div><div class="actions">${button('run', 'Multiply matrices')}</div>${readout}<details><summary>Implementation used by this demo</summary><pre data-source></pre></details>`);
  let engine;
  import('./engine.js').then(m => { engine = m; el.querySelector('[data-source]').textContent = (kind === 'webgpu' ? m.WGSL_GEMM_SHADER : kind === 'webassembly' ? m.buildWasmGemmBytes.toString() : m.jsMatmul.toString()); });
  ui.action('run', async () => {
    const run = el.querySelector('[data-action="run"]'); run.disabled = true;
    try {
      engine ??= await import('./engine.js');
      const A = new Float32Array([1, 2, 3, 4].map(v => v * ui.val('scale'))), B = new Float32Array([5, 6, 7, 8]);
      let result;
      if (kind === 'webgpu') {
        if (!(await engine.initWebGPUBackend()).available) throw new Error('WebGPU is unavailable on this device. No JavaScript fallback is presented as GPU execution.');
        result = await engine.webgpuMatmulAsync(A, B, 2, 2, 2);
      } else if (kind === 'webassembly') {
        const wasm = engine.getWasmBackend();
        if (!wasm.available) throw new Error('WebAssembly is unavailable or blocked.');
        result = wasm.matmul(A, B, 2, 2, 2);
      } else result = engine.jsMatmul(A, B, 2, 2, 2);
      ui.status(`${kind} result: [${Array.from(result, fmt).join(', ')}]. Expected [${[19, 22, 43, 50].map(v => fmt(v * ui.val('scale'))).join(', ')}].`);
    } catch (error) { ui.status(`Cannot run: ${error.message}`); }
    finally { run.disabled = false; }
  });
  el.addEventListener('input', () => ui.status('Input changed. Multiply again to compute the new output.'));
  ui.status('Ready. Press Multiply matrices.');
}
const widgets = { perceptron, landscape, backprop, convolution, probabilities, attention, activations, modern, quantization, backend };
for (const el of document.querySelectorAll('[data-widget]')) widgets[el.dataset.widget](el);
