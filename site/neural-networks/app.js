// @ts-check
import {
  getWasmBackend,
  setBackendMode,
  getBackendMode,
  benchmarkBackends,
  createYieldController,
  generate2DDataset,
  MLPNetwork,
  GLYPH_CLASSES,
  generateGlyphDataset,
  TinyCNN,
  VOCAB_TOKENS,
  generateSequenceBatch,
  MicroTransformer,
  SPRITE_TARGETS,
  MicroDDPM,
  MaskedTextDiffusion,
  DecisionPointerHead,
  BLOCK_REGISTRY,
  registerBlockType,
  analyzeArchitecturePipeline
} from "./engine.js";

const $ = (id) => document.getElementById(id);

function heatClass(val, min = 0, max = 1) {
  if (val < 0) return "heat-neg";
  const norm = Math.max(0, Math.min(1, (val - min) / Math.max(1e-6, max - min)));
  if (norm < 0.2) return "heat-0";
  if (norm < 0.45) return "heat-1";
  if (norm < 0.7) return "heat-2";
  if (norm < 0.9) return "heat-3";
  return "heat-4";
}

// ============================================================================
// 0. KERNEL BAR (WASM vs JS)
// ============================================================================

function initKernelBar() {
  const btnWasm = $("btn-kernel-wasm");
  const btnJs = $("btn-kernel-js");
  const btnBench = $("btn-bench-kernels");
  const readout = $("kernel-readout");
  const wb = getWasmBackend();

  const updateReadout = (msg) => {
    if (!readout) return;
    const mode = getBackendMode();
    readout.textContent = msg || (mode === "wasm"
      ? `Raw WebAssembly (wasm32 f32 GEMM, ${wb.byteLength} bytes compiled in-memory) active`
      : `Raw JavaScript (Float32Array triple-loop GEMM) active`);
  };

  if (btnWasm && btnJs) {
    btnWasm.addEventListener("click", () => {
      setBackendMode("wasm");
      btnWasm.setAttribute("aria-pressed", "true");
      btnJs.setAttribute("aria-pressed", "false");
      updateReadout();
    });
    btnJs.addEventListener("click", () => {
      setBackendMode("js");
      btnWasm.setAttribute("aria-pressed", "false");
      btnJs.setAttribute("aria-pressed", "true");
      updateReadout();
    });
  }

  if (btnBench) {
    btnBench.addEventListener("click", () => {
      const res = benchmarkBackends(64, 30);
      updateReadout(
        `64×64 GEMM ×30: WASM ${res.wasmMs.toFixed(2)}ms (${res.wasmGflops.toFixed(2)} GFLOP/s) vs JS ${res.jsMs.toFixed(2)}ms (${res.jsGflops.toFixed(2)} GFLOP/s) · max Δ=${res.maxDiff.toExponential(1)}`
      );
    });
  }

  updateReadout();
}

// ============================================================================
// 1. MLP & BACKPROPAGATION LAB
// ============================================================================

function initMlpLab() {
  const dsSelect = /** @type {HTMLSelectElement} */ ($("mlp-dataset"));
  const archSelect = /** @type {HTMLSelectElement} */ ($("mlp-arch"));
  const actSelect = /** @type {HTMLSelectElement} */ ($("mlp-act"));
  const optSelect = /** @type {HTMLSelectElement} */ ($("mlp-opt"));
  const lrInput = /** @type {HTMLInputElement} */ ($("mlp-lr"));
  const lrVal = $("mlp-lr-val");
  const canvas = /** @type {HTMLCanvasElement} */ ($("mlp-canvas"));
  if (!canvas || !dsSelect) return;

  let dataset = generate2DDataset(dsSelect.value, 120, 42);
  let mlp = createMlp();

  function createMlp() {
    const sizes = archSelect.value.split(",").map((n) => parseInt(n.trim(), 10));
    return new MLPNetwork(sizes, { activation: /** @type {any} */ (actSelect.value), seed: 101 });
  }

  function renderMlpState(stats = null) {
    if (stats) {
      $("mlp-epoch").textContent = String(stats.step);
      $("mlp-loss").textContent = stats.loss.toFixed(4);
      $("mlp-acc").textContent = `${(stats.accuracy * 100).toFixed(1)}%`;
    } else {
      const { probs } = mlp.forward(dataset.X, dataset.count);
      let loss = 0;
      let correct = 0;
      for (let i = 0; i < dataset.count; i++) {
        const p = Math.max(1e-6, Math.min(1 - 1e-6, probs[i]));
        const y = dataset.Y[i];
        loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
        if ((p >= 0.5 ? 1 : 0) === y) correct++;
      }
      $("mlp-epoch").textContent = String(mlp.stepCount);
      $("mlp-loss").textContent = (loss / dataset.count).toFixed(4);
      $("mlp-acc").textContent = `${((correct / dataset.count) * 100).toFixed(1)}%`;
    }

    // Render per-layer gradient norms
    const gradContainer = $("mlp-grad-bars");
    if (gradContainer) {
      gradContainer.replaceChildren();
      mlp.layers.forEach((layer, idx) => {
        const row = document.createElement("div");
        row.className = "grad-row";
        const label = document.createElement("span");
        label.textContent = `Layer ${idx + 1} (${layer.inDim}→${layer.outDim})`;
        const meter = document.createElement("meter");
        meter.min = 0;
        meter.max = 1.5;
        meter.value = Math.min(1.5, layer.gradNorm);
        const val = document.createElement("span");
        val.className = "number";
        val.textContent = layer.gradNorm.toFixed(4);
        row.append(label, meter, val);
        gradContainer.appendChild(row);
      });
    }

    drawDecisionBoundary();
  }

  function drawDecisionBoundary() {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    const gridRes = 30;
    const cellW = W / gridRes;
    const cellH = H / gridRes;

    const coords = new Float32Array(gridRes * gridRes * 2);
    let ptr = 0;
    for (let gy = 0; gy < gridRes; gy++) {
      const y = 1 - (gy / (gridRes - 1)) * 2;
      for (let gx = 0; gx < gridRes; gx++) {
        const x = (gx / (gridRes - 1)) * 2 - 1;
        coords[ptr++] = x;
        coords[ptr++] = y;
      }
    }
    const { probs } = mlp.forward(coords, gridRes * gridRes);

    for (let gy = 0; gy < gridRes; gy++) {
      for (let gx = 0; gx < gridRes; gx++) {
        const p = probs[gy * gridRes + gx];
        const r = Math.round(25 + (1 - p) * 195);
        const g = Math.round(70 + p * 115 + (1 - Math.abs(p - 0.5) * 2) * 50);
        const b = Math.round(55 + p * 190);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(gx * cellW, gy * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    // Draw dataset points
    for (let i = 0; i < dataset.count; i++) {
      const x = ((dataset.X[i * 2] + 1) * 0.5) * W;
      const y = ((1 - dataset.X[i * 2 + 1]) * 0.5) * H;
      const label = dataset.Y[i];
      ctx.beginPath();
      ctx.arc(x, y, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = label === 1 ? "#0284c7" : "#e11d48";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    }
  }

  lrInput.addEventListener("input", () => {
    lrVal.textContent = Number(lrInput.value).toFixed(3);
  });

  dsSelect.addEventListener("change", () => {
    dataset = generate2DDataset(dsSelect.value, 120, 42);
    mlp = createMlp();
    renderMlpState();
  });

  [archSelect, actSelect].forEach((sel) => {
    sel.addEventListener("change", () => {
      mlp = createMlp();
      renderMlpState();
    });
  });

  $("mlp-reset-btn").addEventListener("click", () => {
    mlp = createMlp();
    renderMlpState();
  });

  $("mlp-step-btn").addEventListener("click", () => {
    const stats = mlp.trainStep(dataset, {
      lr: Number(lrInput.value),
      optimizer: optSelect.value
    });
    renderMlpState(stats);
  });

  $("mlp-train-btn").addEventListener("click", async () => {
    const btn = /** @type {HTMLButtonElement} */ ($("mlp-train-btn"));
    btn.disabled = true;
    const yielder = createYieldController(50);
    let lastStats = null;
    for (let e = 0; e < 80; e++) {
      lastStats = mlp.trainStep(dataset, {
        lr: Number(lrInput.value),
        optimizer: optSelect.value
      });
      if (e % 10 === 9 || e === 79) {
        renderMlpState(lastStats);
      }
      await yielder.maybeYield();
    }
    btn.disabled = false;
  });

  canvas.addEventListener("click", (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x1 = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const x2 = 1 - ((ev.clientY - rect.top) / rect.height) * 2;
    const p = mlp.predictPoint(x1, x2);
    $("mlp-probe-readout").textContent =
      `Probe (${x1.toFixed(2)}, ${x2.toFixed(2)}) → P(Class 1) = ${(p * 100).toFixed(1)}% (${p >= 0.5 ? "Blue Class 1" : "Red Class 0"})`;
  });

  // Pre-train 25 steps so user sees an active boundary on load
  for (let i = 0; i < 25; i++) {
    mlp.trainStep(dataset, { lr: 0.06, optimizer: "adamw" });
  }
  renderMlpState();
}

// ============================================================================
// 2. CNN LAB
// ============================================================================

function initCnnLab() {
  const gridEl = $("cnn-pixel-grid");
  if (!gridEl) return;

  const samples = generateGlyphDataset(8, 77);
  let cnn = new TinyCNN(2026);
  const currentImg = new Float32Array(GLYPH_CLASSES[0].pattern);

  function buildPixelGrid() {
    gridEl.replaceChildren();
    for (let i = 0; i < 64; i++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `pixel-btn ${currentImg[i] > 0.5 ? "lit" : ""}`;
      btn.setAttribute("aria-label", `Pixel row ${Math.floor(i / 8) + 1} col ${(i % 8) + 1}`);
      btn.setAttribute("aria-pressed", String(currentImg[i] > 0.5));
      btn.addEventListener("click", () => {
        currentImg[i] = currentImg[i] > 0.5 ? 0 : 1;
        btn.classList.toggle("lit", currentImg[i] > 0.5);
        btn.setAttribute("aria-pressed", String(currentImg[i] > 0.5));
        renderCnnForward();
      });
      gridEl.appendChild(btn);
    }
  }

  function renderCnnForward(trainStats = null) {
    if (trainStats) {
      $("cnn-epoch").textContent = String(trainStats.step);
      $("cnn-loss").textContent = trainStats.loss.toFixed(4);
      $("cnn-acc").textContent = `${(trainStats.accuracy * 100).toFixed(1)}%`;
    } else {
      $("cnn-epoch").textContent = String(cnn.stepCount);
    }

    const { convA, probs } = cnn.forward(currentImg);

    // Render class probabilities
    const probsEl = $("cnn-probs");
    if (probsEl) {
      probsEl.replaceChildren();
      GLYPH_CLASSES.forEach((cls, c) => {
        const row = document.createElement("div");
        row.className = "bar";
        const name = document.createElement("span");
        name.textContent = cls.name;
        const meter = document.createElement("meter");
        meter.min = 0;
        meter.max = 1;
        meter.value = probs[c];
        const pct = document.createElement("span");
        pct.className = "number";
        pct.textContent = `${(probs[c] * 100).toFixed(1)}%`;
        row.append(name, meter, pct);
        probsEl.appendChild(row);
      });
    }

    // Render 4 filters (3x3) and their 6x6 feature maps
    const fmapContainer = $("cnn-feature-maps");
    if (fmapContainer) {
      fmapContainer.replaceChildren();
      for (let f = 0; f < 4; f++) {
        const card = document.createElement("div");
        card.className = "fmap-card";
        const title = document.createElement("strong");
        title.textContent = `Filter #${f + 1} (3×3 → 6×6)`;

        const kGrid = document.createElement("div");
        kGrid.className = "mini-grid-3";
        for (let k = 0; k < 9; k++) {
          const cell = document.createElement("div");
          const w = cnn.convW[f * 9 + k];
          cell.className = `mini-cell ${heatClass(w, -0.5, 0.8)}`;
          kGrid.appendChild(cell);
        }

        const fGrid = document.createElement("div");
        fGrid.className = "mini-grid-6";
        for (let m = 0; m < 36; m++) {
          const cell = document.createElement("div");
          const a = convA[f * 36 + m];
          cell.className = `mini-cell ${heatClass(a, 0, 1.8)}`;
          fGrid.appendChild(cell);
        }

        card.append(title, kGrid, fGrid);
        fmapContainer.appendChild(card);
      }
    }
  }

  document.querySelectorAll("[data-glyph-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      // @ts-ignore
      const idx = parseInt(btn.dataset.glyphPreset, 10);
      currentImg.set(GLYPH_CLASSES[idx].pattern);
      buildPixelGrid();
      renderCnnForward();
    });
  });

  $("cnn-train-btn").addEventListener("click", async () => {
    const btn = /** @type {HTMLButtonElement} */ ($("cnn-train-btn"));
    btn.disabled = true;
    const yielder = createYieldController(50);
    let stats = null;
    for (let e = 0; e < 30; e++) {
      stats = cnn.trainEpoch(samples, 0.08);
      if (e % 5 === 4 || e === 29) renderCnnForward(stats);
      await yielder.maybeYield();
    }
    btn.disabled = false;
  });

  $("cnn-reset-btn").addEventListener("click", () => {
    cnn = new TinyCNN(Math.floor(Math.random() * 10000));
    $("cnn-loss").textContent = "—";
    $("cnn-acc").textContent = "—";
    renderCnnForward();
  });

  // Pre-train 18 epochs so initial feature maps are meaningful
  let initStats = null;
  for (let e = 0; e < 18; e++) initStats = cnn.trainEpoch(samples, 0.08);
  buildPixelGrid();
  renderCnnForward(initStats);
}

// ============================================================================
// 3. TRANSFORMER & SELF-ATTENTION LAB
// ============================================================================

function initTransformerLab() {
  const taskSelect = /** @type {HTMLSelectElement} */ ($("tf-task"));
  const maskSelect = /** @type {HTMLSelectElement} */ ($("tf-mask"));
  const seqInput = /** @type {HTMLInputElement} */ ($("tf-input-seq"));
  if (!taskSelect || !seqInput) return;

  let batch = generateSequenceBatch(taskSelect.value, 4, 28, 99);
  let tf = createTf();

  function createTf() {
    return new MicroTransformer({
      vocabSize: 8,
      seqLen: 4,
      dModel: 16,
      numHeads: 2,
      causal: maskSelect.value === "causal",
      seed: 314
    });
  }

  function parseProbeTokens() {
    const raw = seqInput.value.toUpperCase().trim().split(/[\s,]+/).filter(Boolean);
    const ids = [];
    for (let i = 0; i < 4; i++) {
      const idx = VOCAB_TOKENS.indexOf(raw[i] || VOCAB_TOKENS[i]);
      ids.push(idx >= 0 ? idx : i);
    }
    return ids;
  }

  function renderTransformer(stats = null) {
    if (stats) {
      $("tf-epoch").textContent = String(stats.step);
      $("tf-loss").textContent = stats.loss.toFixed(4);
      $("tf-acc").textContent = `${(stats.accuracy * 100).toFixed(1)}%`;
    } else {
      $("tf-epoch").textContent = String(tf.stepCount);
    }

    const tokenIds = parseProbeTokens();
    const { attnWeights, predictions } = tf.forward(tokenIds);
    const inTokens = tokenIds.map((i) => VOCAB_TOKENS[i]);
    const outTokens = predictions.map((i) => VOCAB_TOKENS[i]);

    $("tf-pred-banner").textContent =
      `Input: [${inTokens.join(", ")}] → Transformer Output: [${outTokens.join(", ")}]`;

    const headsContainer = $("tf-attn-heads");
    if (headsContainer) {
      headsContainer.replaceChildren();
      attnWeights.forEach((weights, hIdx) => {
        const box = document.createElement("div");
        const title = document.createElement("h4");
        title.textContent = `Head #${hIdx + 1} Attention Weights`;
        const table = document.createElement("table");
        table.className = "attn-table";
        const thead = document.createElement("thead");
        const hr = document.createElement("tr");
        const corner = document.createElement("th");
        corner.textContent = "Q \\ K";
        hr.appendChild(corner);
        inTokens.forEach((tok, pos) => {
          const th = document.createElement("th");
          th.textContent = `${tok}${pos}`;
          hr.appendChild(th);
        });
        thead.appendChild(hr);

        const tbody = document.createElement("tbody");
        for (let r = 0; r < 4; r++) {
          const tr = document.createElement("tr");
          const rHead = document.createElement("th");
          rHead.textContent = `${inTokens[r]}${r}`;
          tr.appendChild(rHead);
          for (let c = 0; c < 4; c++) {
            const td = document.createElement("td");
            const w = weights[r * 4 + c];
            td.className = heatClass(w, 0, 0.85);
            td.textContent = w.toFixed(2);
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        }
        table.append(thead, tbody);
        box.append(title, table);
        headsContainer.appendChild(box);
      });
    }
  }

  [taskSelect, maskSelect].forEach((sel) => {
    sel.addEventListener("change", () => {
      batch = generateSequenceBatch(taskSelect.value, 4, 28, 99);
      tf = createTf();
      let st = null;
      for (let e = 0; e < 25; e++) st = tf.trainEpoch(batch, 0.12);
      renderTransformer(st);
    });
  });

  seqInput.addEventListener("input", () => renderTransformer());

  $("tf-train-btn").addEventListener("click", async () => {
    const btn = /** @type {HTMLButtonElement} */ ($("tf-train-btn"));
    btn.disabled = true;
    const yielder = createYieldController(50);
    let st = null;
    for (let e = 0; e < 40; e++) {
      st = tf.trainEpoch(batch, 0.12);
      if (e % 8 === 7 || e === 39) renderTransformer(st);
      await yielder.maybeYield();
    }
    btn.disabled = false;
  });

  $("tf-reset-btn").addEventListener("click", () => {
    tf = createTf();
    $("tf-loss").textContent = "—";
    $("tf-acc").textContent = "—";
    renderTransformer();
  });

  let initSt = null;
  for (let e = 0; e < 28; e++) initSt = tf.trainEpoch(batch, 0.12);
  renderTransformer(initSt);
}

// ============================================================================
// 4. DIFFUSION LAB (IMAGE DDPM + MASKED TEXT DIFFUSION)
// ============================================================================

function initDiffusionLab() {
  const targetSelect = /** @type {HTMLSelectElement} */ ($("ddpm-target"));
  const stepSlider = /** @type {HTMLInputElement} */ ($("ddpm-step-slider"));
  const stepVal = $("ddpm-step-val");
  if (!targetSelect) return;

  let ddpm = new MicroDDPM({ dim: 36, steps: 16, hiddenDim: 48, seed: 512 });
  let trajectory = [];

  function renderDdpm(stats = null, seed = 808) {
    if (stats) {
      $("ddpm-epoch").textContent = String(stats.step);
      $("ddpm-loss").textContent = stats.loss.toFixed(4);
    }
    trajectory = ddpm.sampleTrajectory(seed);
    renderDdpmSnapshots();
  }

  function renderDdpmSnapshots() {
    const strip = $("ddpm-strip");
    if (!strip || trajectory.length === 0) return;
    strip.replaceChildren();

    const selectedStep = parseInt(stepSlider.value, 10);
    stepVal.textContent = `${selectedStep} (${selectedStep === 16 ? "Pure Noise x_T" : selectedStep === 0 ? "Denoised x_0" : "Intermediate"})`;

    // Show key snapshots: t=16, t=12, t=8, t=4, t=selectedStep (or t=0)
    const milestones = [16, 10, 5, 0];
    if (!milestones.includes(selectedStep)) milestones.splice(2, 0, selectedStep);

    for (const stepNum of milestones) {
      const frame = trajectory.find((f) => f.step === stepNum) || trajectory[trajectory.length - 1];
      const card = document.createElement("div");
      card.className = "sprite-card";
      const label = document.createElement("div");
      label.className = "small";
      label.textContent = `t = ${frame.step}`;
      const grid = document.createElement("div");
      grid.className = "mini-grid-6";
      for (let i = 0; i < 36; i++) {
        const cell = document.createElement("div");
        cell.className = `mini-cell ${heatClass(frame.pixels[i], 0.1, 0.9)}`;
        grid.appendChild(cell);
      }
      card.append(label, grid);
      strip.appendChild(card);
    }
  }

  stepSlider.addEventListener("input", renderDdpmSnapshots);

  targetSelect.addEventListener("change", () => {
    ddpm = new MicroDDPM({ dim: 36, steps: 16, hiddenDim: 48, seed: 512 });
    const target = SPRITE_TARGETS[targetSelect.value] || SPRITE_TARGETS.smiley;
    let st = null;
    for (let i = 0; i < 35; i++) st = ddpm.trainStep(target, 0.04, 16);
    renderDdpm(st);
  });

  $("ddpm-train-btn").addEventListener("click", async () => {
    const btn = /** @type {HTMLButtonElement} */ ($("ddpm-train-btn"));
    btn.disabled = true;
    const yielder = createYieldController(50);
    const target = SPRITE_TARGETS[targetSelect.value] || SPRITE_TARGETS.smiley;
    let st = null;
    for (let i = 0; i < 60; i++) {
      st = ddpm.trainStep(target, 0.04, 16);
      if (i % 15 === 14 || i === 59) renderDdpm(st);
      await yielder.maybeYield();
    }
    btn.disabled = false;
  });

  $("ddpm-sample-btn").addEventListener("click", () => {
    renderDdpm(null, Math.floor(Math.random() * 10000));
  });

  // Discrete Masked Text Diffusion
  const textDiff = new MaskedTextDiffusion(909);
  const textPromptSelect = /** @type {HTMLSelectElement} */ ($("text-diff-prompt"));

  function renderTextDiffusion(stats = null) {
    if (stats) {
      $("text-diff-epoch").textContent = String(stats.step);
      $("text-diff-loss").textContent = stats.loss.toFixed(4);
    }
    const idx = parseInt(textPromptSelect.value, 10) || 0;
    const steps = textDiff.denoiseTrajectory(idx);
    const container = $("text-diff-trajectory");
    if (!container) return;
    container.replaceChildren();

    for (const s of steps) {
      const row = document.createElement("div");
      row.className = "text-step-row";
      const badge = document.createElement("strong");
      badge.textContent = `Step ${s.step}:`;
      row.appendChild(badge);
      s.tokens.forEach((tok, pos) => {
        const chip = document.createElement("span");
        const isMask = tok === "[MASK]";
        chip.className = `tok-chip ${isMask ? "masked" : ""}`;
        chip.textContent = isMask ? "[MASK]" : `${tok} (${Math.round(s.confidences[pos] * 100)}%)`;
        row.appendChild(chip);
      });
      container.appendChild(row);
    }
  }

  textPromptSelect.addEventListener("change", () => renderTextDiffusion());

  $("text-diff-train-btn").addEventListener("click", async () => {
    const btn = /** @type {HTMLButtonElement} */ ($("text-diff-train-btn"));
    btn.disabled = true;
    const yielder = createYieldController(50);
    let st = null;
    for (let i = 0; i < 45; i++) {
      st = textDiff.trainStep(null, 0.2);
      if (i % 15 === 14 || i === 44) renderTextDiffusion(st);
      await yielder.maybeYield();
    }
    btn.disabled = false;
  });

  $("text-diff-sample-btn").addEventListener("click", () => renderTextDiffusion());

  // Initial warm-up for both Image DDPM and Masked Text Diffusion
  let initDdpmSt = null;
  for (let i = 0; i < 35; i++) initDdpmSt = ddpm.trainStep(SPRITE_TARGETS.smiley, 0.04, 16);
  renderDdpm(initDdpmSt);

  let initTextSt = null;
  for (let i = 0; i < 35; i++) initTextSt = textDiff.trainStep(null, 0.2);
  renderTextDiffusion(initTextSt);
}

// ============================================================================
// 5. DECISION POINTER HEAD LAB
// ============================================================================

function initDecisionLab() {
  const scenSelect = /** @type {HTMLSelectElement} */ ($("dec-scenario"));
  const tempInput = /** @type {HTMLInputElement} */ ($("dec-temp"));
  const gateInput = /** @type {HTMLInputElement} */ ($("dec-gate"));
  if (!scenSelect) return;

  const head = new DecisionPointerHead(8, 404);
  const optionLabels = [
    "‹0› Cordon & Swap Host (Hardware ECC)",
    "‹1› Coalesce Blob Shards (Storage I/O)",
    "‹2› Cut Learning Rate 33% (Grad Spike)",
    "‹3› Escalate to Human On-Call"
  ];
  const optionVecs = [
    new Float32Array([1.2, 0.9, -0.4, 0.6, 0.2, -0.3, 0.8, 0.5]),
    new Float32Array([-0.5, 1.1, 0.9, -0.3, 0.7, 0.4, -0.2, 0.3]),
    new Float32Array([0.1, -0.7, 1.2, 0.8, -0.5, 0.9, 0.1, -0.4]),
    new Float32Array([-0.3, -0.2, -0.4, 0.2, 0.9, 0.8, -0.6, 1.0])
  ];

  const stateScenarios = {
    clear: new Float32Array([1.3, 0.85, -0.4, 0.65, 0.2, -0.3, 0.85, 0.5]),
    ambiguous: new Float32Array([0.35, 0.95, 0.35, 0.15, 0.45, 0.1, 0.25, 0.35]),
    ood: new Float32Array([0.02, -0.04, 0.01, 0.03, -0.02, 0.01, -0.01, 0.02])
  };

  function update() {
    const T = Number(tempInput.value);
    const tau = Number(gateInput.value);
    $("dec-temp-val").textContent = T.toFixed(2);
    $("dec-gate-val").textContent = tau.toFixed(2);

    const stateVec = stateScenarios[scenSelect.value] || stateScenarios.clear;
    const res = head.decide(stateVec, optionVecs, { temperature: T, gateThreshold: tau });

    const out = $("dec-output");
    if (!out) return;
    out.replaceChildren();

    const summary = document.createElement("p");
    summary.className = "gate";
    summary.textContent = `Gate Recommendation: ${res.recommendation.toUpperCase()} · Winner: ${optionLabels[res.bestIdx]} · Choice Confidence: ${(res.confidence * 100).toFixed(1)}% · Act Prob: ${(res.actProb * 100).toFixed(1)}%`;
    out.appendChild(summary);

    optionLabels.forEach((label, idx) => {
      const row = document.createElement("div");
      row.className = "bar";
      const name = document.createElement("span");
      name.textContent = label;
      const meter = document.createElement("meter");
      meter.min = 0;
      meter.max = 1;
      meter.value = res.probs[idx];
      const val = document.createElement("span");
      val.className = "number";
      val.textContent = `${(res.probs[idx] * 100).toFixed(1)}%`;
      row.append(name, meter, val);
      out.appendChild(row);
    });
  }

  [scenSelect, tempInput, gateInput].forEach((el) => el.addEventListener("input", update));
  update();
}

// ============================================================================
// 6. EXTENSIBLE PLUG-AND-PLAY ARCHITECTURE BUILDER
// ============================================================================

function initBlockBuilder() {
  const paletteEl = $("builder-palette");
  const stackEl = $("builder-stack");
  if (!paletteEl || !stackEl) return;

  let pipeline = ["embedding", "layernorm", "attention", "layernorm", "mlp_block", "decision_head"];

  function renderPalette() {
    paletteEl.replaceChildren();
    for (const [type, spec] of BLOCK_REGISTRY.entries()) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "secondary";
      btn.textContent = `+ ${spec.label}`;
      btn.addEventListener("click", () => {
        pipeline.push(type);
        renderPipeline();
      });
      paletteEl.appendChild(btn);
    }
  }

  function renderPipeline() {
    const analysis = analyzeArchitecturePipeline(pipeline, { seqLen: 16, dModel: 32 });
    $("builder-depth").textContent = `${analysis.stages.length} blocks`;
    $("builder-params").textContent = analysis.totalParams.toLocaleString();
    $("builder-flops").textContent = analysis.totalFlops.toLocaleString();
    const firstGrad = analysis.stages[0]?.gradNorm ?? 0;
    $("builder-grad").textContent = firstGrad.toFixed(3);

    stackEl.replaceChildren();
    analysis.stages.forEach((stage, idx) => {
      const item = document.createElement("div");
      item.className = "block-item";

      const titleCol = document.createElement("div");
      const strong = document.createElement("strong");
      strong.textContent = `${idx + 1}. ${stage.label}`;
      const cat = document.createElement("div");
      cat.className = "small";
      cat.textContent = `Category: ${stage.category} · Shape: [${stage.outShape}]`;
      titleCol.append(strong, cat);

      const paramCol = document.createElement("div");
      paramCol.textContent = `${stage.params.toLocaleString()} params`;

      const flopCol = document.createElement("div");
      flopCol.textContent = `${stage.flops.toLocaleString()} FLOPs`;

      const actCol = document.createElement("div");
      actCol.textContent = `‖a‖=${stage.actNorm.toFixed(2)}`;

      const gradCol = document.createElement("div");
      gradCol.textContent = `‖∇‖=${stage.gradNorm.toFixed(2)}`;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "secondary";
      removeBtn.textContent = "Remove";
      removeBtn.disabled = pipeline.length <= 1;
      removeBtn.addEventListener("click", () => {
        pipeline.splice(idx, 1);
        renderPipeline();
      });

      item.append(titleCol, paramCol, flopCol, actCol, gradCol, removeBtn);
      stackEl.appendChild(item);
    });
  }

  $("btn-register-ssm").addEventListener("click", () => {
    registerBlockType({
      type: "ssm_mamba",
      label: "Mamba Selective State-Space (SSM)",
      category: "state-space",
      description: "Linear-time O(T·D) selective state scan with input-dependent Δ, B, C.",
      computeStats(shape) {
        return {
          outShape: { ...shape },
          params: 3 * shape.d * shape.d + 2 * shape.d,
          flops: 6 * shape.t * shape.d * shape.d
        };
      }
    });
    pipeline.splice(Math.max(1, pipeline.length - 1), 0, "ssm_mamba");
    renderPalette();
    renderPipeline();
  });

  $("btn-register-moe").addEventListener("click", () => {
    registerBlockType({
      type: "sparse_moe",
      label: "Sparse Mixture-of-Experts (Top-2 of 8)",
      category: "conditional",
      description: "Routes each token to 2 of 8 expert MLPs, scaling capacity at 25% active FLOPs.",
      computeStats(shape) {
        const D = shape.d;
        const T = shape.t;
        return {
          outShape: { ...shape },
          params: 8 * (4 * D * D) + 8 * D,
          flops: 2 * (8 * T * D * D)
        };
      }
    });
    pipeline.splice(Math.max(1, pipeline.length - 1), 0, "sparse_moe");
    renderPalette();
    renderPipeline();
  });

  $("btn-reset-pipeline").addEventListener("click", () => {
    pipeline = ["embedding", "layernorm", "attention", "layernorm", "mlp_block", "decision_head"];
    renderPipeline();
  });

  renderPalette();
  renderPipeline();
}

export function initNeuralNetworksApp() {
  if (typeof document === "undefined") return;
  initKernelBar();
  initMlpLab();
  initCnnLab();
  initTransformerLab();
  initDiffusionLab();
  initDecisionLab();
  initBlockBuilder();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initNeuralNetworksApp);
  } else {
    initNeuralNetworksApp();
  }
}
