// @ts-check
import {
  getWasmBackend,
  initWebGPUBackend,
  getWebGPUStatus,
  webgpuMatmulAsync,
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
  generateTemporalSequenceBatch,
  MicroRNN,
  DeepResidualNetwork,
  BLOCK_REGISTRY,
  registerBlockType,
  analyzeArchitecturePipeline
} from "./engine.js";
import { HYPERPARAMETER_GUIDE, KERNEL_CODE_BLUEPRINTS } from "./curriculum.js";
import {
  createCnnStepper,
  createRnnStepper,
  createResNetStepper,
  createTransformerStepper,
  createDiffusionStepper,
  createDecisionStepper,
  createBuilderStepper
} from "./steppers.js";

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
// 0. KERNEL BAR (WASM vs JS vs WebGPU) & SIDE-BY-SIDE CODE EXPLORERS
// ============================================================================

function initKernelBar() {
  if (!$("btn-kernel-wasm") && !$("kernel-readout")) return;
  const btnWasm = $("btn-kernel-wasm");
  const btnJs = $("btn-kernel-js");
  const btnWebgpu = $("btn-kernel-webgpu");
  const btnBench = $("btn-bench-kernels");
  const readout = $("kernel-readout");
  const wb = getWasmBackend();

  const setPressed = (activeMode) => {
    btnWasm?.setAttribute("aria-pressed", String(activeMode === "wasm"));
    btnJs?.setAttribute("aria-pressed", String(activeMode === "js"));
    btnWebgpu?.setAttribute("aria-pressed", String(activeMode === "webgpu"));
  };

  const updateReadout = (msg) => {
    if (!readout) return;
    if (msg) {
      readout.textContent = msg;
      return;
    }
    const mode = getBackendMode();
    if (mode === "wasm") {
      readout.textContent = `Raw WebAssembly (wasm32 f32 GEMM, ${wb.byteLength} bytes compiled in-memory) active`;
    } else if (mode === "webgpu") {
      const st = getWebGPUStatus();
      readout.textContent = `WebGPU Compute Mode (${st.adapterInfo}) · Sync inner loops backed by wasm32 GEMM`;
    } else {
      readout.textContent = `Raw JavaScript (Float32Array triple-loop GEMM) active`;
    }
  };

  btnWasm?.addEventListener("click", () => {
    setBackendMode("wasm");
    setPressed("wasm");
    updateReadout();
  });

  btnJs?.addEventListener("click", () => {
    setBackendMode("js");
    setPressed("js");
    updateReadout();
  });

  btnWebgpu?.addEventListener("click", async () => {
    setBackendMode("webgpu");
    setPressed("webgpu");
    const st = await initWebGPUBackend();
    updateReadout(`WebGPU Compute Mode · ${st.adapterInfo}`);
    // Also switch global code viewer to WebGPU tab so user sees the WGSL shader immediately
    const wgslTab = /** @type {HTMLButtonElement | null} */ (
      document.querySelector('[data-global-lang="webgpu"]')
    );
    wgslTab?.click();
  });

  btnBench?.addEventListener("click", async () => {
    const res = benchmarkBackends(64, 30);
    const A = new Float32Array(64 * 64).fill(0.5);
    const B = new Float32Array(64 * 64).fill(0.25);
    const t0 = performance.now();
    await webgpuMatmulAsync(A, B, 64, 64, 64);
    const gpuMs = performance.now() - t0;
    const st = getWebGPUStatus();
    const gpuLabel = st.available ? `WebGPU 1-dispatch ${gpuMs.toFixed(2)}ms` : `WebGPU fallback (${gpuMs.toFixed(2)}ms)`;
    updateReadout(
      `64×64 GEMM ×30: WASM ${res.wasmMs.toFixed(2)}ms (${res.wasmGflops.toFixed(2)} GFLOP/s) vs JS ${res.jsMs.toFixed(2)}ms (${res.jsGflops.toFixed(2)} GFLOP/s) · ${gpuLabel} · max Δ=${res.maxDiff.toExponential(1)}`
    );
  });

  updateReadout();
}

function initCodeExplorers() {
  if (!$("global-code-display") && !document.querySelector(".section-code-widget")) return;
  const opSelect = /** @type {HTMLSelectElement | null} */ ($("global-code-op"));
  const summaryEl = $("global-code-summary");
  const codeDisplay = $("global-code-display");
  const langBtns = Array.from(document.querySelectorAll("[data-global-lang]"));
  let currentLang = "js";

  function renderGlobalCode() {
    if (!opSelect || !codeDisplay) return;
    const bp = KERNEL_CODE_BLUEPRINTS[opSelect.value] || KERNEL_CODE_BLUEPRINTS.gemm;
    if (summaryEl) summaryEl.textContent = bp.summary;
    codeDisplay.textContent = bp[currentLang] || bp.js;
    langBtns.forEach((btn) => {
      const lang = btn.getAttribute("data-global-lang");
      const active = lang === currentLang;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    });
  }

  opSelect?.addEventListener("change", renderGlobalCode);
  langBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      currentLang = btn.getAttribute("data-global-lang") || "js";
      renderGlobalCode();
    });
  });
  renderGlobalCode();

  // Populate per-chapter inline code widgets (.section-code-widget[data-code-blueprint])
  document.querySelectorAll(".section-code-widget[data-code-blueprint]").forEach((container) => {
    const key = container.getAttribute("data-code-blueprint") || "gemm";
    const bp = KERNEL_CODE_BLUEPRINTS[key];
    if (!bp) return;

    const details = document.createElement("details");
    details.className = "chapter-code-details";
    const summary = document.createElement("summary");
    summary.textContent = `View First-Principles Code (Raw JS · Raw WASM WAT · WebGPU WGSL) — ${bp.title}`;
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "chapter-code-body";

    const desc = document.createElement("p");
    desc.className = "small";
    desc.textContent = bp.summary;
    body.appendChild(desc);

    const tabBar = document.createElement("div");
    tabBar.className = "code-lang-tabs";
    const pre = document.createElement("pre");
    pre.className = "kernel-code-pre";
    const codeEl = document.createElement("code");
    codeEl.textContent = bp.js;
    pre.appendChild(codeEl);

    const langs = [
      { id: "js", label: "Raw JS (Float32Array)" },
      { id: "wasm", label: "Raw WASM (WAT & Opcodes)" },
      { id: "webgpu", label: "WebGPU (WGSL Compute Shader)" }
    ];

    const tabButtons = langs.map((l, idx) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = `code-lang-btn ${idx === 0 ? "active" : ""}`;
      b.textContent = l.label;
      b.addEventListener("click", () => {
        codeEl.textContent = bp[l.id] || bp.js;
        tabButtons.forEach((other) => other.classList.remove("active"));
        b.classList.add("active");
      });
      tabBar.appendChild(b);
      return b;
    });

    body.append(tabBar, pre);
    details.appendChild(body);
    container.replaceChildren(details);
  });
}

// ============================================================================
// 1. MLP & BACKPROPAGATION LAB
// ============================================================================

function initMlpLab() {
  if (!$("mlp-train-btn") && !$("mlp-canvas")) return;
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
    computeAndRenderStepper();
    renderHyperparameterExplainer();
  }

  function renderHyperparameterExplainer() {
    const container = $("mlp-hyperparam-explainer");
    if (!container) return;
    container.replaceChildren();

    const dsInfo = HYPERPARAMETER_GUIDE.datasets[dsSelect.value] || HYPERPARAMETER_GUIDE.datasets.xor;
    const archInfo = HYPERPARAMETER_GUIDE.architectures[archSelect.value] || HYPERPARAMETER_GUIDE.architectures["2,4,4,1"];
    const actInfo = HYPERPARAMETER_GUIDE.activations[actSelect.value] || HYPERPARAMETER_GUIDE.activations.gelu;
    const optInfo = HYPERPARAMETER_GUIDE.optimizers[optSelect.value] || HYPERPARAMETER_GUIDE.optimizers.adamw;
    const lrInfo = HYPERPARAMETER_GUIDE.learningRate(Number(lrInput.value));

    const header = document.createElement("h4");
    header.textContent = "Active Hyperparameter & Component Deep-Dive";
    container.appendChild(header);

    const items = [
      { badge: "Manifold Dataset", title: dsInfo.title, eq: dsInfo.equation, body: `${dsInfo.topology} ${dsInfo.theory}` },
      { badge: "Hidden Architecture", title: archInfo.title, eq: archInfo.params, body: archInfo.theory },
      { badge: "Activation σ(z)", title: actInfo.title, eq: actInfo.equation, body: actInfo.theory },
      { badge: "Optimizer", title: optInfo.title, eq: optInfo.equation, body: optInfo.theory },
      { badge: "Learning Rate η", title: lrInfo.title, eq: lrInfo.equation, body: lrInfo.theory }
    ];

    for (const item of items) {
      const card = document.createElement("div");
      card.className = "hyperparam-card";

      const top = document.createElement("div");
      top.className = "hyperparam-card-top";
      const badge = document.createElement("span");
      badge.className = "hyperparam-badge";
      badge.textContent = item.badge;
      const strong = document.createElement("strong");
      strong.textContent = item.title;
      top.append(badge, strong);

      const eq = document.createElement("code");
      eq.className = "hyperparam-eq";
      eq.textContent = item.eq;

      const p = document.createElement("p");
      p.className = "small";
      p.textContent = item.body;

      card.append(top, eq, p);
      container.appendChild(card);
    }
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

  // ==========================================================================
  // Interactive Network Node Graph & Step-by-Step Debugger Controller
  // ==========================================================================
  const stepperModeEl = /** @type {HTMLSelectElement | null} */ ($("stepper-mode"));
  const stepperSampleEl = /** @type {HTMLSelectElement | null} */ ($("stepper-sample"));
  const stepperSvg = /** @type {SVGSVGElement | null} */ (/** @type {any} */ ($("nn-stepper-svg")));
  const SVG_NS = "http://www.w3.org/2000/svg";

  /** @type {any} */
  let currentTrace = null;
  let currentMicroStepIdx = 0;
  let selectedNodeId = "L1N0";
  /** @type {any} */
  let autoPlayTimer = null;

  function stopAutoPlay() {
    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
    }
    const playBtn = $("stepper-play-btn");
    if (playBtn) playBtn.textContent = "⏯ Auto-Step";
  }

  function parseStepperSample() {
    if (!stepperSampleEl) return { x: [-0.55, 0.55], y: 1 };
    const parts = stepperSampleEl.value.split(",").map(Number);
    return {
      x: [parts[0] ?? -0.55, parts[1] ?? 0.55],
      y: parts[2] ?? 1
    };
  }

  function findNodeInTrace(trace, nodeId) {
    if (!trace) return null;
    for (const layerNodes of trace.nodesByLayer) {
      for (const node of layerNodes) {
        if (node.id === nodeId) return node;
      }
    }
    return null;
  }

  function computeAndRenderStepper(opts = {}) {
    if (!stepperSvg || !stepperModeEl) return;
    const sample = parseStepperSample();
    currentTrace = mlp.traceStep(sample.x, sample.y, {
      lr: Number(lrInput.value),
      optimizer: optSelect.value,
      mode: /** @type {"train" | "inference"} */ (stepperModeEl.value)
    });

    if (opts.resetStep) {
      currentMicroStepIdx = 0;
    } else if (currentMicroStepIdx >= currentTrace.microSteps.length) {
      currentMicroStepIdx = currentTrace.microSteps.length - 1;
    }

    if (!findNodeInTrace(currentTrace, selectedNodeId)) {
      selectedNodeId = currentTrace.nodesByLayer[1]?.[0]?.id || "L0N0";
    }

    renderStepperUI();
  }

  function renderStepperUI() {
    if (!currentTrace || !stepperSvg) return;
    const step = currentTrace.microSteps[currentMicroStepIdx] || currentTrace.microSteps[0];

    // 1. Render Micro-Step Tape
    const tapeEl = $("stepper-tape");
    if (tapeEl) {
      tapeEl.replaceChildren();
      currentTrace.microSteps.forEach((ms, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.role = "tab";
        const isCurrent = idx === currentMicroStepIdx;
        const isDone = idx < currentMicroStepIdx;
        btn.className = `stepper-pill phase-${ms.phase} ${isCurrent ? "active" : ""} ${isDone ? "done" : ""}`;
        btn.setAttribute("aria-selected", String(isCurrent));
        btn.textContent = ms.shortLabel;
        btn.addEventListener("click", () => {
          stopAutoPlay();
          currentMicroStepIdx = idx;
          const targetLayer = ms.activeLayer ?? 0;
          const firstNode = currentTrace.nodesByLayer[targetLayer]?.[0];
          if (firstNode) selectedNodeId = firstNode.id;
          renderStepperUI();
        });
        tapeEl.appendChild(btn);
      });
    }

    // 2. Render Stage Banner
    const badgeEl = $("stepper-phase-badge");
    if (badgeEl) {
      badgeEl.textContent = step.phase.toUpperCase();
      badgeEl.className = `stepper-phase-badge phase-${step.phase}`;
    }
    const titleEl = $("stepper-stage-title");
    if (titleEl) titleEl.textContent = step.title;
    const counterEl = $("stepper-pass-counter");
    if (counterEl) {
      counterEl.textContent = `Micro-step ${currentMicroStepIdx + 1} / ${currentTrace.microSteps.length} · Epoch ${mlp.stepCount}`;
    }
    const formulaEl = $("stepper-stage-formula");
    if (formulaEl) formulaEl.textContent = step.formula;
    const summaryEl = $("stepper-stage-summary");
    if (summaryEl) summaryEl.textContent = step.summary;

    // 3. Render SVG Graph & Node Inspector
    renderStepperSvg(currentTrace, step);
    renderNodeInspector(currentTrace, step);
  }

  function renderStepperSvg(trace, step) {
    if (!stepperSvg) return;
    stepperSvg.replaceChildren();

    const numLayers = trace.nodesByLayer.length; // L + 1
    const maxDim = Math.max(...trace.layerSizes);
    const W = 1120;
    const H = maxDim >= 10 ? 680 : maxDim >= 8 ? 620 : 560;
    stepperSvg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const nodeRadius = maxDim <= 4 ? 34 : maxDim <= 8 ? 25 : 19;

    // Background
    const bg = document.createElementNS(SVG_NS, "rect");
    bg.setAttribute("width", String(W));
    bg.setAttribute("height", String(H));
    bg.setAttribute("rx", "8");
    bg.setAttribute("fill", "#081424");
    stepperSvg.appendChild(bg);

    // Compute (cx, cy) for every node
    const padLeft = 115;
    const padRight = 275;
    const usableW = W - padLeft - padRight;
    const topY = 86;
    const botY = H - 44;
    const availH = botY - topY;

    /** @type {Array<Array<{ x: number, y: number, node: any }>>} */
    const coords = [];
    for (let l = 0; l < numLayers; l++) {
      const layerNodes = trace.nodesByLayer[l];
      const dim = layerNodes.length;
      const cx = numLayers === 1 ? W / 2 : padLeft + (l / (numLayers - 1)) * usableW;
      const stepY = dim === 1 ? 0 : Math.min(115, availH / (dim - 1));
      const totalH = stepY * (dim - 1);
      const startY = (topY + botY - totalH) / 2;

      const col = [];
      for (let j = 0; j < dim; j++) {
        col.push({
          x: cx,
          y: startY + j * stepY,
          node: layerNodes[j]
        });
      }
      coords.push(col);

      // Column Header Pill
      const isLayerActive = step.activeLayer === l || step.phase === "update";
      const headerGroup = document.createElementNS(SVG_NS, "g");
      const headerBg = document.createElementNS(SVG_NS, "rect");
      headerBg.setAttribute("x", String(cx - 78));
      headerBg.setAttribute("y", "14");
      headerBg.setAttribute("width", "156");
      headerBg.setAttribute("height", "34");
      headerBg.setAttribute("rx", "17");
      headerBg.setAttribute("fill", isLayerActive ? "#0369a1" : "#152c46");
      headerBg.setAttribute("stroke", isLayerActive ? "#bae6fd" : "#3b5978");
      headerBg.setAttribute("stroke-width", isLayerActive ? "1.8" : "1.2");

      const headerTxt = document.createElementNS(SVG_NS, "text");
      headerTxt.setAttribute("x", String(cx));
      headerTxt.setAttribute("y", "36");
      headerTxt.setAttribute("text-anchor", "middle");
      headerTxt.setAttribute("fill", "#ffffff");
      headerTxt.setAttribute("font-size", "13.5");
      headerTxt.setAttribute("font-weight", "800");
      headerTxt.setAttribute("font-family", "ui-monospace, monospace");
      headerTxt.textContent = l === 0
        ? `L0 · Input (${dim})`
        : l === numLayers - 1
          ? `L${l} · Output (${dim})`
          : `L${l} · Hidden (${dim})`;

      headerGroup.append(headerBg, headerTxt);
      stepperSvg.appendChild(headerGroup);
    }

    // Draw Synapse Edges
    const edgesGroup = document.createElementNS(SVG_NS, "g");
    const edgeLabelsGroup = document.createElementNS(SVG_NS, "g");
    const isUpdatePhase = step.phase === "update";

    for (let l = 1; l < numLayers; l++) {
      const isEdgeLayerActive = step.activeEdgeLayer === l - 1 || step.activeEdgeLayer === -2;
      for (let j = 0; j < coords[l].length; j++) {
        const dst = coords[l][j];
        for (let i = 0; i < coords[l - 1].length; i++) {
          const src = coords[l - 1][i];
          const edge = dst.node.incoming[i];
          const wVal = isUpdatePhase ? edge.wAfter : edge.wBefore;
          const isSelectedEdge = selectedNodeId === dst.node.id || selectedNodeId === src.node.id;

          let stroke = wVal >= 0 ? "#38bdf8" : "#fb7185";
          let opacity = isSelectedEdge ? 0.94 : 0.42;
          let width = 1.2 + Math.min(2.8, Math.abs(wVal)) * 1.6;

          if (isEdgeLayerActive) {
            opacity = 0.96;
            width += 1.4;
            if (step.phase === "forward") {
              stroke = edge.contrib >= 0 ? "#38bdf8" : "#fb7185";
            } else if (step.phase === "backward") {
              stroke = edge.dW >= 0 ? "#fbbf24" : "#f472b6";
            } else if (isUpdatePhase) {
              stroke = "#34d399";
            }
          }

          const line = document.createElementNS(SVG_NS, "line");
          line.setAttribute("x1", String(src.x));
          line.setAttribute("y1", String(src.y));
          line.setAttribute("x2", String(dst.x));
          line.setAttribute("y2", String(dst.y));
          line.setAttribute("stroke", stroke);
          line.setAttribute("stroke-width", width.toFixed(2));
          line.setAttribute("opacity", opacity.toFixed(2));
          if (isEdgeLayerActive && (step.phase === "forward" || step.phase === "backward")) {
            line.setAttribute("stroke-dasharray", "8 5");
          }
          edgesGroup.appendChild(line);

          // Show inline weight/gradient badge on incoming edges of the selected node when layer is compact
          if (selectedNodeId === dst.node.id && coords[l - 1].length <= 8) {
            const frac = coords[l - 1].length > 2 ? (0.42 + (i % 2) * 0.20) : 0.52;
            const mx = src.x + (dst.x - src.x) * frac;
            const my = src.y + (dst.y - src.y) * frac;
            const badgeText = step.phase === "backward"
              ? `∇W=${edge.dW >= 0 ? "+" : ""}${edge.dW.toFixed(2)}`
              : isUpdatePhase
                ? `W=${edge.wAfter >= 0 ? "+" : ""}${edge.wAfter.toFixed(2)}`
                : `W=${edge.wBefore >= 0 ? "+" : ""}${edge.wBefore.toFixed(2)}`;

            const lblBg = document.createElementNS(SVG_NS, "rect");
            lblBg.setAttribute("x", String(mx - 44));
            lblBg.setAttribute("y", String(my - 12));
            lblBg.setAttribute("width", "88");
            lblBg.setAttribute("height", "24");
            lblBg.setAttribute("rx", "5");
            lblBg.setAttribute("fill", "#06101e");
            lblBg.setAttribute("stroke", stroke);
            lblBg.setAttribute("stroke-width", "1.4");
            lblBg.setAttribute("opacity", "0.97");

            const lblTxt = document.createElementNS(SVG_NS, "text");
            lblTxt.setAttribute("x", String(mx));
            lblTxt.setAttribute("y", String(my + 4.5));
            lblTxt.setAttribute("text-anchor", "middle");
            lblTxt.setAttribute("fill", "#ffffff");
            lblTxt.setAttribute("font-size", "12");
            lblTxt.setAttribute("font-weight", "700");
            lblTxt.setAttribute("font-family", "ui-monospace, monospace");
            lblTxt.textContent = badgeText;

            edgeLabelsGroup.append(lblBg, lblTxt);
          }
        }
      }
    }
    stepperSvg.appendChild(edgesGroup);
    stepperSvg.appendChild(edgeLabelsGroup);

    // Draw Neuron Nodes
    const nodesGroup = document.createElementNS(SVG_NS, "g");
    for (let l = 0; l < numLayers; l++) {
      const isLayerActive = step.activeLayer === l || isUpdatePhase;
      for (let j = 0; j < coords[l].length; j++) {
        const { x, y, node } = coords[l][j];
        const isSelected = node.id === selectedNodeId;

        const g = document.createElementNS(SVG_NS, "g");
        g.setAttribute("class", "svg-node");
        g.setAttribute("role", "button");
        g.setAttribute("tabindex", "0");
        g.setAttribute(
          "aria-label",
          `Node ${node.label} in layer ${l}: activation ${node.a.toFixed(3)}, gradient delta ${node.delta.toFixed(3)}`
        );

        // Selection or active halo
        if (isSelected || isLayerActive) {
          const halo = document.createElementNS(SVG_NS, "circle");
          halo.setAttribute("cx", String(x));
          halo.setAttribute("cy", String(y));
          halo.setAttribute("r", String(nodeRadius + 6));
          halo.setAttribute("fill", "none");
          halo.setAttribute(
            "stroke",
            isSelected
              ? "#facc15"
              : step.phase === "backward"
                ? "#fb7185"
                : isUpdatePhase
                  ? "#34d399"
                  : "#38bdf8"
          );
          halo.setAttribute("stroke-width", isSelected ? "3.5" : "2.4");
          g.appendChild(halo);
        }

        // Node Circle Fill based on activation sign/magnitude (WCAG AA/AAA contrast for #ffffff text)
        const circle = document.createElementNS(SVG_NS, "circle");
        circle.setAttribute("cx", String(x));
        circle.setAttribute("cy", String(y));
        circle.setAttribute("r", String(nodeRadius));
        const actVal = node.a;
        const fill = actVal >= 0
          ? (actVal > 0.45 ? "#0369a1" : "#16324f")
          : "#9f1239";
        circle.setAttribute("fill", fill);
        circle.setAttribute("stroke", isSelected ? "#fef08a" : "#bae6fd");
        circle.setAttribute("stroke-width", isSelected ? "2.8" : "1.6");
        g.appendChild(circle);

        // Node text labels (high-contrast white text)
        if (nodeRadius >= 24) {
          const fontSizeTop = nodeRadius >= 32 ? "13.5" : "11.5";
          const fontSizeBot = nodeRadius >= 32 ? "12.5" : "11";
          const nameTxt = document.createElementNS(SVG_NS, "text");
          nameTxt.setAttribute("x", String(x));
          nameTxt.setAttribute("y", String(y - 4));
          nameTxt.setAttribute("text-anchor", "middle");
          nameTxt.setAttribute("fill", "#ffffff");
          nameTxt.setAttribute("font-size", fontSizeTop);
          nameTxt.setAttribute("font-weight", "800");
          nameTxt.setAttribute("font-family", "ui-monospace, monospace");
          nameTxt.textContent = node.label;

          const valTxt = document.createElementNS(SVG_NS, "text");
          valTxt.setAttribute("x", String(x));
          valTxt.setAttribute("y", String(y + 13));
          valTxt.setAttribute("text-anchor", "middle");
          valTxt.setAttribute("fill", "#f8fafc");
          valTxt.setAttribute("font-size", fontSizeBot);
          nameTxt.setAttribute("font-weight", "700");
          valTxt.setAttribute("font-family", "ui-monospace, monospace");
          valTxt.textContent = step.phase === "backward"
            ? `δ${node.delta >= 0 ? "+" : ""}${node.delta.toFixed(2)}`
            : node.a.toFixed(2);

          g.append(nameTxt, valTxt);
        } else {
          const valTxt = document.createElementNS(SVG_NS, "text");
          valTxt.setAttribute("x", String(x));
          valTxt.setAttribute("y", String(y + 4));
          valTxt.setAttribute("text-anchor", "middle");
          valTxt.setAttribute("fill", "#ffffff");
          valTxt.setAttribute("font-size", "11");
          valTxt.setAttribute("font-weight", "700");
          valTxt.setAttribute("font-family", "ui-monospace, monospace");
          valTxt.textContent = node.a.toFixed(2);
          g.appendChild(valTxt);
        }

        const selectThisNode = () => {
          selectedNodeId = node.id;
          renderStepperSvg(trace, step);
          renderNodeInspector(trace, step);
        };
        g.addEventListener("click", selectThisNode);
        g.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            selectThisNode();
          }
        });

        nodesGroup.appendChild(g);
      }
    }
    stepperSvg.appendChild(nodesGroup);

    // Output Readout & Loss Summary Box on Right of Output Node
    const outCoord = coords[numLayers - 1][0];
    if (outCoord) {
      const boxX = outCoord.x + nodeRadius + 24;
      const boxY = Math.max(70, Math.min(H - 170, outCoord.y - 76));
      const summaryGroup = document.createElementNS(SVG_NS, "g");

      const boxRect = document.createElementNS(SVG_NS, "rect");
      boxRect.setAttribute("x", String(boxX));
      boxRect.setAttribute("y", String(boxY));
      boxRect.setAttribute("width", "218");
      boxRect.setAttribute("height", "152");
      boxRect.setAttribute("rx", "8");
      boxRect.setAttribute("fill", "#0f243c");
      boxRect.setAttribute("stroke", isUpdatePhase ? "#34d399" : "#7dd3fc");
      boxRect.setAttribute("stroke-width", "1.8");
      summaryGroup.appendChild(boxRect);

      const shownPred = isUpdatePhase ? trace.predAfter : trace.predBefore;
      const shownLoss = isUpdatePhase ? trace.lossAfter : trace.lossBefore;
      const lines = [
        { label: isUpdatePhase ? "ŷ (after)" : "ŷ (pred)", val: shownPred.toFixed(4), color: "#7dd3fc" },
        { label: "Target y", val: String(trace.y), color: "#ffffff" },
        { label: "BCE Loss", val: shownLoss.toFixed(4), color: "#fde047" },
        { label: "δ (ŷ − y)", val: (trace.predBefore - trace.y).toFixed(4), color: "#fda4af" }
      ];

      lines.forEach((item, idx) => {
        const txt = document.createElementNS(SVG_NS, "text");
        txt.setAttribute("x", String(boxX + 14));
        txt.setAttribute("y", String(boxY + 34 + idx * 32));
        txt.setAttribute("fill", item.color);
        txt.setAttribute("font-size", "13.5");
        txt.setAttribute("font-weight", "700");
        txt.setAttribute("font-family", "ui-monospace, monospace");
        txt.textContent = `${item.label}: ${item.val}`;
        summaryGroup.appendChild(txt);
      });

      stepperSvg.appendChild(summaryGroup);
    }
  }

  function renderNodeInspector(trace, step) {
    const pillsEl = $("stepper-node-pills");
    const detailEl = $("stepper-node-detail");
    if (!pillsEl || !detailEl) return;

    // 1. Node selector pills
    pillsEl.replaceChildren();
    for (const layerNodes of trace.nodesByLayer) {
      for (const node of layerNodes) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `node-select-pill ${node.id === selectedNodeId ? "active" : ""}`;
        btn.setAttribute("aria-pressed", String(node.id === selectedNodeId));
        btn.textContent = node.label;
        btn.addEventListener("click", () => {
          selectedNodeId = node.id;
          renderStepperSvg(trace, step);
          renderNodeInspector(trace, step);
        });
        pillsEl.appendChild(btn);
      }
    }

    // 2. Detailed Node & Synapse Table
    const node = findNodeInTrace(trace, selectedNodeId) || trace.nodesByLayer[0][0];
    detailEl.replaceChildren();

    const header = document.createElement("div");
    header.className = "inspector-node-header";
    const title = document.createElement("strong");
    title.textContent = node.role === "input"
      ? `Node ${node.label} · Input Feature (Layer 0)`
      : node.role === "output"
        ? `Node ${node.label} · Output Readout (Layer ${node.layerIndex}, ${node.activationFn})`
        : `Node ${node.label} · Hidden Neuron (Layer ${node.layerIndex}, ${node.activationFn})`;
    header.appendChild(title);
    detailEl.appendChild(header);

    const splitWrap = document.createElement("div");
    splitWrap.className = "inspector-detail-split";

    const kpiGrid = document.createElement("div");
    kpiGrid.className = "inspector-kpi-grid";
    const kpis = node.role === "input"
      ? [
          { k: "Input Value x", v: node.a.toFixed(4) },
          { k: "Input Sensitivity ∂L/∂x", v: node.delta.toFixed(4) }
        ]
      : [
          { k: "Pre-act z", v: node.z.toFixed(4) },
          { k: `Act a = ${node.activationFn}(z)`, v: node.a.toFixed(4) },
          { k: "Local Deriv σ'(z)", v: node.actDeriv.toFixed(4) },
          { k: "Error Signal δ = ∂L/∂z", v: node.delta.toFixed(4) },
          { k: "Bias b (before → after)", v: `${node.bBefore.toFixed(3)} → ${node.bAfter.toFixed(3)}` },
          { k: "Bias Grad ∂L/∂b", v: node.db.toFixed(4) }
        ];

    for (const item of kpis) {
      const cell = document.createElement("div");
      cell.className = "inspector-kpi";
      const lbl = document.createElement("span");
      lbl.className = "inspector-kpi-label";
      lbl.textContent = item.k;
      const val = document.createElement("span");
      val.className = "inspector-kpi-val";
      val.textContent = item.v;
      cell.append(lbl, val);
      kpiGrid.appendChild(cell);
    }
    splitWrap.appendChild(kpiGrid);

    if (node.incoming && node.incoming.length > 0) {
      const tblWrap = document.createElement("div");
      tblWrap.className = "table-scroll";
      const tbl = document.createElement("table");
      tbl.className = "attn-table synapse-table";
      const thead = document.createElement("thead");
      const htr = document.createElement("tr");
      ["From Synapse", "a_prev", "Weight W", "Signal a·W", "Grad ∂L/∂W", "Updated W_new"].forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col;
        htr.appendChild(th);
      });
      thead.appendChild(htr);
      tbl.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const edge of node.incoming) {
        const tr = document.createElement("tr");
        [
          edge.fromLabel,
          edge.aPrev.toFixed(3),
          edge.wBefore.toFixed(3),
          edge.contrib.toFixed(3),
          edge.dW.toFixed(4),
          edge.wAfter.toFixed(3)
        ].forEach((valStr) => {
          const td = document.createElement("td");
          td.textContent = valStr;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      }
      tbl.appendChild(tbody);
      tblWrap.appendChild(tbl);
      splitWrap.appendChild(tblWrap);
    } else {
      const note = document.createElement("p");
      note.className = "small";
      note.textContent = "Input layer node: receives external coordinate feature directly. Click a Hidden (h) or Output (ŷ) node to inspect incoming synapse weights and gradients.";
      splitWrap.appendChild(note);
    }

    detailEl.appendChild(splitWrap);
  }

  const graphSizeSelect = /** @type {HTMLSelectElement | null} */ ($("stepper-graph-size"));
  const svgWrapEl = $("stepper-svg-wrap");
  if (graphSizeSelect && svgWrapEl) {
    graphSizeSelect.addEventListener("change", () => {
      svgWrapEl.classList.remove("scale-compact", "scale-large", "scale-theatre");
      svgWrapEl.classList.add(`scale-${graphSizeSelect.value}`);
    });
  }

  if (stepperModeEl && stepperSampleEl) {
    stepperModeEl.addEventListener("change", () => {
      stopAutoPlay();
      computeAndRenderStepper({ resetStep: true });
    });
    stepperSampleEl.addEventListener("change", () => {
      stopAutoPlay();
      computeAndRenderStepper({ resetStep: false });
    });
  }

  $("stepper-reset-btn")?.addEventListener("click", () => {
    stopAutoPlay();
    currentMicroStepIdx = 0;
    selectedNodeId = "L0N0";
    renderStepperUI();
  });

  $("stepper-prev-btn")?.addEventListener("click", () => {
    stopAutoPlay();
    if (!currentTrace) return;
    currentMicroStepIdx = Math.max(0, currentMicroStepIdx - 1);
    const ms = currentTrace.microSteps[currentMicroStepIdx];
    const firstNode = currentTrace.nodesByLayer[ms.activeLayer ?? 0]?.[0];
    if (firstNode) selectedNodeId = firstNode.id;
    renderStepperUI();
  });

  function stepForwardOne() {
    if (!currentTrace) return;
    if (currentMicroStepIdx < currentTrace.microSteps.length - 1) {
      currentMicroStepIdx++;
      const ms = currentTrace.microSteps[currentMicroStepIdx];
      const firstNode = currentTrace.nodesByLayer[ms.activeLayer ?? 0]?.[0];
      if (firstNode) selectedNodeId = firstNode.id;
      renderStepperUI();
    } else if (currentTrace.mode === "train") {
      // Commit weight update at end of pass and start next pass
      mlp.commitTrace(currentTrace);
      currentMicroStepIdx = 0;
      selectedNodeId = "L0N0";
      renderMlpState();
    } else {
      currentMicroStepIdx = 0;
      selectedNodeId = "L0N0";
      renderStepperUI();
    }
  }

  $("stepper-next-btn")?.addEventListener("click", () => {
    stopAutoPlay();
    stepForwardOne();
  });

  $("stepper-play-btn")?.addEventListener("click", () => {
    if (autoPlayTimer) {
      stopAutoPlay();
      return;
    }
    const playBtn = $("stepper-play-btn");
    if (playBtn) playBtn.textContent = "⏸ Pause Auto-Step";
    autoPlayTimer = setInterval(() => {
      if (!currentTrace) return;
      if (currentMicroStepIdx < currentTrace.microSteps.length - 1) {
        stepForwardOne();
      } else {
        stopAutoPlay();
      }
    }, 600);
  });

  $("stepper-commit-btn")?.addEventListener("click", () => {
    stopAutoPlay();
    if (!currentTrace) return;
    mlp.commitTrace(currentTrace);
    renderMlpState();
    if (currentTrace && currentTrace.mode === "train") {
      currentMicroStepIdx = currentTrace.microSteps.length - 1;
      renderStepperUI();
    }
  });

  lrInput.addEventListener("input", () => {
    lrVal.textContent = Number(lrInput.value).toFixed(3);
    computeAndRenderStepper();
    renderHyperparameterExplainer();
  });

  optSelect.addEventListener("change", () => {
    computeAndRenderStepper();
    renderHyperparameterExplainer();
  });

  dsSelect.addEventListener("change", () => {
    stopAutoPlay();
    dataset = generate2DDataset(dsSelect.value, 120, 42);
    mlp = createMlp();
    renderMlpState();
  });

  [archSelect, actSelect].forEach((sel) => {
    sel.addEventListener("change", () => {
      stopAutoPlay();
      mlp = createMlp();
      selectedNodeId = "L1N0";
      renderMlpState();
    });
  });

  $("mlp-reset-btn").addEventListener("click", () => {
    stopAutoPlay();
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
    stopAutoPlay();
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
    const inferredY = p >= 0.5 ? 1 : 0;
    $("mlp-probe-readout").textContent =
      `Probe (${x1.toFixed(2)}, ${x2.toFixed(2)}) → P(Class 1) = ${(p * 100).toFixed(1)}% (${inferredY === 1 ? "Blue Class 1" : "Red Class 0"})`;

    if (stepperSampleEl) {
      let customOpt = /** @type {HTMLOptionElement | null} */ (stepperSampleEl.querySelector('option[data-custom="1"]'));
      if (!customOpt) {
        customOpt = document.createElement("option");
        customOpt.setAttribute("data-custom", "1");
        stepperSampleEl.appendChild(customOpt);
      }
      customOpt.value = `${x1.toFixed(3)},${x2.toFixed(3)},${inferredY}`;
      customOpt.textContent = `Canvas Probe: x = (${x1.toFixed(2)}, ${x2.toFixed(2)}) → y = ${inferredY}`;
      stepperSampleEl.value = customOpt.value;
      computeAndRenderStepper({ resetStep: false });
    }
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
  if (!$("cnn-train-btn")) return;
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
    cnnStepper?.refresh();
  }

  let cnnStepper = null;

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
  cnnStepper = createCnnStepper($("cnn-stepper"), () => ({ cnn, currentImg }));
  renderCnnForward(initStats);
}

// ============================================================================
// 2B. RECURRENT NEURAL NETWORKS (RNN / GRU) & BPTT LAB
// ============================================================================

function initRnnLab() {
  if (!$("rnn-train-btn")) return;
  const cellSelect = /** @type {HTMLSelectElement | null} */ ($("rnn-cell-type"));
  const lenSelect = /** @type {HTMLSelectElement | null} */ ($("rnn-seq-len"));
  const taskSelect = /** @type {HTMLSelectElement | null} */ ($("rnn-task"));
  const stripEl = $("rnn-temporal-strip");
  if (!cellSelect || !lenSelect || !taskSelect || !stripEl) return;

  let seqLen = parseInt(lenSelect.value, 10) || 8;
  let batch = generateTemporalSequenceBatch(taskSelect.value, seqLen, 28, 303);
  let rnn = createRnn();
  let lastRnnStats = null;
  let rnnStepper = null;

  function createRnn() {
    seqLen = parseInt(lenSelect.value, 10) || 8;
    batch = generateTemporalSequenceBatch(taskSelect.value, seqLen, 28, 303);
    return new MicroRNN({
      hiddenDim: 10,
      cellType: /** @type {"rnn" | "gru"} */ (cellSelect.value),
      seed: 1997
    });
  }

  function renderRnn(stats) {
    if (!stats) return;
    lastRnnStats = stats;
    $("rnn-epoch").textContent = String(stats.step);
    $("rnn-loss").textContent = stats.loss.toFixed(4);
    $("rnn-acc").textContent = `${(stats.accuracy * 100).toFixed(1)}%`;
    $("rnn-grad-ratio").textContent = `${(stats.gradRetentionRatio * 100).toFixed(1)}%`;

    stripEl.replaceChildren();
    const maxGrad = Math.max(1e-4, ...stats.temporalSteps.map((s) => s.gradNorm));
    stats.temporalSteps.forEach((s) => {
      const row = document.createElement("div");
      row.className = "grad-row";
      const label = document.createElement("span");
      const roleTag = s.t === 0 ? "Trigger x₀" : s.t === stats.temporalSteps.length - 1 ? "Query" : "Distractor";
      label.textContent = `t=${s.t} (${roleTag}, z̄=${s.gateAvg.toFixed(2)})`;
      const meter = document.createElement("meter");
      meter.min = 0;
      meter.max = maxGrad;
      meter.value = s.gradNorm;
      const val = document.createElement("span");
      val.className = "number";
      val.textContent = `‖∂L/∂h_${s.t}‖=${s.gradNorm.toFixed(4)}`;
      row.append(label, meter, val);
      stripEl.appendChild(row);
    });
    rnnStepper?.refresh();
  }

  function resetAndWarmup() {
    rnn = createRnn();
    let st = null;
    for (let e = 0; e < 22; e++) st = rnn.trainEpoch(batch, 0.12);
    renderRnn(st);
  }

  [cellSelect, lenSelect, taskSelect].forEach((el) => {
    el.addEventListener("change", resetAndWarmup);
  });

  $("rnn-train-btn")?.addEventListener("click", async () => {
    const btn = /** @type {HTMLButtonElement} */ ($("rnn-train-btn"));
    btn.disabled = true;
    const yielder = createYieldController(50);
    let st = null;
    for (let e = 0; e < 35; e++) {
      st = rnn.trainEpoch(batch, 0.12);
      if (e % 7 === 6 || e === 34) renderRnn(st);
      await yielder.maybeYield();
    }
    btn.disabled = false;
  });

  $("rnn-reset-btn")?.addEventListener("click", () => {
    rnn = createRnn();
    const st = rnn.trainEpoch(batch, 0.0001);
    renderRnn(st);
  });

  resetAndWarmup();
  rnnStepper = createRnnStepper($("rnn-stepper"), () => ({ rnn, batch, lastStats: lastRnnStats }));
}

// ============================================================================
// 2C. VERY DEEP RESIDUAL NETWORKS (RESNETS, RMSNORM & INIT) LAB
// ============================================================================

function initDeepResNetLab() {
  if (!$("resnet-train-btn")) return;
  const depthSelect = /** @type {HTMLSelectElement | null} */ ($("resnet-depth"));
  const skipSelect = /** @type {HTMLSelectElement | null} */ ($("resnet-skip"));
  const normSelect = /** @type {HTMLSelectElement | null} */ ($("resnet-norm"));
  const initSelect = /** @type {HTMLSelectElement | null} */ ($("resnet-init"));
  const barsEl = $("resnet-depth-bars");
  if (!depthSelect || !skipSelect || !normSelect || !initSelect || !barsEl) return;

  const dataset = generate2DDataset("xor", 48, 77);
  let net = createNet();
  let lastResStats = null;
  let resnetStepper = null;

  function createNet() {
    return new DeepResidualNetwork({
      depth: parseInt(depthSelect.value, 10) || 12,
      dim: 8,
      residual: skipSelect.value === "residual",
      norm: /** @type {"rmsnorm" | "none"} */ (normSelect.value),
      initScheme: /** @type {"he" | "xavier" | "small" | "large"} */ (initSelect.value),
      seed: 2015
    });
  }

  function renderResNet(stats) {
    if (!stats) return;
    lastResStats = stats;
    $("resnet-step").textContent = String(stats.step);
    $("resnet-loss").textContent = stats.loss.toFixed(4);
    $("resnet-acc").textContent = `${(stats.accuracy * 100).toFixed(1)}%`;
    $("resnet-grad-ratio").textContent = `${(stats.firstToLastGradRatio * 100).toFixed(1)}%`;

    barsEl.replaceChildren();
    const maxGrad = Math.max(0.05, ...stats.layerStats.map((s) => s.gradNorm));
    stats.layerStats.forEach((s) => {
      const row = document.createElement("div");
      row.className = "grad-row";
      const label = document.createElement("span");
      label.textContent = `Block L${s.layerIndex} (RMS=${s.actRms.toFixed(2)})`;
      const meter = document.createElement("meter");
      meter.min = 0;
      meter.max = maxGrad;
      meter.value = Math.min(maxGrad, s.gradNorm);
      const val = document.createElement("span");
      val.className = "number";
      val.textContent = `‖∇W‖=${s.gradNorm.toFixed(4)}`;
      row.append(label, meter, val);
      barsEl.appendChild(row);
    });
    resnetStepper?.refresh();
  }

  function resetAndProbe() {
    net = createNet();
    let st = null;
    for (let i = 0; i < 8; i++) st = net.trainStep(dataset, 0.05);
    renderResNet(st);
  }

  [depthSelect, skipSelect, normSelect, initSelect].forEach((el) => {
    el.addEventListener("change", resetAndProbe);
  });

  $("resnet-train-btn")?.addEventListener("click", async () => {
    const btn = /** @type {HTMLButtonElement} */ ($("resnet-train-btn"));
    btn.disabled = true;
    const yielder = createYieldController(50);
    let st = null;
    for (let i = 0; i < 25; i++) {
      st = net.trainStep(dataset, 0.05);
      if (i % 5 === 4 || i === 24) renderResNet(st);
      await yielder.maybeYield();
    }
    btn.disabled = false;
  });

  $("resnet-reset-btn")?.addEventListener("click", resetAndProbe);

  resetAndProbe();
  resnetStepper = createResNetStepper($("resnet-stepper"), () => ({ net, lastStats: lastResStats }));
}

// ============================================================================
// 3. TRANSFORMER & SELF-ATTENTION LAB
// ============================================================================

function initTransformerLab() {
  if (!$("tf-train-btn")) return;
  const taskSelect = /** @type {HTMLSelectElement} */ ($("tf-task"));
  const maskSelect = /** @type {HTMLSelectElement} */ ($("tf-mask"));
  const seqInput = /** @type {HTMLInputElement} */ ($("tf-input-seq"));
  if (!taskSelect || !seqInput) return;

  let batch = generateSequenceBatch(taskSelect.value, 4, 28, 99);
  let tf = createTf();
  let tfStepper = null;

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
    tfStepper?.refresh();
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
  tfStepper = createTransformerStepper($("transformer-stepper"), () => ({ tf, tokenIds: parseProbeTokens() }));
  renderTransformer(initSt);
}

// ============================================================================
// 4. DIFFUSION LAB (IMAGE DDPM + MASKED TEXT DIFFUSION)
// ============================================================================

function initDiffusionLab() {
  if (!$("ddpm-train-btn")) return;
  const targetSelect = /** @type {HTMLSelectElement} */ ($("ddpm-target"));
  const stepSlider = /** @type {HTMLInputElement} */ ($("ddpm-step-slider"));
  const stepVal = $("ddpm-step-val");
  if (!targetSelect) return;

  let ddpm = new MicroDDPM({ dim: 36, steps: 16, hiddenDim: 48, seed: 512 });
  let trajectory = [];
  let diffStepper = null;

  function renderDdpm(stats = null, seed = 808) {
    if (stats) {
      $("ddpm-epoch").textContent = String(stats.step);
      $("ddpm-loss").textContent = stats.loss.toFixed(4);
    }
    trajectory = ddpm.sampleTrajectory(seed);
    renderDdpmSnapshots();
    diffStepper?.refresh();
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
    diffStepper?.refresh();
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

  diffStepper = createDiffusionStepper($("diffusion-stepper"), () => ({
    ddpm,
    trajectory,
    textDiff,
    textPromptIdx: parseInt(textPromptSelect.value, 10) || 0
  }));
}

// ============================================================================
// 5. DECISION POINTER HEAD LAB
// ============================================================================

function initDecisionLab() {
  if (!$("dec-temp")) return;
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

  let decStepper = null;
  let lastDecState = null;

  function update() {
    const T = Number(tempInput.value);
    const tau = Number(gateInput.value);
    $("dec-temp-val").textContent = T.toFixed(2);
    $("dec-gate-val").textContent = tau.toFixed(2);

    const stateVec = stateScenarios[scenSelect.value] || stateScenarios.clear;
    const res = head.decide(stateVec, optionVecs, { temperature: T, gateThreshold: tau });
    lastDecState = { stateVec, optionVecs, optionLabels, T, tau, res };

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
    decStepper?.refresh();
  }

  [scenSelect, tempInput, gateInput].forEach((el) => el.addEventListener("input", update));
  update();
  decStepper = createDecisionStepper($("decision-stepper"), () => lastDecState);
}

// ============================================================================
// 6. EXTENSIBLE PLUG-AND-PLAY ARCHITECTURE BUILDER
// ============================================================================

function initBlockBuilder() {
  if (!$("builder-stack")) return;
  const paletteEl = $("builder-palette");
  const stackEl = $("builder-stack");
  if (!paletteEl || !stackEl) return;

  let pipeline = ["embedding", "layernorm", "attention", "layernorm", "mlp_block", "decision_head"];
  let builderStepper = null;

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
    builderStepper?.refresh();
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
  builderStepper = createBuilderStepper($("builder-stepper"), () => pipeline);
}

export function initNeuralNetworksApp() {
  if (typeof document === "undefined") return;
  initKernelBar();
  initCodeExplorers();
  initMlpLab();
  initCnnLab();
  initRnnLab();
  initDeepResNetLab();
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
