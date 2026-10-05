/**
 * Interactive Per-Section Neural Network Node Graph & Step-by-Step Inference/Training Debuggers
 * Renders a full-width, high-contrast SVG computational graph and 2-column Neuron & Synapse Inspector
 * for CNNs, RNNs/GRUs, Deep ResNets, Transformers, Diffusion Models, Decision Pointer Heads, and Custom Stacks.
 */

import { GLYPH_CLASSES, VOCAB_TOKENS, analyzeArchitecturePipeline } from "./engine.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Mounts a reusable, full-width Interactive Network Node Graph & Step-by-Step Debugger inside `containerEl`.
 * @param {HTMLElement | null} containerEl
 * @param {{
 *   idPrefix: string,
 *   title: string,
 *   subtitle: string,
 *   probeLabel?: string,
 *   probeOptions?: Array<{ value: string, label: string }>,
 *   buildTrace: (probeValue: string) => {
 *     columns: Array<{
 *       header: string,
 *       nodes: Array<{
 *         id: string,
 *         label: string,
 *         sublabel?: string,
 *         val: number,
 *         displayVal?: string,
 *         roleTitle: string,
 *         kpis: Array<{ k: string, v: string }>,
 *         incoming: Array<{
 *           fromId: string,
 *           fromLabel: string,
 *           inVal: string,
 *           weightVal: number,
 *           weightStr: string,
 *           contribVal: number,
 *           contribStr: string,
 *           detailStr: string
 *         }>
 *       }>
 *     }>,
 *     microSteps: Array<{
 *       phase: "input" | "forward" | "backward" | "loss" | "update",
 *       badge?: string,
 *       shortLabel: string,
 *       title: string,
 *       formula: string,
 *       summary: string,
 *       activeCol: number,
 *       activeEdgeCol: number
 *     }>,
 *     readoutBox: {
 *       title?: string,
 *       lines: Array<{ label: string, val: string, color: string }>
 *     },
 *     epochLabel: string
 *   }
 * }} config
 */
function el(tag, text, cls = "", attrs = {}, children = []) {
  const n = document.createElement(tag);
  if (text) n.textContent = text;
  if (cls) n.className = cls;
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'id' && !v) continue; // skip empty ids
    n.setAttribute(k, v);
  }
  if (children.length) n.append(...children);
  return n;
}

export function mountSectionStepper(containerEl, config) {
  if (!containerEl) return { refresh() {} };

  const pfx = config.idPrefix;
  let currentStepIdx = 0;
  let selectedNodeId = "";
  let autoPlayTimer = null;
  let currentTrace = null;

  // Build Workbench DOM
  containerEl.className = "nn-stepper-workbench";
  containerEl.innerHTML = "";

  const titleDiv = el('div', '', '', {}, [
    el('h3', config.title),
    el('p', config.subtitle, 'small')
  ]);

  const sizeSelect = el('select', '', '', { id: `${pfx}-graph-size` }, [
    el('option', 'Large Stage (520px Height)', '', { value: 'large', selected: 'selected' }),
    el('option', 'Extra-Large Theatre (680px Height)', '', { value: 'theatre' }),
    el('option', 'Compact (380px Height)', '', { value: 'compact' })
  ]);
  const sizeDiv = el('div', '', 'stepper-size-control', {}, [
    el('label', '', 'small', { for: `${pfx}-graph-size` }, [el('strong', 'Graph Canvas Scale')]),
    sizeSelect
  ]);
  const headerDiv = el('div', '', 'stepper-header', {}, [titleDiv, sizeDiv]);

  const toolbarDiv = el('div', '', 'stepper-toolbar');
  if (config.probeOptions && config.probeOptions.length > 0) {
    const probeSelect = el('select', '', '', { id: `${pfx}-probe-select` }, 
      config.probeOptions.map(o => el('option', o.label, '', { value: o.value }))
    );
    const probeLabel = el('label', config.probeLabel || "Inspection Target / Mode", '', { for: `${pfx}-probe-select` });
    toolbarDiv.append(el('div', '', 'fields stepper-fields', {}, [
      el('div', '', '', {}, [probeLabel, probeSelect])
    ]));
  }
  const actionsDiv = el('div', '', 'actions stepper-actions', {}, [
    el('button', '⏮ Start of Pass', 'secondary', { type: 'button', id: `${pfx}-reset-btn` }),
    el('button', '◀ Prev Step', 'secondary', { type: 'button', id: `${pfx}-prev-btn` }),
    el('button', 'Next Step ▶', '', { type: 'button', id: `${pfx}-next-btn` }),
    el('button', '⏯ Auto-Step', 'secondary', { type: 'button', id: `${pfx}-play-btn` })
  ]);
  toolbarDiv.append(actionsDiv);

  const tapeDiv = el('div', '', 'stepper-tape', { id: `${pfx}-tape`, role: 'tablist', 'aria-label': `${config.title} micro-steps` });

  const stageBannerDiv = el('div', '', 'stepper-stage-banner', {}, [
    el('div', '', 'stepper-stage-top', {}, [
      el('span', 'INPUT', 'stepper-phase-badge', { id: `${pfx}-phase-badge` }),
      el('strong', 'Stage 1', '', { id: `${pfx}-stage-title` }),
      el('span', 'Step 1', 'stepper-pass-counter', { id: `${pfx}-pass-counter` })
    ]),
    el('div', '', 'stepper-stage-formula', { id: `${pfx}-stage-formula` }),
    el('div', '', 'stepper-stage-summary', { id: `${pfx}-stage-summary` })
  ]);

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("id", `${pfx}-svg`);
  svg.setAttribute("viewBox", "0 0 1120 560");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${config.title} node graph`);
  svg.setAttribute("class", "nn-stepper-svg");

  const svgWrapDiv = el('div', '', 'stepper-svg-container scale-large', { id: `${pfx}-svg-wrap` });
  svgWrapDiv.append(svg);

  const legendDiv = el('div', '', 'stepper-legend', {}, [
    el('span', '', 'legend-item', {}, [el('span', '', 'legend-swatch swatch-pos'), ' Positive signal / weight (> 0)']),
    el('span', '', 'legend-item', {}, [el('span', '', 'legend-swatch swatch-neg'), ' Negative / inhibitory (< 0)']),
    el('span', '', 'legend-item', {}, [el('span', '', 'legend-swatch swatch-fwd'), ' Active Forward / Routing Wave']),
    el('span', '', 'legend-item', {}, [el('span', '', 'legend-swatch swatch-upd'), ' Highway / Gate / Attention Mix']),
    el('span', '', 'legend-item', {}, [el('span', '', 'legend-swatch swatch-bwd'), ' Backward Gradient Wave (∂L/∂h)'])
  ]);

  const graphCardDiv = el('div', '', 'stepper-graph-card', {}, [svgWrapDiv, legendDiv]);

  const inspectorCardDiv = el('div', '', 'stepper-inspector-card', {}, [
    el('h4', 'Neuron, Filter & Routing Inspector (Click any node in graph above)'),
    el('div', '', 'stepper-node-pills', { id: `${pfx}-node-pills`, role: 'group', 'aria-label': 'Select node to inspect' }),
    el('div', '', 'stepper-node-detail', { id: `${pfx}-node-detail`, 'aria-live': 'polite' })
  ]);

  const bodyGridDiv = el('div', '', 'stepper-body-grid', {}, [
    graphCardDiv,
    inspectorCardDiv
  ]);

  containerEl.append(headerDiv, toolbarDiv, tapeDiv, stageBannerDiv, bodyGridDiv);

  const byId = (id) => containerEl.querySelector(`#${id}`);
  const probeSelect = /** @type {HTMLSelectElement | null} */ (byId(`${pfx}-probe-select`));
  const sizeSelect = /** @type {HTMLSelectElement | null} */ (byId(`${pfx}-graph-size`));
  const svgWrap = byId(`${pfx}-svg-wrap`);
  const svgEl = byId(`${pfx}-svg`);

  function stopAutoPlay() {
    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
    }
    const playBtn = byId(`${pfx}-play-btn`);
    if (playBtn) playBtn.textContent = "⏯ Auto-Step";
  }

  function findNode(trace, id) {
    for (const col of trace.columns) {
      for (const n of col.nodes) {
        if (n.id === id) return n;
      }
    }
    return trace.columns[0]?.nodes[0] || null;
  }

  function renderUI() {
    if (!currentTrace) return;
    currentStepIdx = Math.max(0, Math.min(currentStepIdx, currentTrace.microSteps.length - 1));
    const step = currentTrace.microSteps[currentStepIdx];

    if (!selectedNodeId || !findNode(currentTrace, selectedNodeId)) {
      const targetCol = Math.max(0, Math.min(currentTrace.columns.length - 1, step.activeCol ?? 0));
      selectedNodeId = currentTrace.columns[targetCol]?.nodes[0]?.id || currentTrace.columns[0]?.nodes[0]?.id || "";
    }

    // 1. Tape pills
    const tapeEl = byId(`${pfx}-tape`);
    if (tapeEl) {
      tapeEl.replaceChildren();
      currentTrace.microSteps.forEach((ms, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.role = "tab";
        const isCurrent = idx === currentStepIdx;
        const isDone = idx < currentStepIdx;
        btn.className = `stepper-pill phase-${ms.phase} ${isCurrent ? "active" : ""} ${isDone ? "done" : ""}`;
        btn.setAttribute("aria-selected", String(isCurrent));
        btn.textContent = ms.shortLabel;
        btn.addEventListener("click", () => {
          stopAutoPlay();
          currentStepIdx = idx;
          const targetCol = Math.max(0, Math.min(currentTrace.columns.length - 1, ms.activeCol ?? 0));
          const firstNode = currentTrace.columns[targetCol]?.nodes[0];
          if (firstNode) selectedNodeId = firstNode.id;
          renderUI();
        });
        tapeEl.appendChild(btn);
      });
    }

    // 2. Stage Banner
    const badgeEl = byId(`${pfx}-phase-badge`);
    if (badgeEl) {
      badgeEl.textContent = (step.badge || step.phase).toUpperCase();
      badgeEl.className = `stepper-phase-badge phase-${step.phase}`;
    }
    const titleEl = byId(`${pfx}-stage-title`);
    if (titleEl) titleEl.textContent = step.title;
    const counterEl = byId(`${pfx}-pass-counter`);
    if (counterEl) {
      counterEl.textContent = `Micro-step ${currentStepIdx + 1} / ${currentTrace.microSteps.length} · ${currentTrace.epochLabel}`;
    }
    const formulaEl = byId(`${pfx}-stage-formula`);
    if (formulaEl) formulaEl.textContent = step.formula;
    const summaryEl = byId(`${pfx}-stage-summary`);
    if (summaryEl) summaryEl.textContent = step.summary;

    renderSvg(currentTrace, step);
    renderInspector(currentTrace, step);
  }

  function renderSvg(trace, step) {
    if (!svgEl) return;
    svgEl.replaceChildren();

    const numCols = trace.columns.length;
    const maxDim = Math.max(...trace.columns.map((c) => c.nodes.length));
    const W = 1120;
    const H = maxDim >= 6 ? 600 : 550;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const nodeRadius = maxDim <= 4 ? 30 : maxDim <= 6 ? 25 : 20;

    const bg = document.createElementNS(SVG_NS, "rect");
    bg.setAttribute("width", String(W));
    bg.setAttribute("height", String(H));
    bg.setAttribute("rx", "8");
    bg.setAttribute("fill", "#081424");
    svgEl.appendChild(bg);

    const padLeft = 110;
    const padRight = 292;
    const usableW = W - padLeft - padRight;
    const topY = 92;
    const botY = H - 54;
    const availH = botY - topY;

    /** @type {Map<string, { x: number, y: number, colIdx: number, node: any }>} */
    const nodePosMap = new Map();
    /** @type {Array<Array<{ x: number, y: number, node: any }>>} */
    const coords = [];

    for (let c = 0; c < numCols; c++) {
      const colObj = trace.columns[c];
      const dim = colObj.nodes.length;
      const cx = numCols === 1 ? W / 2 : padLeft + (c / (numCols - 1)) * usableW;
      const stepY = dim === 1 ? 0 : Math.min(104, availH / (dim - 1));
      const totalH = stepY * (dim - 1);
      const startY = (topY + botY - totalH) / 2;

      const colCoords = [];
      for (let j = 0; j < dim; j++) {
        const pt = { x: cx, y: startY + j * stepY, colIdx: c, node: colObj.nodes[j] };
        colCoords.push(pt);
        nodePosMap.set(colObj.nodes[j].id, pt);
      }
      coords.push(colCoords);

      // Column Header Pill
      const isColActive = step.activeCol === c || step.activeEdgeCol === -2;
      const headerGroup = document.createElementNS(SVG_NS, "g");
      const headerBg = document.createElementNS(SVG_NS, "rect");
      headerBg.setAttribute("x", String(cx - 82));
      headerBg.setAttribute("y", "14");
      headerBg.setAttribute("width", "164");
      headerBg.setAttribute("height", "34");
      headerBg.setAttribute("rx", "17");
      headerBg.setAttribute("fill", isColActive ? "#0369a1" : "#152c46");
      headerBg.setAttribute("stroke", isColActive ? "#bae6fd" : "#3b5978");
      headerBg.setAttribute("stroke-width", isColActive ? "1.8" : "1.2");

      const headerTxt = document.createElementNS(SVG_NS, "text");
      headerTxt.setAttribute("x", String(cx));
      headerTxt.setAttribute("y", "36");
      headerTxt.setAttribute("text-anchor", "middle");
      headerTxt.setAttribute("fill", "#ffffff");
      headerTxt.setAttribute("font-size", "13");
      headerTxt.setAttribute("font-weight", "800");
      headerTxt.setAttribute("font-family", "ui-monospace, monospace");
      headerTxt.textContent = colObj.header;

      headerGroup.append(headerBg, headerTxt);
      svgEl.appendChild(headerGroup);
    }

    // Draw Edges from each node's `incoming` list
    const edgesGroup = document.createElementNS(SVG_NS, "g");
    const edgeLabelsGroup = document.createElementNS(SVG_NS, "g");

    for (let c = 1; c < numCols; c++) {
      const isEdgeLayerActive = step.activeEdgeCol === c - 1 || step.activeEdgeCol === -2;
      for (let j = 0; j < coords[c].length; j++) {
        const dst = coords[c][j];
        const incoming = dst.node.incoming || [];
        incoming.forEach((edge, i) => {
          const src = nodePosMap.get(edge.fromId);
          if (!src) return;

          const wVal = edge.weightVal;
          const isSelectedEdge = selectedNodeId === dst.node.id || selectedNodeId === src.node.id;
          let stroke = wVal >= 0 ? "#38bdf8" : "#fb7185";
          let opacity = isSelectedEdge ? 0.94 : 0.42;
          let width = 1.3 + Math.min(2.8, Math.abs(wVal)) * 1.5;

          if (isEdgeLayerActive) {
            opacity = 0.96;
            width += 1.3;
            if (step.phase === "backward") {
              stroke = "#fbbf24";
            } else if (step.phase === "update") {
              stroke = "#34d399";
            } else {
              stroke = edge.contribVal >= 0 ? "#38bdf8" : "#fb7185";
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

          // Inline weight/routing badge on incoming edges of selected node
          if (selectedNodeId === dst.node.id && incoming.length <= 6) {
            const frac = incoming.length > 2 ? 0.4 + (i % 2) * 0.22 : 0.5;
            const mx = src.x + (dst.x - src.x) * frac;
            const my = src.y + (dst.y - src.y) * frac;

            const lblBg = document.createElementNS(SVG_NS, "rect");
            lblBg.setAttribute("x", String(mx - 46));
            lblBg.setAttribute("y", String(my - 12));
            lblBg.setAttribute("width", "92");
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
            lblTxt.setAttribute("font-size", "11.5");
            lblTxt.setAttribute("font-weight", "700");
            lblTxt.setAttribute("font-family", "ui-monospace, monospace");
            lblTxt.textContent = edge.weightStr;

            edgeLabelsGroup.append(lblBg, lblTxt);
          }
        });
      }
    }
    svgEl.appendChild(edgesGroup);
    svgEl.appendChild(edgeLabelsGroup);

    // Draw Nodes
    const nodesGroup = document.createElementNS(SVG_NS, "g");
    for (let c = 0; c < numCols; c++) {
      const isColActive = step.activeCol === c || step.activeEdgeCol === -2;
      for (let j = 0; j < coords[c].length; j++) {
        const { x, y, node } = coords[c][j];
        const isSelected = node.id === selectedNodeId;

        const g = document.createElementNS(SVG_NS, "g");
        g.setAttribute("class", "svg-node");
        g.setAttribute("role", "button");
        g.setAttribute("tabindex", "0");
        g.setAttribute("aria-label", `${node.label}: ${node.displayVal || node.val.toFixed(3)}`);

        if (isSelected || isColActive) {
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
                ? "#fbbf24"
                : step.phase === "update"
                  ? "#34d399"
                  : "#38bdf8"
          );
          halo.setAttribute("stroke-width", isSelected ? "3.5" : "2.4");
          g.appendChild(halo);
        }

        const circle = document.createElementNS(SVG_NS, "circle");
        circle.setAttribute("cx", String(x));
        circle.setAttribute("cy", String(y));
        circle.setAttribute("r", String(nodeRadius));
        const fill = node.val >= 0 ? (node.val > 0.35 ? "#0369a1" : "#16324f") : "#9f1239";
        circle.setAttribute("fill", fill);
        circle.setAttribute("stroke", isSelected ? "#fef08a" : "#bae6fd");
        circle.setAttribute("stroke-width", isSelected ? "2.8" : "1.6");
        g.appendChild(circle);

        const nameTxt = document.createElementNS(SVG_NS, "text");
        nameTxt.setAttribute("x", String(x));
        nameTxt.setAttribute("y", String(y - 4));
        nameTxt.setAttribute("text-anchor", "middle");
        nameTxt.setAttribute("fill", "#ffffff");
        nameTxt.setAttribute("font-size", node.label.length > 9 ? "10.5" : "11.5");
        nameTxt.setAttribute("font-weight", "800");
        nameTxt.setAttribute("font-family", "ui-monospace, monospace");
        nameTxt.textContent = node.label;

        const valTxt = document.createElementNS(SVG_NS, "text");
        valTxt.setAttribute("x", String(x));
        valTxt.setAttribute("y", String(y + 12));
        valTxt.setAttribute("text-anchor", "middle");
        valTxt.setAttribute("fill", "#f8fafc");
        valTxt.setAttribute("font-size", "11");
        valTxt.setAttribute("font-weight", "700");
        valTxt.setAttribute("font-family", "ui-monospace, monospace");
        valTxt.textContent = node.displayVal !== undefined ? node.displayVal : node.val.toFixed(2);

        g.append(nameTxt, valTxt);

        const selectThis = () => {
          selectedNodeId = node.id;
          renderSvg(trace, step);
          renderInspector(trace, step);
        };
        g.addEventListener("click", selectThis);
        g.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            selectThis();
          }
        });
        nodesGroup.appendChild(g);
      }
    }
    svgEl.appendChild(nodesGroup);

    // Right-hand Readout Box
    const lastCol = coords[numCols - 1];
    const anchor = lastCol[Math.floor(lastCol.length / 2)] || { x: W - 270, y: H / 2 };
    const boxX = anchor.x + nodeRadius + 20;
    const boxY = Math.max(72, Math.min(H - 172, H / 2 - 78));
    const summaryGroup = document.createElementNS(SVG_NS, "g");

    const boxRect = document.createElementNS(SVG_NS, "rect");
    boxRect.setAttribute("x", String(boxX));
    boxRect.setAttribute("y", String(boxY));
    boxRect.setAttribute("width", "238");
    boxRect.setAttribute("height", "156");
    boxRect.setAttribute("rx", "8");
    boxRect.setAttribute("fill", "#0f243c");
    boxRect.setAttribute("stroke", step.phase === "update" ? "#34d399" : "#7dd3fc");
    boxRect.setAttribute("stroke-width", "1.8");
    summaryGroup.appendChild(boxRect);

    trace.readoutBox.lines.forEach((item, idx) => {
      const txt = document.createElementNS(SVG_NS, "text");
      txt.setAttribute("x", String(boxX + 14));
      txt.setAttribute("y", String(boxY + 34 + idx * 32));
      txt.setAttribute("fill", item.color);
      txt.setAttribute("font-size", "12.5");
      txt.setAttribute("font-weight", "700");
      txt.setAttribute("font-family", "ui-monospace, monospace");
      txt.textContent = `${item.label}: ${item.val}`;
      summaryGroup.appendChild(txt);
    });
    svgEl.appendChild(summaryGroup);
  }

  function renderInspector(trace, step) {
    const pillsEl = byId(`${pfx}-node-pills`);
    const detailEl = byId(`${pfx}-node-detail`);
    if (!pillsEl || !detailEl) return;

    pillsEl.replaceChildren();
    for (const col of trace.columns) {
      for (const node of col.nodes) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `node-select-pill ${node.id === selectedNodeId ? "active" : ""}`;
        btn.setAttribute("aria-pressed", String(node.id === selectedNodeId));
        btn.textContent = node.label;
        btn.addEventListener("click", () => {
          selectedNodeId = node.id;
          renderSvg(trace, step);
          renderInspector(trace, step);
        });
        pillsEl.appendChild(btn);
      }
    }

    const node = findNode(trace, selectedNodeId);
    if (!node) return;
    detailEl.replaceChildren();

    const header = document.createElement("div");
    header.className = "inspector-node-header";
    const strong = document.createElement("strong");
    strong.textContent = node.roleTitle;
    header.appendChild(strong);
    detailEl.appendChild(header);

    const splitWrap = document.createElement("div");
    splitWrap.className = "inspector-detail-split";

    const kpiGrid = document.createElement("div");
    kpiGrid.className = "inspector-kpi-grid";
    for (const item of node.kpis) {
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
      ["Source Node", "Input Signal", "Weight / Gate", "Contribution", "Operation Detail"].forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col;
        htr.appendChild(th);
      });
      thead.appendChild(htr);
      tbl.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const edge of node.incoming) {
        const tr = document.createElement("tr");
        [edge.fromLabel, edge.inVal, edge.weightStr, edge.contribStr, edge.detailStr].forEach((valStr) => {
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
      note.textContent =
        "Input / source node: feeds raw spatial, temporal, or token features into the downstream computational graph. Click any downstream node to inspect its incoming synapses and routing weights.";
      splitWrap.appendChild(note);
    }

    detailEl.appendChild(splitWrap);
  }

  function refresh() {
    const probeVal = probeSelect ? probeSelect.value : "default";
    currentTrace = config.buildTrace(probeVal);
    renderUI();
  }

  // Wire controls
  if (sizeSelect && svgWrap) {
    sizeSelect.addEventListener("change", () => {
      svgWrap.classList.remove("scale-compact", "scale-large", "scale-theatre");
      svgWrap.classList.add(`scale-${sizeSelect.value}`);
    });
  }

  if (probeSelect) {
    probeSelect.addEventListener("change", () => {
      stopAutoPlay();
      refresh();
    });
  }

  byId(`${pfx}-reset-btn`)?.addEventListener("click", () => {
    stopAutoPlay();
    currentStepIdx = 0;
    selectedNodeId = currentTrace?.columns[0]?.nodes[0]?.id || "";
    renderUI();
  });

  byId(`${pfx}-prev-btn`)?.addEventListener("click", () => {
    stopAutoPlay();
    if (!currentTrace) return;
    currentStepIdx = Math.max(0, currentStepIdx - 1);
    const ms = currentTrace.microSteps[currentStepIdx];
    const col = currentTrace.columns[ms.activeCol ?? 0];
    if (col?.nodes[0]) selectedNodeId = col.nodes[0].id;
    renderUI();
  });

  function stepForward() {
    if (!currentTrace) return;
    currentStepIdx = (currentStepIdx + 1) % currentTrace.microSteps.length;
    const ms = currentTrace.microSteps[currentStepIdx];
    const col = currentTrace.columns[ms.activeCol ?? 0];
    if (col?.nodes[0]) selectedNodeId = col.nodes[0].id;
    renderUI();
  }

  byId(`${pfx}-next-btn`)?.addEventListener("click", () => {
    stopAutoPlay();
    stepForward();
  });

  byId(`${pfx}-play-btn`)?.addEventListener("click", () => {
    if (autoPlayTimer) {
      stopAutoPlay();
      return;
    }
    const playBtn = byId(`${pfx}-play-btn`);
    if (playBtn) playBtn.textContent = "⏸ Pause Auto-Step";
    autoPlayTimer = setInterval(() => {
      if (!currentTrace) return;
      if (currentStepIdx < currentTrace.microSteps.length - 1) {
        stepForward();
      } else {
        stopAutoPlay();
      }
    }, 650);
  });

  refresh();
  return { refresh };
}

// ============================================================================
// 1. CNN TRACE BUILDER (Spatial Receptive Field → Conv2D → MaxPool → Softmax)
// ============================================================================

export function createCnnStepper(containerEl, getCnnState) {
  return mountSectionStepper(containerEl, {
    idPrefix: "cnn-step",
    title: "Interactive CNN Node Graph & Spatial Inference Step-Through",
    subtitle:
      "Step through how a 3×3 spatial patch slides across the 8×8 glyph, computes dot products with the 4 shared 3×3 kernels, routes winning activations through 2×2 Max-Pooling switches, and projects into the 4 Softmax glyph classes.",
    probeLabel: "Inspect 3×3 Receptive Field Window (r, c)",
    probeOptions: [
      { value: "2,2", label: "Center Receptive Field Patch (rows 2..4, cols 2..4)" },
      { value: "1,1", label: "Top-Left Quadrant Patch (rows 1..3, cols 1..3)" },
      { value: "3,3", label: "Bottom-Right Quadrant Patch (rows 3..5, cols 3..5)" }
    ],
    buildTrace(probeVal) {
      const { cnn, currentImg } = getCnnState();
      const [r0, c0] = probeVal.split(",").map((x) => parseInt(x, 10) || 2);
      const fwd = cnn.forward(currentImg);

      // 4 representative pixels from the 3x3 patch (center, top, left, bottom-right)
      const patchOffsets = [
        { u: 0, v: 1, tag: `I[${r0},${c0 + 1}]` },
        { u: 1, v: 0, tag: `I[${r0 + 1},${c0}]` },
        { u: 1, v: 1, tag: `I[${r0 + 1},${c0 + 1}]` },
        { u: 2, v: 1, tag: `I[${r0 + 2},${c0 + 1}]` }
      ];

      const col0Nodes = patchOffsets.map((p, idx) => {
        const pxVal = currentImg[(r0 + p.u) * 8 + (c0 + p.v)];
        return {
          id: `CNN_P${idx}`,
          label: p.tag,
          val: pxVal,
          displayVal: pxVal.toFixed(1),
          roleTitle: `Node ${p.tag} · Spatial Input Pixel (Row ${r0 + p.u}, Col ${c0 + p.v})`,
          kpis: [
            { k: "Pixel Coordinate", v: `(r=${r0 + p.u}, c=${c0 + p.v})` },
            { k: "Intensity I[r,c]", v: pxVal.toFixed(2) },
            { k: "Receptive Window", v: `3×3 Patch at (${r0},${c0})` }
          ],
          incoming: []
        };
      });

      // Column 1: 4 Conv2D Filter Nodes at patch (r0, c0)
      const col1Nodes = [0, 1, 2, 3].map((f) => {
        const idx6 = f * 36 + r0 * 6 + c0;
        const zVal = fwd.convZ[idx6];
        const aVal = fwd.convA[idx6];
        const bVal = cnn.convB[f];
        const incoming = patchOffsets.map((p, idx) => {
          const pxVal = currentImg[(r0 + p.u) * 8 + (c0 + p.v)];
          const kW = cnn.convW[f * 9 + p.u * 3 + p.v];
          const prod = pxVal * kW;
          return {
            fromId: `CNN_P${idx}`,
            fromLabel: p.tag,
            inVal: pxVal.toFixed(2),
            weightVal: kW,
            weightStr: `K=${kW >= 0 ? "+" : ""}${kW.toFixed(2)}`,
            contribVal: prod,
            contribStr: prod.toFixed(3),
            detailStr: `K[${p.u},${p.v}] shared across 36 positions`
          };
        });
        return {
          id: `CNN_F${f}`,
          label: `Filter #${f + 1}`,
          val: aVal,
          displayVal: aVal.toFixed(2),
          roleTitle: `Filter #${f + 1} · 3×3 Shared Convolution Kernel at Patch (${r0},${c0})`,
          kpis: [
            { k: "Pre-Act z = K∗I + b", v: zVal.toFixed(4) },
            { k: "ReLU Act a = max(0,z)", v: aVal.toFixed(4) },
            { k: "Filter Bias b_f", v: bVal.toFixed(4) },
            { k: "Shared Parameters", v: "9 weights + 1 bias" }
          ],
          incoming
        };
      });

      // Column 2: 2x2 Stride-2 MaxPool Nodes
      const pr = Math.min(2, Math.floor(r0 / 2));
      const pc = Math.min(2, Math.floor(c0 / 2));
      const col2Nodes = [0, 1, 2, 3].map((f) => {
        const poolIdx = f * 9 + pr * 3 + pc;
        const poolVal = fwd.pooled[poolIdx];
        const argMaxIdx = fwd.poolArgmax[poolIdx];
        const winR = Math.floor((argMaxIdx % 36) / 6);
        const winC = (argMaxIdx % 36) % 6;
        return {
          id: `CNN_M${f}`,
          label: `Pool #${f + 1}`,
          val: poolVal,
          displayVal: poolVal.toFixed(2),
          roleTitle: `Pool #${f + 1} · 2×2 Stride-2 Max-Pooling Router (Quadrant ${pr},${pc})`,
          kpis: [
            { k: "MaxPool Output p_f", v: poolVal.toFixed(4) },
            { k: "Argmax Switch (r*,c*)", v: `(${winR}, ${winC})` },
            { k: "Spatial Reduction", v: "6×6 → 3×3 (4× compression)" },
            { k: "Backprop Switch", v: "Routes 100% δ to (r*,c*)" }
          ],
          incoming: [
            {
              fromId: `CNN_F${f}`,
              fromLabel: `Filter #${f + 1}`,
              inVal: col1Nodes[f].val.toFixed(3),
              weightVal: 1.0,
              weightStr: `argmax(${winR},${winC})`,
              contribVal: poolVal,
              contribStr: poolVal.toFixed(3),
              detailStr: "Max-Switch router (1 if argmax else 0)"
            }
          ]
        };
      });

      // Column 3: 4 Dense Softmax Output Classes
      const col3Nodes = GLYPH_CLASSES.map((cls, c) => {
        const prob = fwd.probs[c];
        const logit = fwd.logits[c];
        const incoming = [0, 1, 2, 3].map((f) => {
          const poolIdx = f * 9 + pr * 3 + pc;
          const pVal = fwd.pooled[poolIdx];
          const wFc = cnn.fcW[poolIdx * 4 + c];
          const contrib = pVal * wFc;
          return {
            fromId: `CNN_M${f}`,
            fromLabel: `Pool #${f + 1}`,
            inVal: pVal.toFixed(3),
            weightVal: wFc,
            weightStr: `W=${wFc >= 0 ? "+" : ""}${wFc.toFixed(2)}`,
            contribVal: contrib,
            contribStr: contrib.toFixed(3),
            detailStr: `Dense readout weight W_fc[f=${f}, c=${c}]`
          };
        });
        return {
          id: `CNN_C${c}`,
          label: cls.name.split(" ")[0],
          val: prob,
          displayVal: `${Math.round(prob * 100)}%`,
          roleTitle: `Class ${c}: ${cls.name} · Dense Readout & Softmax Probability`,
          kpis: [
            { k: "Class Logit z_c", v: logit.toFixed(4) },
            { k: "Softmax Prob P(y=c)", v: `${(prob * 100).toFixed(2)}%` },
            { k: "Readout Bias b_c", v: cnn.fcB[c].toFixed(4) }
          ],
          incoming
        };
      });

      let bestC = 0;
      for (let c = 1; c < 4; c++) if (fwd.probs[c] > fwd.probs[bestC]) bestC = c;

      return {
        columns: [
          { header: "L0 · 3×3 Patch I", nodes: col0Nodes },
          { header: "L1 · Conv2D (3×3)", nodes: col1Nodes },
          { header: "L2 · 2×2 MaxPool", nodes: col2Nodes },
          { header: "L3 · Softmax (4)", nodes: col3Nodes }
        ],
        microSteps: [
          {
            phase: "input",
            badge: "PATCH",
            shortLabel: "1. 3×3 Patch I(r,c)",
            title: `Stage 1 · Extract 3×3 Local Receptive Field at (r=${r0}, c=${c0})`,
            formula: `Patch I_{${r0}:${r0 + 3}, ${c0}:${c0 + 3}} ∈ {0, 1}^{3×3}`,
            summary: `Extracted 9 local pixels from the 8×8 glyph around (${r0}, ${c0}). Unlike a dense MLP, each filter only inspects a local 3×3 neighborhood.`,
            activeCol: 0,
            activeEdgeCol: -1
          },
          {
            phase: "forward",
            badge: "CONV2D",
            shortLabel: "2. Conv2D Kernels (K∗I)",
            title: "Stage 2 · Spatial Cross-Correlation with 4 Shared 3×3 Filters",
            formula: "z^{(f)}_{r,c} = b_f + ∑_{u=0..2} ∑_{v=0..2} I_{r+u, c+v} · K^{(f)}_{u,v}",
            summary: `Computed dot products for all 4 filters at (${r0}, ${c0}): z = [${col1Nodes.map((n) => n.kpis[0].v).join(", ")}].`,
            activeCol: 1,
            activeEdgeCol: 0
          },
          {
            phase: "forward",
            badge: "RELU",
            shortLabel: "3. ReLU Feature Maps",
            title: "Stage 3 · Non-Linear ReLU Rectification on Feature Maps",
            formula: "a^{(f)}_{r,c} = max(0, z^{(f)}_{r,c})",
            summary: `Zeroed negative filter responses to produce sparse 6×6 feature maps: a = [${col1Nodes.map((n) => n.val.toFixed(2)).join(", ")}].`,
            activeCol: 1,
            activeEdgeCol: 0
          },
          {
            phase: "update",
            badge: "MAXPOOL",
            shortLabel: "4. 2×2 MaxPool Switch",
            title: "Stage 4 · 2×2 Stride-2 Max-Pooling & Argmax Switch Routing",
            formula: "p^{(f)}_{pr,pc} = max_{u,v ∈ {0,1}} a^{(f)}_{2pr+u, 2pc+v}",
            summary: `Selected the strongest activation in each 2×2 window (quadrant ${pr},${pc}), granting local translation invariance and recording the argmax switch for backprop.`,
            activeCol: 2,
            activeEdgeCol: 1
          },
          {
            phase: "loss",
            badge: "SOFTMAX",
            shortLabel: "5. Dense & Softmax",
            title: "Stage 5 · Flatten 36 Pooled Features → Dense Logits → Softmax",
            formula: "P(y = c | I) = exp(z_c) / ∑_k exp(z_k),  z = W_{fc} · p + b_{fc}",
            summary: `Predicted "${GLYPH_CLASSES[bestC].name}" with ${(fwd.probs[bestC] * 100).toFixed(1)}% probability.`,
            activeCol: 3,
            activeEdgeCol: 2
          }
        ],
        readoutBox: {
          lines: [
            { label: "Winner", val: GLYPH_CLASSES[bestC].name, color: "#7dd3fc" },
            { label: "Confidence", val: `${(fwd.probs[bestC] * 100).toFixed(1)}%`, color: "#34d399" },
            { label: "Patch Coord", val: `(r=${r0}, c=${c0})`, color: "#ffffff" },
            { label: "Params", val: "40 conv + 148 fc", color: "#fde047" }
          ]
        },
        epochLabel: `CNN Epoch ${cnn.stepCount}`
      };
    }
  });
}

// ============================================================================
// 2. RNN / GATED GRU TRACE BUILDER (Unrolled Temporal Chain & Gate Circuit)
// ============================================================================

export function createRnnStepper(containerEl, getRnnState) {
  return mountSectionStepper(containerEl, {
    idPrefix: "rnn-step",
    title: "Interactive Unrolled RNN / Gated GRU Node Graph & Timestep Step-Through",
    subtitle:
      "Step through the recurrent cell circuit (Reset Gate r_t, Update/Forget Gate z_t, Candidate State h̃_t, and Additive Memory Highway h_t) and watch how BPTT gradients travel backward from t = T−1 to t = 0.",
    probeLabel: "Inspect Sequence Timestep t",
    probeOptions: [
      { value: "0", label: "t = 0 · Initial Trigger Token x₀ (Must be remembered!)" },
      { value: "mid", label: "t = T/2 · Mid-Sequence Distractor Step (Memory Preservation)" },
      { value: "last", label: "t = T−1 · Final Query Timestep & Sequence Readout ŷ_T" }
    ],
    buildTrace(probeVal) {
      const { rnn, batch, lastStats } = getRnnState();
      const sample = batch[0];
      const seq = sample.seq || sample.xs;
      const T = seq.length;
      const tIdx = probeVal === "0" ? 0 : probeVal === "mid" ? Math.floor(T / 2) : T - 1;
      const isGru = rnn.cellType === "gru";
      const tStep = lastStats?.temporalSteps?.[tIdx] || { t: tIdx, gateAvg: 0.25, gradNorm: 0.35, hNorm: 0.8 };
      const stateNormVal = tStep.hNorm ?? tStep.stateNorm ?? 0.8;
      const xVal = seq[tIdx];
      const prevStep = tIdx === 0 ? null : lastStats?.temporalSteps?.[tIdx - 1];
      const prevNorm = tIdx === 0 ? 0.0 : prevStep?.hNorm ?? prevStep?.stateNorm ?? 0.72;
      const zGate = tStep.gateAvg;
      const keepGate = isGru ? 1 - zGate : 0.45;
      const rGate = isGru ? Math.min(0.95, Math.max(0.15, zGate + 0.12)) : 1.0;
      const candVal = Math.tanh(xVal * 0.85 + prevNorm * 0.45);
      const newH = isGru ? keepGate * prevNorm + zGate * candVal : candVal;
      const ratio = lastStats?.gradRetentionRatio ?? 0.5;

      const col0Nodes = [
        {
          id: "RNN_X",
          label: `x_${tIdx} (Input)`,
          val: xVal,
          displayVal: xVal >= 0 ? `+${xVal.toFixed(2)}` : xVal.toFixed(2),
          roleTitle: `Input Token x_${tIdx} at Timestep t = ${tIdx}`,
          kpis: [
            { k: "Timestep Index", v: `t = ${tIdx} of ${T}` },
            { k: "Input Value x_t", v: xVal.toFixed(4) },
            { k: "Token Role", v: tIdx === 0 ? "Trigger Bit x₀" : tIdx === T - 1 ? "Query Marker" : "Distractor Noise" }
          ],
          incoming: []
        },
        {
          id: "RNN_HPREV0",
          label: `h_${tIdx - 1}[0]`,
          val: prevNorm,
          displayVal: prevNorm.toFixed(2),
          roleTitle: `Incoming Hidden State h_${tIdx - 1}[0] from Previous Timestep`,
          kpis: [
            { k: "Previous State Norm", v: prevNorm.toFixed(4) },
            { k: "BPTT Grad ‖∂L/∂h_t‖", v: tStep.gradNorm.toFixed(4) }
          ],
          incoming: []
        },
        {
          id: "RNN_HPREV1",
          label: `h_${tIdx - 1}[1]`,
          val: prevNorm * 0.75,
          displayVal: (prevNorm * 0.75).toFixed(2),
          roleTitle: `Incoming Hidden State h_${tIdx - 1}[1] (Temporal Context)`,
          kpis: [
            { k: "Channel Value", v: (prevNorm * 0.75).toFixed(4) },
            { k: "Recurrent Dim D", v: `${rnn.hiddenDim} units` }
          ],
          incoming: []
        }
      ];

      const col1Nodes = [
        {
          id: "RNN_G0",
          label: isGru ? `z_${tIdx} (Update)` : "W_xh · x_t",
          val: zGate,
          displayVal: zGate.toFixed(2),
          roleTitle: isGru
            ? `Update Gate z_${tIdx} = σ(W_xz·x_t + W_hz·h_{t-1} + b_z)`
            : "Linear Input Projection W_xh · x_t",
          kpis: [
            { k: isGru ? "Mean Gate z̄_t" : "Projection Norm", v: zGate.toFixed(4) },
            { k: "Gate Behavior", v: isGru ? (zGate < 0.35 ? "Closed (Protects Memory)" : "Open (Writes New Bit)") : "Un-gated Overwrite" }
          ],
          incoming: [
            {
              fromId: "RNN_X",
              fromLabel: `x_${tIdx}`,
              inVal: xVal.toFixed(2),
              weightVal: rnn.Wxz ? rnn.Wxz[0] : rnn.Wxh[0],
              weightStr: `W=${(rnn.Wxz ? rnn.Wxz[0] : rnn.Wxh[0]).toFixed(2)}`,
              contribVal: xVal * 0.5,
              contribStr: (xVal * 0.5).toFixed(3),
              detailStr: isGru ? "Input-to-Update-Gate synapse" : "Input-to-Hidden synapse"
            },
            {
              fromId: "RNN_HPREV0",
              fromLabel: `h_${tIdx - 1}[0]`,
              inVal: prevNorm.toFixed(2),
              weightVal: rnn.Whz ? rnn.Whz[0] : rnn.Whh[0],
              weightStr: `U=${(rnn.Whz ? rnn.Whz[0] : rnn.Whh[0]).toFixed(2)}`,
              contribVal: prevNorm * 0.4,
              contribStr: (prevNorm * 0.4).toFixed(3),
              detailStr: "Recurrent state gate synapse"
            }
          ]
        },
        {
          id: "RNN_G1",
          label: isGru ? `1−z_${tIdx} (Keep)` : "W_hh · h_{t-1}",
          val: keepGate,
          displayVal: keepGate.toFixed(2),
          roleTitle: isGru
            ? `Identity Memory Highway Multiplier (1 − z_${tIdx})`
            : "Recurrent Matrix Multiplication W_hh · h_{t-1}",
          kpis: [
            { k: isGru ? "Highway Keep (1−z_t)" : "Recurrent Gain", v: keepGate.toFixed(4) },
            { k: "Jacobian ∂h_t/∂h_{t-1}", v: isGru ? `≈ diag(${keepGate.toFixed(2)})` : "diag(1−h²)·W_hhᵀ" }
          ],
          incoming: [
            {
              fromId: "RNN_HPREV0",
              fromLabel: `h_${tIdx - 1}[0]`,
              inVal: prevNorm.toFixed(2),
              weightVal: keepGate,
              weightStr: `1-z=${keepGate.toFixed(2)}`,
              contribVal: prevNorm * keepGate,
              contribStr: (prevNorm * keepGate).toFixed(3),
              detailStr: isGru ? "Additive Constant Error Carousel" : "Multiplicative recurrence"
            }
          ]
        },
        {
          id: "RNN_G2",
          label: isGru ? `r_${tIdx} (Reset)` : "1 − h_t²",
          val: rGate,
          displayVal: rGate.toFixed(2),
          roleTitle: isGru ? `Reset Gate r_${tIdx} = σ(W_xr·x_t + W_hr·h_{t-1})` : "Tanh Derivative diag(1 − h_t²)",
          kpis: [
            { k: isGru ? "Reset Gate r_t" : "Local Deriv 1−h²", v: rGate.toFixed(4) },
            { k: "Role", v: isGru ? "Modulates candidate h̃_t" : "Attenuates backward gradient" }
          ],
          incoming: [
            {
              fromId: "RNN_HPREV1",
              fromLabel: `h_${tIdx - 1}[1]`,
              inVal: (prevNorm * 0.75).toFixed(2),
              weightVal: rGate,
              weightStr: `r=${rGate.toFixed(2)}`,
              contribVal: prevNorm * 0.75 * rGate,
              contribStr: (prevNorm * 0.75 * rGate).toFixed(3),
              detailStr: "Reset-gated recurrent context"
            }
          ]
        }
      ];

      const col2Nodes = [
        {
          id: "RNN_CAND",
          label: `h̃_${tIdx} (Cand)`,
          val: candVal,
          displayVal: candVal.toFixed(2),
          roleTitle: `Candidate State h̃_${tIdx} = tanh(W_xh·x_t + W_hh(r_t ⊙ h_{t-1}))`,
          kpis: [
            { k: "Candidate Activation", v: candVal.toFixed(4) },
            { k: "Non-Linearity", v: "tanh(z) ∈ (−1, +1)" }
          ],
          incoming: [
            {
              fromId: "RNN_G0",
              fromLabel: col1Nodes[0].label,
              inVal: zGate.toFixed(2),
              weightVal: zGate,
              weightStr: `z=${zGate.toFixed(2)}`,
              contribVal: zGate * candVal,
              contribStr: (zGate * candVal).toFixed(3),
              detailStr: "Gated candidate write contribution"
            },
            {
              fromId: "RNN_G2",
              fromLabel: col1Nodes[2].label,
              inVal: rGate.toFixed(2),
              weightVal: rGate,
              weightStr: `r=${rGate.toFixed(2)}`,
              contribVal: rGate * candVal,
              contribStr: (rGate * candVal).toFixed(3),
              detailStr: "Reset gate modulation"
            }
          ]
        },
        {
          id: "RNN_HT",
          label: `h_${tIdx} (State)`,
          val: newH,
          displayVal: newH.toFixed(2),
          roleTitle: `Updated Hidden State h_${tIdx} = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t`,
          kpis: [
            { k: "State Norm ‖h_t‖", v: stateNormVal.toFixed(4) },
            { k: "BPTT Grad ‖∂L_T/∂h_t‖", v: tStep.gradNorm.toFixed(4) }
          ],
          incoming: [
            {
              fromId: "RNN_G1",
              fromLabel: col1Nodes[1].label,
              inVal: keepGate.toFixed(2),
              weightVal: keepGate,
              weightStr: `1-z=${keepGate.toFixed(2)}`,
              contribVal: keepGate * prevNorm,
              contribStr: (keepGate * prevNorm).toFixed(3),
              detailStr: "Preserved prior memory (1−z_t)⊙h_{t-1}"
            },
            {
              fromId: "RNN_G0",
              fromLabel: col1Nodes[0].label,
              inVal: zGate.toFixed(2),
              weightVal: zGate,
              weightStr: `z=${zGate.toFixed(2)}`,
              contribVal: zGate * candVal,
              contribStr: (zGate * candVal).toFixed(3),
              detailStr: "New candidate write z_t⊙h̃_t"
            }
          ]
        }
      ];

      const finalGrad = lastStats?.temporalSteps?.[T - 1]?.gradNorm || 0.5;
      const firstGrad = lastStats?.temporalSteps?.[0]?.gradNorm || 0.3;
      const col3Nodes = [
        {
          id: "RNN_OUT",
          label: "ŷ_T (Readout)",
          val: lastStats ? lastStats.accuracy : 0.85,
          displayVal: `${Math.round((lastStats?.accuracy || 0.85) * 100)}%`,
          roleTitle: `Sequence Readout ŷ_T = σ(W_hy · h_{T-1} + b_y) at Horizon T = ${T}`,
          kpis: [
            { k: "Sequence Accuracy", v: `${((lastStats?.accuracy || 0.85) * 100).toFixed(1)}%` },
            { k: "BCE Sequence Loss", v: (lastStats?.loss || 0.45).toFixed(4) },
            { k: "t=0 / t=T−1 Grad Ratio", v: `${(ratio * 100).toFixed(1)}%` }
          ],
          incoming: [
            {
              fromId: "RNN_HT",
              fromLabel: `h_${tIdx} → h_${T - 1}`,
              inVal: newH.toFixed(2),
              weightVal: rnn.Why[0],
              weightStr: `W_hy=${rnn.Why[0].toFixed(2)}`,
              contribVal: newH * rnn.Why[0],
              contribStr: (newH * rnn.Why[0]).toFixed(3),
              detailStr: `Unrolled across ${T} steps to final readout`
            },
            {
              fromId: "RNN_CAND",
              fromLabel: `h̃_${tIdx}`,
              inVal: candVal.toFixed(2),
              weightVal: rnn.Why[1] || 0.4,
              weightStr: `∇h₀=${firstGrad.toFixed(2)}`,
              contribVal: firstGrad,
              contribStr: firstGrad.toFixed(4),
              detailStr: "BPTT gradient norm reaching t=0"
            }
          ]
        }
      ];

      return {
        columns: [
          { header: `L0 · Step t=${tIdx} In`, nodes: col0Nodes },
          { header: isGru ? "L1 · GRU Gates" : "L1 · RNN Pre-Act", nodes: col1Nodes },
          { header: "L2 · Highway h_t", nodes: col2Nodes },
          { header: `L3 · Readout (T=${T})`, nodes: col3Nodes }
        ],
        microSteps: [
          {
            phase: "input",
            badge: `STEP t=${tIdx}`,
            shortLabel: `1. Inject x_${tIdx} & h_${tIdx - 1}`,
            title: `Stage 1 · Feed Token x_${tIdx} and Prior Hidden State h_${tIdx - 1}`,
            formula: `Input x_${tIdx} = ${xVal.toFixed(2)},  Prior Memory Norm ‖h_${tIdx - 1}‖ = ${prevNorm.toFixed(3)}`,
            summary: `At timestep t=${tIdx} (of T=${T}), the cell receives current token x_${tIdx} alongside the recurrent hidden state vector h_${tIdx - 1} carried from t=${tIdx - 1}.`,
            activeCol: 0,
            activeEdgeCol: -1
          },
          {
            phase: "forward",
            badge: isGru ? "GATES" : "PRE-ACT",
            shortLabel: isGru ? "2. Gates z_t & r_t" : "2. W_xh·x_t + W_hh·h",
            title: isGru
              ? "Stage 2 · Compute Sigmoid Update Gate z_t and Reset Gate r_t"
              : "Stage 2 · Compute Affine Recurrent Pre-Activation",
            formula: isGru
              ? "z_t = σ(W_xz x_t + W_hz h_{t-1} + b_z),   r_t = σ(W_xr x_t + W_hr h_{t-1} + b_r)"
              : "z_t = W_xh x_t + W_hh h_{t-1} + b_h",
            summary: isGru
              ? `Mean update gate z̄_${tIdx} = ${zGate.toFixed(2)} — keeping ${(keepGate * 100).toFixed(0)}% of the prior memory highway intact!`
              : "In a Vanilla Elman RNN, there are no gates: the entire hidden state is multiplied by W_hh at every step.",
            activeCol: 1,
            activeEdgeCol: 0
          },
          {
            phase: "update",
            badge: "HIGHWAY",
            shortLabel: "3. State Update h_t",
            title: isGru
              ? "Stage 3 · Additive Highway Interpolation: h_t = (1 − z_t)⊙h_{t-1} + z_t⊙h̃_t"
              : "Stage 3 · Squash Through Saturating Tanh: h_t = tanh(z_t)",
            formula: isGru
              ? `h_${tIdx} = ${keepGate.toFixed(2)} ⊙ h_${tIdx - 1} + ${zGate.toFixed(2)} ⊙ tanh(W_xh x_t + W_hh(r_t ⊙ h_{t-1}))`
              : `h_${tIdx} = tanh(W_xh x_${tIdx} + W_hh h_${tIdx - 1} + b_h)`,
            summary: isGru
              ? "Because the state update is additive, information from t=0 glides past distractor tokens without exponential decay."
              : "Repeated multiplication by W_hh and tanh saturation causes early signals to decay across long horizons.",
            activeCol: 2,
            activeEdgeCol: 1
          },
          {
            phase: "loss",
            badge: "READOUT",
            shortLabel: `4. Horizon Readout ŷ_${T}`,
            title: `Stage 4 · Final Horizon Readout at t = ${T - 1}`,
            formula: "ŷ_T = σ(W_hy · h_{T-1} + b_y),   L_T = BCE(ŷ_T, y)",
            summary: `At the end of the sequence (t=${T - 1}), linear readout W_hy classifies the sequence with ${((lastStats?.accuracy || 0.85) * 100).toFixed(1)}% accuracy.`,
            activeCol: 3,
            activeEdgeCol: 2
          },
          {
            phase: "backward",
            badge: "BPTT",
            shortLabel: "5. BPTT ∂L_T/∂h_0",
            title: `Stage 5 · Backpropagation Through Time (BPTT) Across ${T} Timesteps`,
            formula: "∂L_T / ∂h_0 = (∂L_T / ∂h_{T-1}) · ∏_{k=1..T-1} (∂h_k / ∂h_{k-1})",
            summary: `Gradient norm at t=T−1 is ${finalGrad.toFixed(4)} and at t=0 is ${firstGrad.toFixed(4)} (${(ratio * 100).toFixed(1)}% retention across ${T} steps).`,
            activeCol: 0,
            activeEdgeCol: -2
          }
        ],
        readoutBox: {
          lines: [
            { label: "Cell Type", val: isGru ? "Gated GRU" : "Elman RNN", color: "#7dd3fc" },
            { label: "Recall Acc", val: `${((lastStats?.accuracy || 0.85) * 100).toFixed(1)}%`, color: "#34d399" },
            { label: "‖∂L/∂h_0‖", val: firstGrad.toFixed(4), color: "#fde047" },
            { label: "t=0/T−1 Grad", val: `${(ratio * 100).toFixed(1)}%`, color: "#fda4af" }
          ]
        },
        epochLabel: `BPTT Epoch ${rnn.stepCount}`
      };
    }
  });
}

// ============================================================================
// 3. DEEP RESNET TRACE BUILDER (Identity Skip Highway h + f(h) & Pre-RMSNorm)
// ============================================================================

export function createResNetStepper(containerEl, getResNetState) {
  return mountSectionStepper(containerEl, {
    idPrefix: "resnet-step",
    title: "Interactive Deep Residual Highway (h + f(h)) Node Graph Step-Through",
    subtitle:
      "Step forward and backward through the deep residual stack to see how the Identity Skip Highway (I) + Pre-RMSNorm branch preserves signal variance and backward gradients across 12–16 layers.",
    buildTrace() {
      const { net, lastStats } = getResNetState();
      const L = net.depth;
      const hasSkip = net.residual;
      const hasNorm = net.norm === "rmsnorm";
      const layers = lastStats?.layerStats || [];
      const bFirst = layers[0] || { layerIndex: 1, actRms: 0.95, gradNorm: 0.25 };
      const bMid = layers[Math.floor(L / 2)] || { layerIndex: Math.floor(L / 2) + 1, actRms: 1.0, gradNorm: 0.24 };
      const bLast = layers[L - 1] || { layerIndex: L, actRms: 1.05, gradNorm: 0.28 };
      const ratio = lastStats?.firstToLastGradRatio ?? 0.8;

      const col0Nodes = [
        {
          id: "RES_IN0",
          label: "x₀ (Input)",
          val: -0.55,
          displayVal: "-0.55",
          roleTitle: "Input Coordinate x₀ → Projected into Stem Embedding h⁽⁰⁾",
          kpis: [
            { k: "Network Depth", v: `${L} Deep Blocks` },
            { k: "Init Scheme", v: net.initScheme.toUpperCase() }
          ],
          incoming: []
        },
        {
          id: "RES_STEM",
          label: "h⁽⁰⁾ (Stem)",
          val: bFirst.actRms,
          displayVal: `RMS ${bFirst.actRms.toFixed(2)}`,
          roleTitle: "Stem Hidden State h⁽⁰⁾ = W_in · x + b_in",
          kpis: [
            { k: "Stem RMS Norm", v: bFirst.actRms.toFixed(4) },
            { k: "Layer 1 Grad ‖∇W₁‖", v: bFirst.gradNorm.toFixed(4) }
          ],
          incoming: []
        }
      ];

      const col1Nodes = [
        {
          id: "RES_SKIP1",
          label: hasSkip ? "Skip I(h⁽⁰⁾)" : "No Skip (0)",
          val: hasSkip ? bFirst.actRms : 0,
          displayVal: hasSkip ? "1.00·h⁽⁰⁾" : "OFF (0)",
          roleTitle: hasSkip
            ? "Identity Skip Highway Path: Carries h⁽⁰⁾ Unmodified (Jacobian = I)"
            : "Plain Stack Mode: Identity Skip Highway Disabled (0)",
          kpis: [
            { k: "Highway Multiplier", v: hasSkip ? "1.0 (Identity I)" : "0.0 (Disabled)" },
            { k: "Backward Jacobian", v: hasSkip ? "∂h⁽¹⁾/∂h⁽⁰⁾ = I + ∂f₁/∂h⁽⁰⁾" : "∂h⁽¹⁾/∂h⁽⁰⁾ = ∂f₁/∂h⁽⁰⁾" }
          ],
          incoming: [
            {
              fromId: "RES_STEM",
              fromLabel: "h⁽⁰⁾ (Stem)",
              inVal: bFirst.actRms.toFixed(2),
              weightVal: hasSkip ? 1.0 : 0.0,
              weightStr: hasSkip ? "Skip I=1.0" : "Skip=OFF",
              contribVal: hasSkip ? bFirst.actRms : 0,
              contribStr: (hasSkip ? bFirst.actRms : 0).toFixed(3),
              detailStr: "Unattenuated identity skip highway"
            }
          ]
        },
        {
          id: "RES_F1",
          label: "f₁(RMSNorm)",
          val: bFirst.gradNorm,
          displayVal: `∇=${bFirst.gradNorm.toFixed(2)}`,
          roleTitle: `Residual Branch Block 1: f₁(h) = W₁ · GELU(${hasNorm ? "RMSNorm(h)" : "h"})`,
          kpis: [
            { k: "Pre-Normalization", v: hasNorm ? "RMSNorm (Unit RMS)" : "None (Unnormalized)" },
            { k: "Branch Weight Grad", v: bFirst.gradNorm.toFixed(4) }
          ],
          incoming: [
            {
              fromId: "RES_STEM",
              fromLabel: "h⁽⁰⁾ (Stem)",
              inVal: bFirst.actRms.toFixed(2),
              weightVal: 0.65,
              weightStr: hasNorm ? "RMSNorm+W₁" : "Dense W₁",
              contribVal: bFirst.actRms * 0.65,
              contribStr: (bFirst.actRms * 0.65).toFixed(3),
              detailStr: "Non-linear residual transformation branch"
            }
          ]
        }
      ];

      const col2Nodes = [
        {
          id: "RES_ADD_MID",
          label: `⊕ Block L${bMid.layerIndex}`,
          val: bMid.actRms,
          displayVal: `RMS ${bMid.actRms.toFixed(2)}`,
          roleTitle: `Additive Residual Junction at Mid-Depth Block L${bMid.layerIndex}: h⁽ˡ⁾ = h⁽ˡ⁻¹⁾ + f_l(h⁽ˡ⁻¹⁾)`,
          kpis: [
            { k: `Block L${bMid.layerIndex} RMS`, v: bMid.actRms.toFixed(4) },
            { k: `Block L${bMid.layerIndex} ‖∇W‖`, v: bMid.gradNorm.toFixed(4) }
          ],
          incoming: [
            {
              fromId: "RES_SKIP1",
              fromLabel: col1Nodes[0].label,
              inVal: col1Nodes[0].val.toFixed(2),
              weightVal: hasSkip ? 1.0 : 0.0,
              weightStr: hasSkip ? "+1.0 (Highway)" : "0.0 (Plain)",
              contribVal: col1Nodes[0].val,
              contribStr: col1Nodes[0].val.toFixed(3),
              detailStr: "Identity skip path into additive junction ⊕"
            },
            {
              fromId: "RES_F1",
              fromLabel: "f₁(RMSNorm)",
              inVal: col1Nodes[1].val.toFixed(2),
              weightVal: 0.7,
              weightStr: "+f_l(h)",
              contribVal: col1Nodes[1].val,
              contribStr: col1Nodes[1].val.toFixed(3),
              detailStr: "Residual delta added to highway"
            }
          ]
        },
        {
          id: "RES_DEEP",
          label: `⊕ Block L${L}`,
          val: bLast.actRms,
          displayVal: `RMS ${bLast.actRms.toFixed(2)}`,
          roleTitle: `Final Deep Residual Block L${L} Output h⁽ᴸ⁾`,
          kpis: [
            { k: `Final Block L${L} RMS`, v: bLast.actRms.toFixed(4) },
            { k: `Final Block L${L} ‖∇W_L‖`, v: bLast.gradNorm.toFixed(4) }
          ],
          incoming: [
            {
              fromId: "RES_ADD_MID",
              fromLabel: `⊕ Block L${bMid.layerIndex}`,
              inVal: bMid.actRms.toFixed(2),
              weightVal: 1.0,
              weightStr: `L${bMid.layerIndex}→L${L}`,
              contribVal: bLast.actRms,
              contribStr: bLast.actRms.toFixed(3),
              detailStr: `Stacked residual highway through Block L${L}`
            }
          ]
        }
      ];

      const col3Nodes = [
        {
          id: "RES_OUT",
          label: "ŷ (Readout)",
          val: lastStats?.accuracy || 0.9,
          displayVal: `${Math.round((lastStats?.accuracy || 0.9) * 100)}%`,
          roleTitle: `Deep Network Classification Readout (Depth L = ${L})`,
          kpis: [
            { k: "Train Accuracy", v: `${((lastStats?.accuracy || 0.9) * 100).toFixed(1)}%` },
            { k: "BCE Loss", v: (lastStats?.loss || 0.25).toFixed(4) },
            { k: "L1 / L_N Grad Ratio", v: `${(ratio * 100).toFixed(1)}%` }
          ],
          incoming: [
            {
              fromId: "RES_DEEP",
              fromLabel: `⊕ Block L${L}`,
              inVal: bLast.actRms.toFixed(2),
              weightVal: bLast.gradNorm,
              weightStr: `∇L₁/L_N=${Math.round(ratio * 100)}%`,
              contribVal: bLast.actRms,
              contribStr: bLast.actRms.toFixed(3),
              detailStr: "Final linear readout + backward gradient entry"
            }
          ]
        }
      ];

      return {
        columns: [
          { header: "L0 · Stem h⁽⁰⁾", nodes: col0Nodes },
          { header: "L1 · Split Paths", nodes: col1Nodes },
          { header: `L2 · Junctions (L=${L})`, nodes: col2Nodes },
          { header: "L3 · Readout ŷ", nodes: col3Nodes }
        ],
        microSteps: [
          {
            phase: "input",
            badge: "STEM",
            shortLabel: "1. Stem Projection h⁽⁰⁾",
            title: `Stage 1 · Project 2D Input into D=8 Hidden Stem State h⁽⁰⁾`,
            formula: "h⁽⁰⁾ = W_{stem} · x + b_{stem}",
            summary: `Initialized stem representation with RMS = ${bFirst.actRms.toFixed(2)} using ${net.initScheme.toUpperCase()} weight initialization.`,
            activeCol: 0,
            activeEdgeCol: -1
          },
          {
            phase: "forward",
            badge: "BRANCH",
            shortLabel: "2. Pre-RMSNorm & f_l(h)",
            title: "Stage 2 · Split into Identity Skip Path (I) and Pre-RMSNorm Residual Branch",
            formula: "RMSNorm(h) = (h / √(mean(h²) + ε)) ⊙ γ,   f_l(h) = W_l · GELU(RMSNorm(h))",
            summary: hasNorm
              ? "Pre-RMSNorm normalizes activation scale before each dense weight matrix, preventing variance explosion across depth."
              : "Without RMSNorm, activation RMS drifts exponentially with depth L.",
            activeCol: 1,
            activeEdgeCol: 0
          },
          {
            phase: "update",
            badge: "HIGHWAY ⊕",
            shortLabel: "3. Highway Add h + f(h)",
            title: `Stage 3 · Additive Residual Junctions Across All L = ${L} Blocks`,
            formula: hasSkip ? "h⁽ˡ⁾ = h⁽ˡ⁻¹⁾ + α · f_l(h⁽ˡ⁻¹⁾)" : "h⁽ˡ⁾ = f_l(h⁽ˡ⁻¹⁾)  [Skip Highway Disabled]",
            summary: hasSkip
              ? `Each block adds its learned delta f_l(h) onto the identity highway, keeping signal RMS at ${bLast.actRms.toFixed(2)} at depth L=${L}.`
              : `With the skip highway disabled, signal RMS at Block L${L} is ${bLast.actRms.toFixed(2)}.`,
            activeCol: 2,
            activeEdgeCol: 1
          },
          {
            phase: "loss",
            badge: "READOUT",
            shortLabel: "4. Readout ŷ & Loss",
            title: "Stage 4 · Final Classification Readout & Binary Cross-Entropy",
            formula: "ŷ = σ(W_{out} · RMSNorm(h⁽ᴸ⁾) + b_{out})",
            summary: `Deep ${L}-block network achieves ${((lastStats?.accuracy || 0) * 100).toFixed(1)}% accuracy (Loss = ${(lastStats?.loss || 0).toFixed(4)}).`,
            activeCol: 3,
            activeEdgeCol: 2
          },
          {
            phase: "backward",
            badge: "GRAD HIGHWAY",
            shortLabel: "5. Backward Identity ∂L/∂h⁽⁰⁾",
            title: "Stage 5 · Backward Pass Along the Identity Jacobian Highway",
            formula: "∂L / ∂h⁽⁰⁾ = (∂L / ∂h⁽ᴸ⁾) · ∏_{l=1..L} (I + ∂f_l / ∂h⁽ˡ⁻¹⁾)",
            summary: `Because every block Jacobian contains +I, Layer 1 receives ${(ratio * 100).toFixed(1)}% of Layer ${L}'s gradient norm (‖∇W₁‖=${bFirst.gradNorm.toFixed(4)} vs ‖∇W_L‖=${bLast.gradNorm.toFixed(4)}).`,
            activeCol: 1,
            activeEdgeCol: -2
          }
        ],
        readoutBox: {
          lines: [
            { label: "Topology", val: hasSkip ? `ResNet (L=${L})` : `Plain (L=${L})`, color: "#7dd3fc" },
            { label: "Norm", val: hasNorm ? "Pre-RMSNorm" : "Unnormalized", color: "#34d399" },
            { label: "‖∇W₁‖/‖∇W_L‖", val: `${(ratio * 100).toFixed(1)}%`, color: "#fde047" },
            { label: "Accuracy", val: `${((lastStats?.accuracy || 0) * 100).toFixed(1)}%`, color: "#fda4af" }
          ]
        },
        epochLabel: `Update Step ${net.stepCount}`
      };
    }
  });
}

// ============================================================================
// 4. TRANSFORMER TRACE BUILDER (Token+Pos → Q,K,V → Softmax(QK^T/√d_k) → Out)
// ============================================================================

export function createTransformerStepper(containerEl, getTfState) {
  return mountSectionStepper(containerEl, {
    idPrefix: "tf-step",
    title: "Interactive Multi-Head Self-Attention Node Graph & Routing Step-Through",
    subtitle:
      "Step through how input tokens combine with sinusoidal positional encodings, project into Queries (Q), Keys (K), and Values (V), route information across tokens via Softmax(QKᵀ/√d_k) attention edges, and predict output tokens.",
    probeLabel: "Inspect Attention Head Routing",
    probeOptions: [
      { value: "0", label: "Head #1 Self-Attention Routing Graph (d_k = 8)" },
      { value: "1", label: "Head #2 Self-Attention Routing Graph (d_k = 8)" }
    ],
    buildTrace(probeVal) {
      const { tf, tokenIds } = getTfState();
      const headIdx = parseInt(probeVal, 10) || 0;
      const fwd = tf.forward(tokenIds);
      const weights = fwd.attnWeights[headIdx] || fwd.attnWeights[0];
      const inTokens = tokenIds.map((i) => VOCAB_TOKENS[i]);
      const outTokens = fwd.predictions.map((i) => VOCAB_TOKENS[i]);

      const col0Nodes = [0, 1, 2, 3].map((pos) => ({
        id: `TF_X${pos}`,
        label: `Pos ${pos}: ${inTokens[pos]}`,
        val: 0.65,
        displayVal: `tok="${inTokens[pos]}"`,
        roleTitle: `Token Position ${pos} ("${inTokens[pos]}") · Embedding E(x_${pos}) + Sinusoidal PE(${pos})`,
        kpis: [
          { k: "Input Token", v: `"${inTokens[pos]}" (id=${tokenIds[pos]})` },
          { k: "Sequence Position", v: `pos = ${pos}` },
          { k: "Embedding Dim", v: "d_model = 16" }
        ],
        incoming: []
      }));

      const col1Nodes = [0, 1, 2, 3].map((pos) => ({
        id: `TF_QK${pos}`,
        label: `Q_${pos} / K_${pos}`,
        val: 0.55,
        displayVal: `Head #${headIdx + 1}`,
        roleTitle: `Position ${pos} ("${inTokens[pos]}") · Query Q_${pos} & Key K_${pos} Projections (Head #${headIdx + 1})`,
        kpis: [
          { k: "Head Subspace", v: `Head #${headIdx + 1} (d_k = 8)` },
          { k: "Scale Factor", v: "1 / √d_k = 1 / √8 ≈ 0.3536" },
          { k: "Masking Mode", v: tf.causal ? "Causal (j ≤ i)" : "Bidirectional (All i,j)" }
        ],
        incoming: [
          {
            fromId: `TF_X${pos}`,
            fromLabel: `Pos ${pos}: ${inTokens[pos]}`,
            inVal: inTokens[pos],
            weightVal: 0.8,
            weightStr: "W_Q, W_K",
            contribVal: 0.8,
            contribStr: "d_k=8 vec",
            detailStr: `Projects X[${pos}] into Query & Key subspaces`
          }
        ]
      }));

      // Column 2: Attention-routed Context Nodes (edges weighted by live 4x4 attention matrix!)
      const col2Nodes = [0, 1, 2, 3].map((qPos) => {
        let maxK = 0;
        for (let k = 1; k < 4; k++) if (weights[qPos * 4 + k] > weights[qPos * 4 + maxK]) maxK = k;
        const maxAttn = weights[qPos * 4 + maxK];

        const incoming = [0, 1, 2, 3].map((kPos) => {
          const aVal = weights[qPos * 4 + kPos];
          return {
            fromId: `TF_QK${kPos}`,
            fromLabel: `K_${kPos} (${inTokens[kPos]})`,
            inVal: `V_${kPos}(${inTokens[kPos]})`,
            weightVal: aVal,
            weightStr: `α=${aVal.toFixed(2)}`,
            contribVal: aVal,
            contribStr: `${(aVal * 100).toFixed(1)}%`,
            detailStr: `Attention weight A[q=${qPos}, k=${kPos}] in Head #${headIdx + 1}`
          };
        });

        return {
          id: `TF_CTX${qPos}`,
          label: `Ctx_${qPos}←${inTokens[maxK]}`,
          val: maxAttn,
          displayVal: `α=${maxAttn.toFixed(2)}`,
          roleTitle: `Context Node ${qPos} · Attention-Weighted Value Sum ∑_j A[${qPos},j] · V_j (Top source: Pos ${maxK} "${inTokens[maxK]}")`,
          kpis: [
            { k: "Top Attended Token", v: `Pos ${maxK} ("${inTokens[maxK]}")` },
            { k: "Peak Attention α", v: `${(maxAttn * 100).toFixed(1)}%` },
            { k: "Row Entropy", v: "Softmax normalized (∑ α = 1.0)" }
          ],
          incoming
        };
      });

      const col3Nodes = [0, 1, 2, 3].map((pos) => ({
        id: `TF_OUT${pos}`,
        label: `ŷ_${pos} = ${outTokens[pos]}`,
        val: 0.85,
        displayVal: `→ "${outTokens[pos]}"`,
        roleTitle: `Output Position ${pos} · Residual Add + LayerNorm + Vocab Projection → "${outTokens[pos]}"`,
        kpis: [
          { k: "Predicted Token ŷ", v: `"${outTokens[pos]}"` },
          { k: "Input Token x", v: `"${inTokens[pos]}"` },
          { k: "Residual Block", v: "LayerNorm(X + W_O · Concat(Heads))" }
        ],
        incoming: [
          {
            fromId: `TF_CTX${pos}`,
            fromLabel: col2Nodes[pos].label,
            inVal: col2Nodes[pos].displayVal,
            weightVal: 0.9,
            weightStr: "W_O + FFN",
            contribVal: 0.9,
            contribStr: `→ "${outTokens[pos]}"`,
            detailStr: "Multi-head output projection + Residual + Vocab head"
          }
        ]
      }));

      return {
        columns: [
          { header: "L0 · Token+PE (4)", nodes: col0Nodes },
          { header: `L1 · Head #${headIdx + 1} Q,K`, nodes: col1Nodes },
          { header: "L2 · Attn A·V (4×4)", nodes: col2Nodes },
          { header: "L3 · Output ŷ (4)", nodes: col3Nodes }
        ],
        microSteps: [
          {
            phase: "input",
            badge: "EMBED + PE",
            shortLabel: "1. Token + Pos Embed",
            title: `Stage 1 · Lookup Token Embeddings + Sinusoidal Positional Encoding`,
            formula: "X_pos = E[token_id] + PE(pos),   X ∈ ℝ^{4 × 16}",
            summary: `Embedded input sequence [${inTokens.join(", ")}] and added position-dependent sinusoids so permutation-equivariant attention knows token order.`,
            activeCol: 0,
            activeEdgeCol: -1
          },
          {
            phase: "forward",
            badge: "Q, K, V PROJ",
            shortLabel: "2. Project Q, K, V",
            title: `Stage 2 · Project X into Query (Q), Key (K), and Value (V) Subspaces (Head #${headIdx + 1})`,
            formula: "Q^{(h)} = X · W_Q^{(h)},   K^{(h)} = X · W_K^{(h)},   V^{(h)} = X · W_V^{(h)} ∈ ℝ^{4 × 8}",
            summary: `Split d_model=16 across 2 heads (d_k = 8 per head) so Head #1 and Head #2 can attend to different positional/content patterns simultaneously.`,
            activeCol: 1,
            activeEdgeCol: 0
          },
          {
            phase: "update",
            badge: "ATTENTION",
            shortLabel: "3. Softmax(QKᵀ/√d_k)",
            title: `Stage 3 · All-Pairs Scaled Dot-Product Attention Routing (Head #${headIdx + 1})`,
            formula: "A^{(h)} = Softmax( (Q^{(h)} · K^{(h)T}) / √8 + M_{mask} ) ∈ [0, 1]^{4 × 4}",
            summary: `Computed 4×4 token-to-token routing weights! Click any Ctx_i node in L2 to inspect its exact attention weights α_{i,j} from every token.`,
            activeCol: 2,
            activeEdgeCol: 1
          },
          {
            phase: "forward",
            badge: "CONTEXT MIX",
            shortLabel: "4. Context ∑ α_{ij} V_j",
            title: "Stage 4 · Aggregate Value Vectors & Concat Multi-Head Outputs",
            formula: "Ctx_i = ∑_{j=0..3} A_{i,j} · V_j,   H = LayerNorm(X + Concat(Head_1, Head_2) · W_O)",
            summary: `Each position i gathers a weighted combination of Value vectors V_j according to its live attention distribution A[i, :].`,
            activeCol: 2,
            activeEdgeCol: 1
          },
          {
            phase: "loss",
            badge: "VOCAB HEAD",
            shortLabel: "5. Output Logits ŷ",
            title: "Stage 5 · Position-Wise FFN + Vocabulary Logits Projection",
            formula: "ŷ_i = argmax_v Softmax( H_i · W_{vocab} + b_{vocab} )",
            summary: `Input [${inTokens.join(", ")}] mapped in a single parallel forward pass to [${outTokens.join(", ")}].`,
            activeCol: 3,
            activeEdgeCol: 2
          }
        ],
        readoutBox: {
          lines: [
            { label: "Input Seq", val: `[${inTokens.join(" ")}]`, color: "#ffffff" },
            { label: "Pred Seq", val: `[${outTokens.join(" ")}]`, color: "#34d399" },
            { label: "Active Head", val: `Head #${headIdx + 1} of 2`, color: "#7dd3fc" },
            { label: "Mask Mode", val: tf.causal ? "Causal Mask" : "Bidirectional", color: "#fde047" }
          ]
        },
        epochLabel: `Transformer Epoch ${tf.stepCount}`
      };
    }
  });
}

// ============================================================================
// 5. DIFFUSION TRACE BUILDER (Continuous DDPM ε_θ & Discrete Masked MDLM)
// ============================================================================

export function createDiffusionStepper(containerEl, getDiffState) {
  return mountSectionStepper(containerEl, {
    idPrefix: "diff-step",
    title: "Interactive Denoising Diffusion Node Graph & Reverse Markov Step-Through",
    subtitle:
      "Step through the time-conditioned score network ε_θ(x_t, t) (Continuous Pixel DDPM) and bidirectional unmasking network p_θ(x_0 | x_t) (Discrete Masked Text Diffusion) across the reverse trajectory t = T → 0.",
    probeLabel: "Select Diffusion Modality",
    probeOptions: [
      { value: "ddpm", label: "Continuous Image DDPM: Time-Conditioned Noise Predictor ε_θ(x_t, t)" },
      { value: "mdlm", label: "Discrete Masked Text Diffusion (MDLM): Confidence Unmasking p_θ(x_0 | x_t)" }
    ],
    buildTrace(probeVal) {
      const { ddpm, trajectory, textDiff, textPromptIdx } = getDiffState();
      const isText = probeVal === "mdlm";

      if (!isText) {
        const midFrame = trajectory.find((f) => f.step === 8) || trajectory[0] || { step: 8, pixels: new Float32Array(36) };
        const cleanFrame = trajectory[trajectory.length - 1] || midFrame;
        const meanNoisy = midFrame.pixels.reduce((a, b) => a + b, 0) / 36;
        const meanClean = cleanFrame.pixels.reduce((a, b) => a + b, 0) / 36;
        const estEps = meanNoisy - meanClean;

        return {
          columns: [
            {
              header: "L0 · State x_t & τ(t)",
              nodes: [
                {
                  id: "DF_XT0",
                  label: "x_t (Center)",
                  val: midFrame.pixels[14] || 0.4,
                  displayVal: (midFrame.pixels[14] || 0.4).toFixed(2),
                  roleTitle: "Noisy Latents x_t at Reverse Step t = 8 (Center Patch Pixels)",
                  kpis: [
                    { k: "Diffusion Timestep", v: "t = 8 / 16" },
                    { k: "Signal Scale √ᾱ_t", v: "0.684" },
                    { k: "Noise Scale √(1−ᾱ_t)", v: "0.729" }
                  ],
                  incoming: []
                },
                {
                  id: "DF_XT1",
                  label: "x_t (Mean)",
                  val: meanNoisy,
                  displayVal: meanNoisy.toFixed(2),
                  roleTitle: "Mean Spatial Intensity of Noisy 6×6 Sprite x_t",
                  kpis: [{ k: "Mean Intensity", v: meanNoisy.toFixed(4) }],
                  incoming: []
                },
                {
                  id: "DF_TIME",
                  label: "τ(t) Sinusoid",
                  val: Math.sin(8 / 16),
                  displayVal: "t=8/16",
                  roleTitle: "Sinusoidal Timestep Embedding τ(t) = [sin(ω·t), cos(ω·t)]",
                  kpis: [
                    { k: "Normalized Time", v: "t / T = 0.50" },
                    { k: "Role", v: "Tells network current noise variance" }
                  ],
                  incoming: []
                }
              ]
            },
            {
              header: "L1 · Score MLP h₁",
              nodes: [0, 1, 2].map((idx) => ({
                id: `DF_H${idx}`,
                label: `Score h_${idx}`,
                val: 0.48 + idx * 0.12,
                displayVal: (0.48 + idx * 0.12).toFixed(2),
                roleTitle: `Time-Conditioned Hidden Neuron h_${idx} = GELU(W₁ · [x_t, τ(t)] + b₁)`,
                kpis: [
                  { k: "Hidden Width", v: "48 GELU units" },
                  { k: "Conditioning", v: "Concat(x_t ∈ ℝ³⁶, τ(t) ∈ ℝ⁴)" }
                ],
                incoming: [
                  {
                    fromId: "DF_XT0",
                    fromLabel: "x_t (Center)",
                    inVal: (midFrame.pixels[14] || 0.4).toFixed(2),
                    weightVal: ddpm.W1[idx] || 0.35,
                    weightStr: `W₁=${(ddpm.W1[idx] || 0.35).toFixed(2)}`,
                    contribVal: 0.32,
                    contribStr: "0.320",
                    detailStr: "Noisy pixel input weight"
                  },
                  {
                    fromId: "DF_TIME",
                    fromLabel: "τ(t) Sinusoid",
                    inVal: "0.50",
                    weightVal: 0.42,
                    weightStr: "W_t=+0.42",
                    contribVal: 0.21,
                    contribStr: "0.210",
                    detailStr: "Timestep embedding modulation"
                  }
                ]
              }))
            },
            {
              header: "L2 · Noise Pred ε̂_θ",
              nodes: [
                {
                  id: "DF_EPS",
                  label: "ε̂_θ(x_t, t)",
                  val: estEps,
                  displayVal: estEps.toFixed(2),
                  roleTitle: "Predicted Gaussian Noise Tensor ε̂_θ(x_t, t) ≈ −√(1−ᾱ_t) ∇_x log p_t(x_t)",
                  kpis: [
                    { k: "Mean Predicted ε̂", v: estEps.toFixed(4) },
                    { k: "Training Objective", v: "MSE ‖ε − ε_θ(x_t, t)‖²" }
                  ],
                  incoming: [0, 1, 2].map((idx) => ({
                    fromId: `DF_H${idx}`,
                    fromLabel: `Score h_${idx}`,
                    inVal: (0.48 + idx * 0.12).toFixed(2),
                    weightVal: ddpm.W2[idx] || 0.28,
                    weightStr: `W₂=${(ddpm.W2[idx] || 0.28).toFixed(2)}`,
                    contribVal: 0.18,
                    contribStr: "0.180",
                    detailStr: "Projects hidden score into 36D noise prediction"
                  }))
                },
                {
                  id: "DF_X0HAT",
                  label: "x̂₀ (Clean Est)",
                  val: meanClean,
                  displayVal: meanClean.toFixed(2),
                  roleTitle: "One-Step Clean Tweedie Estimate x̂₀ = (x_t − √(1−ᾱ_t) ε̂_θ) / √ᾱ_t",
                  kpis: [{ k: "Estimated Clean Mean", v: meanClean.toFixed(4) }],
                  incoming: [
                    {
                      fromId: "DF_H1",
                      fromLabel: "Score h_1",
                      inVal: "0.60",
                      weightVal: 0.55,
                      weightStr: "Tweedie",
                      contribVal: meanClean,
                      contribStr: meanClean.toFixed(3),
                      detailStr: "Subtracts predicted noise component"
                    }
                  ]
                }
              ]
            },
            {
              header: "L3 · Step x_{t-1} → x₀",
              nodes: [
                {
                  id: "DF_OUT",
                  label: "x_{t-1} → x₀",
                  val: meanClean,
                  displayVal: "Denoised",
                  roleTitle: "Reverse Markov Posterior Update x_{t-1} = μ_θ(x_t, t) + σ_t z",
                  kpis: [
                    { k: "Clean Pixel Mean", v: meanClean.toFixed(4) },
                    { k: "Total Reverse Steps", v: "16 DDPM steps (t=16 → 0)" }
                  ],
                  incoming: [
                    {
                      fromId: "DF_EPS",
                      fromLabel: "ε̂_θ(x_t, t)",
                      inVal: estEps.toFixed(2),
                      weightVal: -0.5,
                      weightStr: "−β_t/√1-ᾱ_t",
                      contribVal: -estEps * 0.5,
                      contribStr: (-estEps * 0.5).toFixed(3),
                      detailStr: "Subtracts scaled noise to step toward data manifold"
                    },
                    {
                      fromId: "DF_X0HAT",
                      fromLabel: "x̂₀ (Clean Est)",
                      inVal: meanClean.toFixed(2),
                      weightVal: 1.0,
                      weightStr: "1/√α_t",
                      contribVal: meanClean,
                      contribStr: meanClean.toFixed(3),
                      detailStr: "Posterior mean drift toward x₀"
                    }
                  ]
                }
              ]
            }
          ],
          microSteps: [
            {
              phase: "input",
              badge: "NOISY x_t",
              shortLabel: "1. Input x_t & Time τ(t)",
              title: "Stage 1 · Feed Corrupted Image x_t and Sinusoidal Time Embedding τ(t)",
              formula: "x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε,   τ(t) = [sin(ω t), cos(ω t)]",
              summary: "The denoiser receives the current noisy 6×6 pixel grid alongside the continuous time embedding indicating the noise level β_t.",
              activeCol: 0,
              activeEdgeCol: -1
            },
            {
              phase: "forward",
              badge: "SCORE MLP",
              shortLabel: "2. Score Layer h₁",
              title: "Stage 2 · Fuse Spatial Pixels & Noise Schedule in Hidden Score Layer",
              formula: "h₁ = GELU( W₁ · Concat(x_t, τ(t)) + b₁ )",
              summary: "48 hidden neurons jointly evaluate local pixel correlations and current timestep t.",
              activeCol: 1,
              activeEdgeCol: 0
            },
            {
              phase: "forward",
              badge: "PREDICT ε̂_θ",
              shortLabel: "3. Predict Noise ε̂_θ",
              title: "Stage 3 · Estimate the Added Gaussian Noise ε̂_θ(x_t, t)",
              formula: "ε̂_θ(x_t, t) = W₂ · h₁ + b₂ ≈ −√(1 − ᾱ_t) ∇_{x_t} log p_t(x_t)",
              summary: "Instead of predicting x_0 directly in one shot, the network predicts the noise vector ε that corrupted x_0 into x_t.",
              activeCol: 2,
              activeEdgeCol: 1
            },
            {
              phase: "update",
              badge: "POSTERIOR",
              shortLabel: "4. Step x_t → x_{t-1}",
              title: "Stage 4 · Reverse Markov Posterior Step from x_t to x_{t-1}",
              formula: "x_{t-1} = (1 / √α_t) · (x_t − (β_t / √(1 − ᾱ_t)) · ε̂_θ(x_t, t)) + σ_t z",
              summary: "Subtracting a fraction of the predicted noise nudges the sample one step closer to the high-density sprite manifold.",
              activeCol: 3,
              activeEdgeCol: 2
            }
          ],
          readoutBox: {
            lines: [
              { label: "Modality", val: "Continuous DDPM", color: "#7dd3fc" },
              { label: "Schedule", val: "T = 16 Steps", color: "#ffffff" },
              { label: "Denoised Mean", val: meanClean.toFixed(3), color: "#34d399" },
              { label: "DDPM Epoch", val: String(ddpm.stepCount), color: "#fde047" }
            ]
          },
          epochLabel: `DDPM Step ${ddpm.stepCount}`
        };
      }

      // Discrete Masked Text Diffusion (MDLM)
      const steps = textDiff.denoiseTrajectory(textPromptIdx);
      const midStep = steps[1] || steps[0];
      const finalStep = steps[steps.length - 1];

      return {
        columns: [
          {
            header: "L0 · Masked Seq x_t",
            nodes: midStep.tokens.slice(0, 4).map((tok, pos) => ({
              id: `MD_IN${pos}`,
              label: `Pos ${pos}: ${tok}`,
              val: tok === "[MASK]" ? -0.4 : 0.85,
              displayVal: tok,
              roleTitle: `Sequence Slot ${pos} at Reverse Step 1: "${tok}"`,
              kpis: [
                { k: "Current Token", v: tok },
                { k: "Slot Confidence", v: `${Math.round(midStep.confidences[pos] * 100)}%` }
              ],
              incoming: []
            }))
          },
          {
            header: "L1 · Bidirectional Ctx",
            nodes: [0, 1, 2, 3].map((pos) => ({
              id: `MD_CTX${pos}`,
              label: `BiCtx_${pos}`,
              val: 0.7,
              displayVal: "All-to-All",
              roleTitle: `Bidirectional Context Representation at Slot ${pos} (Sees Both Left & Right Unmasked Anchors)`,
              kpis: [{ k: "Attention Mask", v: "Full Bidirectional (Non-Causal)" }],
              incoming: [0, 1, 2, 3].map((src) => ({
                fromId: `MD_IN${src}`,
                fromLabel: `Pos ${src}: ${midStep.tokens[src]}`,
                inVal: midStep.tokens[src],
                weightVal: 0.5,
                weightStr: "Bi-Attn",
                contribVal: 0.5,
                contribStr: "Context",
                detailStr: "Bidirectional context from unmasked anchor tokens"
              }))
            }))
          },
          {
            header: "L2 · Unmask p_θ(x₀|x_t)",
            nodes: finalStep.tokens.slice(0, 4).map((tok, pos) => ({
              id: `MD_OUT${pos}`,
              label: `${tok} (${Math.round(finalStep.confidences[pos] * 100)}%)`,
              val: finalStep.confidences[pos],
              displayVal: `${Math.round(finalStep.confidences[pos] * 100)}%`,
              roleTitle: `Unmasked Token "${tok}" at Slot ${pos} (Confidence ${(finalStep.confidences[pos] * 100).toFixed(1)}%)`,
              kpis: [
                { k: "Predicted Token", v: tok },
                { k: "Unmask Confidence", v: `${(finalStep.confidences[pos] * 100).toFixed(1)}%` }
              ],
              incoming: [
                {
                  fromId: `MD_CTX${pos}`,
                  fromLabel: `BiCtx_${pos}`,
                  inVal: "Context",
                  weightVal: finalStep.confidences[pos],
                  weightStr: `p=${finalStep.confidences[pos].toFixed(2)}`,
                  contribVal: finalStep.confidences[pos],
                  contribStr: tok,
                  detailStr: "Categorical softmax over vocabulary"
                }
              ]
            }))
          }
        ],
        microSteps: [
          {
            phase: "input",
            badge: "MASKED x_t",
            shortLabel: "1. Absorbing [MASK] State",
            title: "Stage 1 · Partially Masked Sequence x_t in Reverse Trajectory",
            formula: "q(x_t | x_0) = (1 − γ_t) δ_{x_0} + γ_t δ_{[MASK]}",
            summary: `Sequence state: [${midStep.tokens.join(" ")}]. Unmasked anchor tokens provide bidirectional clues for the remaining [MASK] slots.`,
            activeCol: 0,
            activeEdgeCol: -1
          },
          {
            phase: "forward",
            badge: "BI-ENCODER",
            shortLabel: "2. Bidirectional Context",
            title: "Stage 2 · Non-Causal Bidirectional Context Aggregation",
            formula: "H = BiTransformer(x_t)  [Every [MASK] attends to both past and future unmasked tokens]",
            summary: "Unlike autoregressive GPT models that only look left, Masked Diffusion attends to both left and right unmasked anchors simultaneously.",
            activeCol: 1,
            activeEdgeCol: 0
          },
          {
            phase: "loss",
            badge: "UNMASK",
            shortLabel: "3. Confidence Unmasking",
            title: "Stage 3 · Parallel Categorical Prediction & Highest-Confidence Unmasking",
            formula: "L_{MDLM} = ∑_{i : x_t[i]=[MASK]} −log p_θ(x_0[i] | x_t)",
            summary: `Fully denoised sequence: "${finalStep.tokens.join(" ")}" in 4 parallel unmasking steps!`,
            activeCol: 2,
            activeEdgeCol: 1
          }
        ],
        readoutBox: {
          lines: [
            { label: "Modality", val: "Masked Text MDLM", color: "#7dd3fc" },
            { label: "Denoised", val: finalStep.tokens.slice(0, 3).join(" "), color: "#34d399" },
            { label: "Steps", val: "4 Parallel Steps", color: "#ffffff" },
            { label: "MDLM Epoch", val: String(textDiff.stepCount), color: "#fde047" }
          ]
        },
        epochLabel: `MDLM Step ${textDiff.stepCount}`
      };
    }
  });
}

// ============================================================================
// 6. DECISION POINTER HEAD TRACE BUILDER (Delimiter Gather & Abstention Gate)
// ============================================================================

export function createDecisionStepper(containerEl, getDecisionState) {
  return mountSectionStepper(containerEl, {
    idPrefix: "dec-step",
    title: "Interactive Decision Pointer Network (⟨opt_k⟩ Gather & Gate) Node Graph Step-Through",
    subtitle:
      "Step through how a Decision Model scores delimiter-packed options in a single forward pass—without open-ended token generation—and applies temperature calibration and a two-sided abstention gate.",
    buildTrace() {
      const { stateVec, optionVecs, optionLabels, T, tau, res } = getDecisionState();
      const logits = res.rawLogits || res.logits;

      const col0Nodes = [0, 1, 2].map((i) => ({
        id: `DEC_S${i}`,
        label: `State s[${i * 2}..${i * 2 + 1}]`,
        val: stateVec[i * 2],
        displayVal: stateVec[i * 2].toFixed(2),
        roleTitle: `Packed Incident Context State Vector Segment s[${i * 2}..${i * 2 + 1}]`,
        kpis: [
          { k: "Feature s[2i]", v: stateVec[i * 2].toFixed(3) },
          { k: "Feature s[2i+1]", v: stateVec[i * 2 + 1].toFixed(3) }
        ],
        incoming: []
      }));

      const col1Nodes = optionLabels.map((lbl, k) => ({
        id: `DEC_OPT${k}`,
        label: `⟨opt_${k}⟩`,
        val: logits[k],
        displayVal: `z=${logits[k].toFixed(2)}`,
        roleTitle: `Delimiter Slot ⟨opt_${k}⟩: ${lbl} · Pointer Dot-Product Score z_${k}`,
        kpis: [
          { k: "Option Label", v: lbl },
          { k: "Raw Pointer Logit z_k", v: logits[k].toFixed(4) },
          { k: "Scaled Logit z_k / T", v: (logits[k] / T).toFixed(4) }
        ],
        incoming: col0Nodes.map((sn, i) => ({
          fromId: sn.id,
          fromLabel: sn.label,
          inVal: stateVec[i * 2].toFixed(2),
          weightVal: optionVecs[k][i * 2],
          weightStr: `e_${k}=${optionVecs[k][i * 2].toFixed(2)}`,
          contribVal: stateVec[i * 2] * optionVecs[k][i * 2],
          contribStr: (stateVec[i * 2] * optionVecs[k][i * 2]).toFixed(3),
          detailStr: `Pointer dot product h_state · h_{pos(⟨opt_${k}⟩)}`
        }))
      }));

      const col2Nodes = optionLabels.map((lbl, k) => ({
        id: `DEC_P${k}`,
        label: `P(opt_${k})`,
        val: res.probs[k],
        displayVal: `${(res.probs[k] * 100).toFixed(1)}%`,
        roleTitle: `Calibrated Option Probability P(⟨opt_${k}⟩ | T=${T.toFixed(2)}) = ${(res.probs[k] * 100).toFixed(1)}%`,
        kpis: [
          { k: "Softmax Prob p_k", v: `${(res.probs[k] * 100).toFixed(2)}%` },
          { k: "Temperature T", v: T.toFixed(2) }
        ],
        incoming: [
          {
            fromId: `DEC_OPT${k}`,
            fromLabel: `⟨opt_${k}⟩`,
            inVal: `z=${logits[k].toFixed(2)}`,
            weightVal: res.probs[k],
            weightStr: `/ T=${T.toFixed(2)}`,
            contribVal: res.probs[k],
            contribStr: `${(res.probs[k] * 100).toFixed(1)}%`,
            detailStr: "Closed-set Softmax across option delimiters only"
          }
        ]
      }));

      const isAct = res.recommendation === "act";
      const col3Nodes = [
        {
          id: "DEC_GATE",
          label: isAct ? `ACT: ⟨opt_${res.bestIdx}⟩` : "ABSTAIN",
          val: isAct ? res.confidence : -0.5,
          displayVal: isAct ? `${Math.round(res.confidence * 100)}% ≥ τ` : `< τ (${Math.round(tau * 100)}%)`,
          roleTitle: `Confidence & Abstention Gate: Verdict = ${res.recommendation.toUpperCase()}`,
          kpis: [
            { k: "Gate Verdict", v: res.recommendation.toUpperCase() },
            { k: "Choice Confidence", v: `${(res.confidence * 100).toFixed(1)}%` },
            { k: "Gate Threshold τ", v: `${(tau * 100).toFixed(1)}%` },
            { k: "Noul Act Probability", v: `${(res.actProb * 100).toFixed(1)}%` }
          ],
          incoming: col2Nodes.map((pn, k) => ({
            fromId: pn.id,
            fromLabel: `P(opt_${k})`,
            inVal: `${(res.probs[k] * 100).toFixed(1)}%`,
            weightVal: k === res.bestIdx ? 1.0 : 0.2,
            weightStr: k === res.bestIdx ? "p_max" : `p_${k}`,
            contribVal: res.probs[k],
            contribStr: `${(res.probs[k] * 100).toFixed(1)}%`,
            detailStr: k === res.bestIdx ? "Winning option compared against gate τ" : "Competitor option probability"
          }))
        }
      ];

      return {
        columns: [
          { header: "L0 · Context State", nodes: col0Nodes },
          { header: "L1 · ⟨opt_k⟩ Gather", nodes: col1Nodes },
          { header: "L2 · Softmax(z/T)", nodes: col2Nodes },
          { header: "L3 · Gate Verdict", nodes: col3Nodes }
        ],
        microSteps: [
          {
            phase: "input",
            badge: "PACKING",
            shortLabel: "1. Pack State & ⟨opt_k⟩",
            title: "Stage 1 · Pack Context State and Delimiter-Framed Candidate Options",
            formula: "Seq = [State_Tokens, ⟨opt_0⟩, ..., ⟨opt_3⟩]  (Single Forward Pass)",
            summary: "Instead of autoregressive token-by-token generation, all candidate options are packed with dedicated delimiter markers in one pass.",
            activeCol: 0,
            activeEdgeCol: -1
          },
          {
            phase: "forward",
            badge: "POINTER",
            shortLabel: "2. Pointer Dot-Product z_k",
            title: "Stage 2 · Gather Hidden Vectors at Delimiter Positions & Score via Dot Product",
            formula: "z_k = (W_s · h_{state})ᵀ · h_{pos(⟨opt_k⟩)}",
            summary: `Computed raw pointer logits at each delimiter slot: z = [${ Array.from(logits).map((v) => v.toFixed(2)).join(", ") }].`,
            activeCol: 1,
            activeEdgeCol: 0
          },
          {
            phase: "update",
            badge: "CALIBRATE",
            shortLabel: "3. Softmax(z_k / T)",
            title: `Stage 3 · Temperature-Scaled Closed-Set Softmax (T = ${T.toFixed(2)})`,
            formula: "p_k = exp(z_k / T) / ∑_{j=0..K-1} exp(z_j / T)",
            summary: `Softmax is taken strictly over the K=${optionLabels.length} delimiter slots, guaranteeing zero out-of-schema syntax errors.`,
            activeCol: 2,
            activeEdgeCol: 1
          },
          {
            phase: "loss",
            badge: "GATE τ",
            shortLabel: "4. Abstention Gate (τ)",
            title: `Stage 4 · Two-Sided Abstention Gate (Confidence ${(res.confidence * 100).toFixed(1)}% vs τ = ${(tau * 100).toFixed(0)}%)`,
            formula: "Verdict = ACT(argmax_k p_k)  if Confidence ≥ τ  else  ABSTAIN (Unclear)",
            summary: `Verdict: ${res.recommendation.toUpperCase()} (${optionLabels[res.bestIdx]}, confidence ${(res.confidence * 100).toFixed(1)}%).`,
            activeCol: 3,
            activeEdgeCol: 2
          }
        ],
        readoutBox: {
          lines: [
            { label: "Verdict", val: res.recommendation.toUpperCase(), color: isAct ? "#34d399" : "#fda4af" },
            { label: "Top Option", val: `⟨opt_${res.bestIdx}⟩`, color: "#7dd3fc" },
            { label: "Confidence", val: `${(res.confidence * 100).toFixed(1)}%`, color: "#fde047" },
            { label: "Gate τ", val: `${(tau * 100).toFixed(0)}% (T=${T.toFixed(2)})`, color: "#ffffff" }
          ]
        },
        epochLabel: "Single-Pass Pointer Readout"
      };
    }
  });
}

// ============================================================================
// 7. COMPOSABLE BLOCK BUILDER TRACE BUILDER (Dynamic Layer-by-Layer Graph)
// ============================================================================

export function createBuilderStepper(containerEl, getPipeline) {
  return mountSectionStepper(containerEl, {
    idPrefix: "bld-step",
    title: "Interactive Custom Architecture Node Graph & Layer-by-Layer Signal Propagator",
    subtitle:
      "Add or remove blocks in the palette above and step forward or backward through your custom neural architecture to inspect tensor shapes, FLOPs, activation norms ‖a⁽ˡ⁾‖, and backpropagated gradient norms ‖∇⁽ˡ⁾‖.",
    buildTrace() {
      const pipeline = getPipeline();
      const analysis = analyzeArchitecturePipeline(pipeline, { seqLen: 16, dModel: 32 });
      const stages = analysis.stages.slice(0, 5); // Show up to 5 columns cleanly in SVG

      const columns = stages.map((st, cIdx) => {
        const prevStage = cIdx > 0 ? stages[cIdx - 1] : null;
        const nodes = [0, 1, 2].map((ch) => {
          const val = st.actNorm * (1 - ch * 0.12);
          return {
            id: `BLD_C${cIdx}_N${ch}`,
            label: cIdx === 0 ? `In_Ch${ch}` : `B${st.index}_Ch${ch}`,
            val,
            displayVal: `‖a‖=${val.toFixed(2)}`,
            roleTitle: `Block ${st.index}: ${st.label} (${st.category}) · Channel Group ${ch + 1}`,
            kpis: [
              { k: "Block Type", v: st.label },
              { k: "Output Tensor Shape", v: `[${st.outShape}]` },
              { k: "Block Parameters", v: st.params.toLocaleString() },
              { k: "Block FLOPs", v: st.flops.toLocaleString() },
              { k: "Forward Norm ‖a‖", v: st.actNorm.toFixed(4) },
              { k: "Backward Grad ‖∇‖", v: st.gradNorm.toFixed(4) }
            ],
            incoming: prevStage
              ? [0, 1, 2].map((pCh) => ({
                  fromId: `BLD_C${cIdx - 1}_N${pCh}`,
                  fromLabel: cIdx - 1 === 0 ? `In_Ch${pCh}` : `B${prevStage.index}_Ch${pCh}`,
                  inVal: prevStage.actNorm.toFixed(2),
                  weightVal: pCh === ch ? 0.85 : 0.35,
                  weightStr: pCh === ch ? `‖∇‖=${st.gradNorm.toFixed(2)}` : `W_${pCh}${ch}`,
                  contribVal: st.actNorm * 0.4,
                  contribStr: `[${st.outShape}]`,
                  detailStr: `${prevStage.label} → ${st.label}`
                }))
              : []
          };
        });
        return {
          header: `B${st.index} · ${st.label.split(" ")[0]}`,
          nodes
        };
      });

      const microSteps = stages.map((st, idx) => ({
        phase: idx === 0 ? "input" : idx === stages.length - 1 ? "loss" : "forward",
        badge: `BLOCK ${st.index}`,
        shortLabel: `${idx + 1}. ${st.label.split(" ")[0]}`,
        title: `Stage ${idx + 1} · Execute Block ${st.index}: ${st.label} (${st.category})`,
        formula: `OutShape = [${st.outShape}],   Params = ${st.params.toLocaleString()},   FLOPs = ${st.flops.toLocaleString()}`,
        summary: `Propagated tensor through ${st.label}: forward activation norm ‖a‖ = ${st.actNorm.toFixed(3)}, backward gradient norm ‖∇‖ = ${st.gradNorm.toFixed(3)}.`,
        activeCol: idx,
        activeEdgeCol: idx - 1
      }));

      microSteps.push({
        phase: "backward",
        badge: "BACKPROP",
        shortLabel: `${stages.length + 1}. Global Backprop`,
        title: `Stage ${stages.length + 1} · End-to-End Reverse-Mode Differentiation Across Custom Stack`,
        formula: "∂L / ∂θ_1 = (∂L / ∂a_L) · ∏_{l=2..L} (∂a_l / ∂a_{l-1}) · (∂a_1 / ∂θ_1)",
        summary: `Total architecture: ${analysis.stages.length} blocks, ${analysis.totalParams.toLocaleString()} parameters, ${analysis.totalFlops.toLocaleString()} FLOPs. First-block gradient norm ‖∇₁‖ = ${(analysis.stages[0]?.gradNorm ?? 0).toFixed(3)}.`,
        activeCol: 0,
        activeEdgeCol: -2
      });

      return {
        columns,
        microSteps,
        readoutBox: {
          lines: [
            { label: "Total Blocks", val: `${analysis.stages.length} layers`, color: "#7dd3fc" },
            { label: "Parameters", val: analysis.totalParams.toLocaleString(), color: "#ffffff" },
            { label: "Total FLOPs", val: analysis.totalFlops.toLocaleString(), color: "#34d399" },
            { label: "‖∇_block1‖", val: (analysis.stages[0]?.gradNorm ?? 0).toFixed(3), color: "#fde047" }
          ]
        },
        epochLabel: `${analysis.stages.length}-Block Custom Stack`
      };
    }
  });
}
