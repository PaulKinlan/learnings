// @ts-check
/**
 * OPT-175B Chronicles - Interactive Logbook & Visualizer
 * Based on Meta AI's OPT chronicles: https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles
 */

// Timeline Events Data
export const CHRONICLE_EVENTS = [
  {
    id: "e-plan-b",
    date: "2021-10-18",
    phase: "pre",
    category: "tune",
    title: "Plan B Contingency Established",
    summary: "Team sets decision rule: abort custom Fairseq configs if unstable, switch to Megatron/OpenAI GPT-3 settings.",
    quote: "We decided to follow through with our 'plan B' that we set for ourselves on October 18 before starting any of these runs, where we would abort from these configurations and adopt as much of the Megatron/OpenAI GPT-3 settings as possible.",
    tokens: "0B",
    step: 0,
    impact: "Strategic pivot plan that saved the project after 11.xx lineage stalled."
  },
  {
    id: "e-dataset-glitch",
    date: "2021-10-28",
    phase: "pre",
    category: "stability",
    title: "Dataset Dedup & Regex Formatting Crisis",
    summary: "1.3B pilot runs showed suspiciously low perplexity caused by extra backslashes in JSON decoding and corrupted whitespace.",
    quote: "Found issues with the new dataset where perplexity was unreasonably low, which ended up being a combination of JSON encoding issues (extra backslashes got added in the dedup process) and general corpus issues (Enron/EuroParl/DM/StackExchange).",
    tokens: "0B",
    step: 0,
    impact: "Forced fallback to GPT-2 BPE due to lack of time to retrain custom tokenizer before EOY deadline."
  },
  {
    id: "e-run-11-launch",
    date: "2021-11-05",
    phase: "phase-10",
    category: "tune",
    title: "Launch Exp 11.0 at 175B Scale",
    summary: "First 175B run kicks off with 1024 80GB A100s, 8x Tensor Parallelism, FP32 Adam, Normformer, sinusoidal LPE, 2M batch size.",
    quote: "Finally launched experiment 11.0 on November 5 at full 175B scale... Batch-size of 2M, FP32 Adam, 8x Tensor Parallelism, LPE with sinusoidal initialization, Normformer, LR of 3e-4.",
    tokens: "1.2B",
    step: 300,
    impact: "Marked the beginning of the ill-fated 11.xx lineage."
  },
  {
    id: "e-run-11-loss-spike",
    date: "2021-11-07",
    phase: "phase-10",
    category: "stability",
    title: "Exp 11.4: Loss Explosion & Underflow",
    summary: "Grad norm explosions and NaN loss after a few hundred updates. Loss scale dropped to minimum and underflowed.",
    quote: "From experiment 11.4 onward, we saw grad norm explosions / loss explosions / nans after a couple hundred updates after each restart, along with extremely unstable loss scales that would drop to the point of massively underflowing.",
    tokens: "5.4B",
    step: 1350,
    impact: "Gradient clipping lowered from 2.5 to 1.0; Adam beta2 lowered to 0.95."
  },
  {
    id: "e-relu-hotswap",
    date: "2021-11-09",
    phase: "phase-10",
    category: "tune",
    title: "Exp 11.10: Hot-swapped GeLU with ReLU",
    summary: "Identified GeLU's x³ term as a likely source of FP16 instability; hot-swapped to ReLU mid-lineage and rebuilt MHA.",
    quote: "We started taking more and more drastic actions then... until finally by experiment 11.10 we hot-swapped in ReLU and also switched to a more stable MHA calculation (noting that the x**3 term in GeLU might be a source of instability with FP16).",
    tokens: "14.2B",
    step: 3550,
    impact: "Delayed collapse, but model still stagnated at ~4750 steps."
  },
  {
    id: "e-lpe-frozen",
    date: "2021-11-10",
    phase: "phase-10",
    category: "stability",
    title: "Learned Positional Embeddings Frozen",
    summary: "Ablations in RSC revealed sinusoidal initialization effectively froze positional embeddings in place.",
    quote: "At the same time, we also found that our LPEs were not training... the sinusoidal initialization of LPEs (with the rest of the model configuration) effectively froze the positional embeddings, which seemed to indicate a potential issue with our initialization scheme in general.",
    tokens: "19.0B",
    step: 4750,
    impact: "Exposed deep architectural bugs in 11.xx; prompted immediate trigger of Plan B."
  },
  {
    id: "e-run-12-pivot",
    date: "2021-11-11",
    phase: "phase-10",
    category: "tune",
    title: "The Pivot: Launch Exp 12.00 (Megatron Settings)",
    summary: "Terminated 11.xx lineage. Reverted to OpenAI GPT-3 / Megatron architecture: standard init, removed Normformer, standard LPE.",
    quote: "On November 11, we started training our 12.00 experiment with all of these changes, and since then, the only restarts we've had to make were all related to hardware issues... activation norms, grad norms, param norms all appear sane.",
    tokens: "22.5B",
    step: 5600,
    impact: "Saved the run. Initial loss was higher than 11.xx, but stability unlocked rapid monotonic convergence."
  },
  {
    id: "e-10-percent-milestone",
    date: "2021-11-17",
    phase: "phase-10",
    category: "milestone",
    title: "10% Training Milestone Reached",
    summary: "Published 10% update. Validation loss on track to match GPT-3 175B. Team realizes 180B token corpus will need repetition.",
    quote: "We are currently almost ~10% of the way through... our tokenized dataset is only around 180B tokens, and repeating data at this scale may not be desirable, so we may have to terminate before hitting 300B.",
    tokens: "30.0B",
    step: 7500,
    impact: "Proved 12.xx configuration was viable; initiated evaluation prep."
  },
  {
    id: "e-thanksgiving-break",
    date: "2021-11-25",
    phase: "phase-27",
    category: "hardware",
    title: "Thanksgiving Hardware Nightmare (40+ Restarts)",
    summary: "Cluster suffered ~2 dead nodes per day with uncorrectable ECC errors. Replacement nodes repeatedly failed or came back broken.",
    quote: "It’s been really rough for the team... Since then, we've had 40+ restarts in the 175B experiment for a variety of hardware, infrastructure, or experimental stability issues. Machines seemed to have enjoyed going on break then too.",
    tokens: "52.0B",
    step: 13000,
    impact: "Engineers wrote custom GPU burn-in testing, InfiniBand monitors, and automated node-replacement scripts."
  },
  {
    id: "e-checkpoint-hangs",
    date: "2021-11-28",
    phase: "phase-27",
    category: "hardware",
    title: "1.6 TB Checkpoint Download Bottleneck",
    summary: "Restarts stalled for hours because downloading 992 checkpoint files (1.7GB each) caused non-deterministic cloud blob store hangs.",
    quote: "There were also issues with blob store when downloading 1.6TB of a single model checkpoint (992 files, each ~1.7GB) on restarts, at which point the downloads themselves would start hanging nondeterministically, which then delayed training recovery even further.",
    tokens: "64.0B",
    step: 16000,
    impact: "Forced redesign of restart coalescing and sharding pipelines."
  },
  {
    id: "e-sgd-experiment",
    date: "2021-11-30",
    phase: "phase-27",
    category: "stability",
    title: "Gradient Overflow & The Emergency SGD Experiment",
    summary: "Loss scale collapsed again. Team attempted switching Adam to SGD behavior (beta1=0, eps=1), then custom FP16 vanilla SGD.",
    quote: "We restarted from previous checkpoints a couple of times... we tried something a bit more drastic by testing out 'SGD'-like settings for Adam, only to realize that reinitializing Adam from a checkpoint also reloaded the previously saved betas. We tried switching to true vanilla SGD...",
    tokens: "72.0B",
    step: 18000,
    impact: "Vanilla SGD progressed too slowly; team resolved issue by dropping Adam LR by 10%, which stabilized grad norms."
  },
  {
    id: "e-27-percent-milestone",
    date: "2021-12-03",
    phase: "phase-27",
    category: "milestone",
    title: "27% Milestone & Open Call for On-Call Volunteers",
    summary: "27% update published. Massive downtime costs prompted team to recruit internal volunteer rotation to share 24/7 on-call pain.",
    quote: "We are also happily inviting people to join our on-call. We have established runbooks and some tooling for dealing with the most common issues... each hour the experiment stagnates or goes down costs us $$, so the goal of the on-call is to minimize this downtime.",
    tokens: "81.0B",
    step: 20250,
    impact: "Moya Chen, Kurt Shuster, Punit Singh Koura, Mikel Artetxe, and Dániel Simig joined the rotation."
  },
  {
    id: "e-five-nodes-down",
    date: "2021-12-06",
    phase: "phase-56",
    category: "hardware",
    title: "5 Hosts Down Simultaneously: Buffer Depleted",
    summary: "A cascade of hardware faults killed 5 hosts at once when the buffer pool had only 4 spare nodes.",
    quote: "Our cloud provider has agreed to provision an extra 18 hosts for us to use as a 'buffer'... This has greatly reduced our experiment recovery time, especially considering 5 hosts going down all together on Dec 6 (back when we only had a buffer of 4 hosts).",
    tokens: "95.0B",
    step: 23750,
    impact: "Management and business development intervened to secure an 18-host hot-standby buffer pool."
  },
  {
    id: "e-megatron-profiler",
    date: "2021-12-10",
    phase: "phase-56",
    category: "tune",
    title: "The Megatron v2.6 JIT Flag Mystery",
    summary: "Accidental upgrade to Megatron 2.6 shifted activation norms and boosted throughput by 2%.",
    quote: "One interesting change in our training dynamics occurred when we accidentally upgraded our Megatron dependency to 2.6 (from 2.4)... Myle ended up chasing this down to a one-line change in Megatron which prevents the following code from executing: torch._C._jit_set_profiling_executor(False).",
    tokens: "120.0B",
    step: 30000,
    impact: "Uncovered hidden interaction between PyTorch JIT execution graphs and distributed activation norms."
  },
  {
    id: "e-run-12-45-divergence",
    date: "2021-12-14",
    phase: "phase-56",
    category: "stability",
    title: "Run 12.45.2 Divergence Averted",
    summary: "Grad norms spiked violently and perplexity began diverging. LR cut to 2/3 of GPT-3 baseline to pull run back from the brink.",
    quote: "On run 12.45.2, our training perplexity started to diverge after a few large grad norm spikes. We lowered our learning rate at this point to 2/3 of what OpenAI 175B GPT-3 used, and managed to continue training.",
    tokens: "148.0B",
    step: 37000,
    impact: "Prevented terminal divergence; established dynamic LR scaling runbook."
  },
  {
    id: "e-56-percent-milestone",
    date: "2021-12-16",
    phase: "phase-56",
    category: "milestone",
    title: "56% Milestone & Record 2.8-Day Run",
    summary: "Team achieves longest uninterrupted run (2.8 days). DeepMind releases Gopher (280B), highlighting massive compute cost of ablations.",
    quote: "We managed to hit our top three record long runs of the experiment these past two weeks, lasting 1.5, 2.8, and 2 days each!... DeepMind just released details on their 280B Gopher model... allocating just enough compute budget to train a large-scale model won't be enough to guarantee a better model.",
    tokens: "168.0B",
    step: 42000,
    impact: "Proved that cluster stability tooling was working; began layer-sliced FSDP evaluation."
  },
  {
    id: "e-omicron-cluster-deletion",
    date: "2021-12-21",
    phase: "phase-100",
    category: "sev",
    title: "THE OMICRON SEV: Accidental Cluster Deletion!",
    summary: "Cloud provider support accidentally deleted the ENTIRE 1024-GPU cluster while attempting to replenish the 12-node buffer pool!",
    quote: "In the process of replenishing this pool, the cloud provider's support team accidentally deleted our entire cluster on December 21. While the cluster was restored fairly quickly, it unfortunately came back with 16 machines that did not pass our infrastructure checks.",
    tokens: "198.0B",
    step: 49500,
    impact: "Catastrophic Sev-1 incident. Required all-hands escalation with TPM and cloud provider leadership to restore minimum nodes by EOD Dec 23."
  },
  {
    id: "e-automated-recovery-live",
    date: "2021-12-25",
    phase: "phase-100",
    category: "auto",
    title: "Automated Cluster Self-Healing Goes Live",
    summary: "Team wired together automated health checks, IB diagnostics, and auto-restart scripts to handle Christmas break.",
    quote: "The team started gluing together all of our monitoring and health-checks for the cluster in order to enable fully automated recovery. It took several iterations to get right, but we were able to automatically recover from 8 hardware failures between Christmas and New Years.",
    tokens: "235.0B",
    step: 58750,
    impact: "Achieved autonomous cluster self-healing; on-call engineers were not woken up once during New Year."
  },
  {
    id: "e-run-complete",
    date: "2022-01-06",
    phase: "phase-100",
    category: "milestone",
    title: "Training Complete: 300B Tokens, 4.30E+23 FLOPs",
    summary: "At 12:46 PM PST, OPT-175B finished training. Completed 300B tokens across ~33 days of GPU compute with ~90 restarts.",
    quote: "As of yesterday, at 12:46pm PST on January 6, our 175B model finally completed its training run on 300B tokens. This required ~4.30E+23 FLOPs of compute, or roughly ~33 days of continuous training on 1024 80GB A100s... initial evaluation results, when compared with the OpenAI davinci API, seems very close.",
    tokens: "300.0B",
    step: 75000,
    impact: "Delivered first open-source 175B model with full public logbooks and training chronicles."
  }
];

// Loss Trajectory Reference Curves
export const LOSS_DATA = {
  // Step, 11.xx loss (diverged), 12.xx loss (converged), GPT-3 reference
  steps: [0, 500, 1000, 2000, 3500, 5000, 6500, 8000, 12000, 16000, 24000, 32000, 42000, 55000, 75000],
  exp11: [10.8, 8.2, 6.9, 5.8, 5.1, 4.95, 5.4, null, null, null, null, null, null, null, null],
  exp12: [11.4, 9.1, 7.8, 6.4, 5.4, 4.7, 4.2, 3.85, 3.42, 3.20, 2.98, 2.82, 2.68, 2.54, 2.42],
  gpt3:  [11.2, 8.9, 7.5, 6.1, 5.2, 4.5, 4.0, 3.65, 3.25, 3.02, 2.80, 2.65, 2.50, 2.38, 2.28]
};

// UI State
let activePhaseFilter = "all";
let activeCategoryFilter = "all";
let activeSearchQuery = "";
let selectedEventId = "e-omicron-cluster-deletion";

// Initialize App
export function initChronicles() {
  renderTimelineSVG();
  renderLossChartSVG();
  renderEventList();
  renderSelectedEvent();
  setupEventListeners();
  initSimulator();
}

function setupEventListeners() {
  // Phase Filters
  document.querySelectorAll("[data-phase-filter]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-phase-filter]").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-pressed", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");
      // @ts-ignore
      activePhaseFilter = btn.dataset.phaseFilter;
      applyFilters();
    });
  });

  // Category Filters
  document.querySelectorAll("[data-cat-filter]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-cat-filter]").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-pressed", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");
      // @ts-ignore
      activeCategoryFilter = btn.dataset.catFilter;
      applyFilters();
    });
  });

  // Search input
  const searchInput = document.getElementById("chronicle-search");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      // @ts-ignore
      activeSearchQuery = e.target.value.toLowerCase().trim();
      applyFilters();
    });
  }
}

function applyFilters() {
  renderTimelineSVG();
  renderEventList();
}

function getFilteredEvents() {
  return CHRONICLE_EVENTS.filter(ev => {
    const matchPhase = activePhaseFilter === "all" || ev.phase === activePhaseFilter;
    const matchCat = activeCategoryFilter === "all" || ev.category === activeCategoryFilter;
    const matchSearch = !activeSearchQuery || 
      ev.title.toLowerCase().includes(activeSearchQuery) ||
      ev.summary.toLowerCase().includes(activeSearchQuery) ||
      ev.quote.toLowerCase().includes(activeSearchQuery) ||
      ev.date.includes(activeSearchQuery);

    return matchPhase && matchCat && matchSearch;
  });
}

function selectEvent(id) {
  selectedEventId = id;
  renderSelectedEvent();
  renderTimelineSVG();
  renderEventList();
  
  // Smooth scroll inspector into view if on small screen
  const inspector = document.getElementById("event-inspector");
  if (inspector && window.innerWidth < 860) {
    inspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function renderSelectedEvent() {
  const ev = CHRONICLE_EVENTS.find(e => e.id === selectedEventId);
  const container = document.getElementById("event-inspector");
  if (!ev || !container) return;

  const catBadgeClass = {
    sev: "badge-sev",
    hardware: "badge-hardware",
    stability: "badge-stability",
    tune: "badge-tune",
    milestone: "badge-auto",
    auto: "badge-auto"
  }[ev.category] || "badge-tune";

  container.innerHTML = `
    <div class="inspector-meta">
      <span class="badge ${catBadgeClass}">${ev.category.toUpperCase()}</span>
      <span class="inspector-date">${ev.date}</span>
      <span class="inspector-tokens">Tokens: ${ev.tokens} (${ev.step} steps)</span>
    </div>
    <h3 class="inspector-title">${ev.title}</h3>
    <p class="inspector-summary">${ev.summary}</p>
    <div class="log-quote"><strong>Excerpt from Logbook:</strong>\n"${ev.quote}"</div>
    <div class="inspector-impact"><strong class="inspector-impact-label">Key Impact:</strong> ${ev.impact}</div>
  `;
}

function renderEventList() {
  const listEl = document.getElementById("chronicle-event-list");
  if (!listEl) return;

  const filtered = getFilteredEvents();
  if (filtered.length === 0) {
    listEl.innerHTML = `<div class="event-empty">No matching logbook entries found. Try clearing filters.</div>`;
    return;
  }

  listEl.innerHTML = filtered.map(ev => {
    const isActive = ev.id === selectedEventId ? "active" : "";
    const catBadgeClass = {
      sev: "badge-sev",
      hardware: "badge-hardware",
      stability: "badge-stability",
      tune: "badge-tune",
      milestone: "badge-auto",
      auto: "badge-auto"
    }[ev.category] || "badge-tune";

    return `
      <div class="event-row ${isActive}" role="button" tabindex="0" aria-pressed="${ev.id === selectedEventId}" data-id="${ev.id}">
        <div class="event-row-date">${ev.date}</div>
        <div class="event-category"><span class="badge ${catBadgeClass}">${ev.category}</span></div>
        <div>
          <strong class="event-row-title">${ev.title}</strong>
          <div class="event-row-summary">${ev.summary}</div>
        </div>
        <div class="event-row-tokens">${ev.tokens}</div>
      </div>
    `;
  }).join("");

  listEl.querySelectorAll(".event-row").forEach(row => {
    row.addEventListener("click", () => {
      // @ts-ignore
      selectEvent(row.dataset.id);
    });
    row.addEventListener("keydown", (e) => {
      // @ts-ignore
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        // @ts-ignore
        selectEvent(row.dataset.id);
      }
    });
  });
}

function renderTimelineSVG() {
  const svg = document.getElementById("timeline-svg");
  if (!svg) return;

  const width = 960;
  const height = 180;
  const paddingX = 50;
  const baselineY = 90;

  // Start Date: Oct 15, 2021 | End Date: Jan 10, 2022 (87 days)
  const startTime = new Date("2021-10-15").getTime();
  const endTime = new Date("2022-01-10").getTime();
  const totalSpan = endTime - startTime;

  const filtered = getFilteredEvents();

  // Phase background bands
  const phases = [
    { start: "2021-10-15", end: "2021-11-05", label: "Ablations & Prep", color: "rgba(230, 240, 250, 0.4)" },
    { start: "2021-11-05", end: "2021-11-17", label: "0-10% (11.xx vs 12.xx)", color: "rgba(254, 237, 237, 0.5)" },
    { start: "2021-11-17", end: "2021-12-03", label: "10-27% (HW Crisis)", color: "rgba(254, 243, 230, 0.5)" },
    { start: "2021-12-03", end: "2021-12-16", label: "27-56% (Scale & Stability)", color: "rgba(230, 248, 240, 0.5)" },
    { start: "2021-12-16", end: "2022-01-10", label: "56-100% (Omicron SEV & Finish)", color: "rgba(240, 235, 255, 0.5)" },
  ];

  let phaseBandsSVG = phases.map(p => {
    const x1 = paddingX + ((new Date(p.start).getTime() - startTime) / totalSpan) * (width - 2 * paddingX);
    const x2 = paddingX + ((new Date(p.end).getTime() - startTime) / totalSpan) * (width - 2 * paddingX);
    return `
      <rect x="${x1}" y="20" width="${x2 - x1}" height="140" fill="${p.color}" />
      <text x="${x1 + 6}" y="36" font-size="11" font-weight="600" fill="#627d98">${p.label}</text>
    `;
  }).join("");

  // Category Colors
  const catColors = {
    sev: "#e02424",
    hardware: "#d97706",
    stability: "#dc2626",
    tune: "#2563eb",
    milestone: "#057a55",
    auto: "#0d9488"
  };

  // Markers
  let markersSVG = CHRONICLE_EVENTS.map(ev => {
    const t = new Date(ev.date).getTime();
    const x = paddingX + ((t - startTime) / totalSpan) * (width - 2 * paddingX);
    const isVisible = filtered.some(f => f.id === ev.id);
    const isSelected = ev.id === selectedEventId;
    const color = catColors[ev.category] || "#4b5563";
    const radius = isSelected ? 9 : 6;
    const opacity = isVisible ? (isSelected ? 1.0 : 0.85) : 0.15;
    
    // Stagger alternate markers up/down to avoid crowding
    const idx = CHRONICLE_EVENTS.indexOf(ev);
    const yOffset = (idx % 2 === 0) ? -24 : 24;
    const markerY = baselineY + yOffset;

    return `
      <g class="event-marker ${isSelected ? 'selected' : ''}" role="button" tabindex="0" aria-label="${ev.date}: ${ev.title}" data-id="${ev.id}" opacity="${opacity}">
        <line x1="${x}" y1="${baselineY}" x2="${x}" y2="${markerY}" stroke="${color}" stroke-width="${isSelected ? 2.5 : 1.2}" stroke-dasharray="${isSelected ? 'none' : '2,2'}" />
        <circle cx="${x}" cy="${markerY}" r="${radius}" fill="${color}" stroke="${isSelected ? '#fff' : 'none'}" stroke-width="2" />
        <text x="${x}" y="${markerY + (yOffset > 0 ? 16 : -10)}" font-size="10" font-weight="${isSelected ? '750' : '600'}" fill="${isSelected ? '#0e2b44' : '#486581'}" text-anchor="middle">
          ${ev.date.slice(5)}
        </text>
      </g>
    `;
  }).join("");

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <!-- Background Phases -->
    ${phaseBandsSVG}
    <!-- Main Axis Line -->
    <line x1="${paddingX}" y1="${baselineY}" x2="${width - paddingX}" y2="${baselineY}" stroke="#9fb3c8" stroke-width="2" />
    <!-- Markers -->
    ${markersSVG}
  `;

  // Attach click and keyboard handlers to SVG markers
  svg.querySelectorAll(".event-marker").forEach(g => {
    g.addEventListener("click", () => {
      // @ts-ignore
      selectEvent(g.dataset.id);
    });
    g.addEventListener("keydown", (e) => {
      // @ts-ignore
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        // @ts-ignore
        selectEvent(g.dataset.id);
      }
    });
  });
}

function renderLossChartSVG() {
  const svg = document.getElementById("loss-chart-svg");
  if (!svg) return;

  const width = 560;
  const height = 240;
  const padL = 45;
  const padR = 25;
  const padT = 30;
  const padB = 40;

  const maxStep = 75000;
  const minLoss = 2.0;
  const maxLoss = 12.0;

  const toX = (step) => padL + (step / maxStep) * (width - padL - padR);
  const toY = (loss) => padB + (1 - (loss - minLoss) / (maxLoss - minLoss)) * (height - padT - padB);

  // Grid lines
  let gridLines = "";
  for (let l = 3; l <= 11; l += 2) {
    const y = toY(l);
    gridLines += `
      <line x1="${padL}" y1="${y}" x2="${width - padR}" y2="${y}" stroke="#e4e7eb" stroke-dasharray="3,3" />
      <text x="${padL - 8}" y="${y + 4}" font-size="10" fill="#829ab1" text-anchor="end">${l}.0</text>
    `;
  }

  // X axis labels
  const stepLabels = [0, 20000, 40000, 60000, 75000];
  let xLabels = stepLabels.map(s => {
    const x = toX(s);
    return `<text x="${x}" y="${height - 15}" font-size="10" fill="#829ab1" text-anchor="middle">${s > 0 ? (s/1000) + 'k' : '0'}</text>`;
  }).join("");

  // Path 11.xx (Diverged)
  let d11 = "";
  LOSS_DATA.steps.forEach((s, i) => {
    const val = LOSS_DATA.exp11[i];
    if (val !== null) {
      d11 += `${d11 === "" ? "M" : "L"} ${toX(s).toFixed(1)} ${toY(val).toFixed(1)} `;
    }
  });

  // Path 12.xx (Converged)
  let d12 = "";
  LOSS_DATA.steps.forEach((s, i) => {
    const val = LOSS_DATA.exp12[i];
    if (val !== null) {
      d12 += `${d12 === "" ? "M" : "L"} ${toX(s).toFixed(1)} ${toY(val).toFixed(1)} `;
    }
  });

  // Path GPT-3 Reference
  let dGpt = "";
  LOSS_DATA.steps.forEach((s, i) => {
    const val = LOSS_DATA.gpt3[i];
    if (val !== null) {
      dGpt += `${dGpt === "" ? "M" : "L"} ${toX(s).toFixed(1)} ${toY(val).toFixed(1)} `;
    }
  });

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <!-- Grid -->
    ${gridLines}
    ${xLabels}
    <text x="${width/2}" y="${height - 2}" font-size="10" fill="#627d98" text-anchor="middle">Training Steps (2M tokens/step)</text>
    <text x="12" y="${height/2}" font-size="10" fill="#627d98" text-anchor="middle" transform="rotate(-90 12 ${height/2})">Validation Loss</text>
    
    <!-- Lines -->
    <path d="${dGpt}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4,4" />
    <path d="${d11}" fill="none" stroke="#ef4444" stroke-width="2.5" />
    <path d="${d12}" fill="none" stroke="#0284c7" stroke-width="2.8" />

    <!-- Exploded 11.xx point -->
    <circle cx="${toX(6500)}" cy="${toY(5.4)}" r="5" fill="#ef4444" stroke="#fff" stroke-width="2" />
    <text x="${toX(6500) + 8}" y="${toY(5.4) - 8}" font-size="10" font-weight="700" fill="#dc2626">11.xx Aborted (Stagnated/NaN)</text>

    <!-- 12.xx final point -->
    <circle cx="${toX(75000)}" cy="${toY(2.42)}" r="5" fill="#0284c7" stroke="#fff" stroke-width="2" />
    <text x="${toX(75000) - 10}" y="${toY(2.42) - 10}" font-size="10" font-weight="700" fill="#0369a1" text-anchor="end">12.xx Final: 2.42</text>
  `;
}

// Interactive Hardware & Cost Failure Simulator
function initSimulator() {
  const gpusInput = document.getElementById("sim-gpus");
  const failRateInput = document.getElementById("sim-fail-rate");
  const mttrInput = document.getElementById("sim-mttr");
  const costPerHourInput = document.getElementById("sim-cost");

  function update() {
    // @ts-ignore
    const gpus = parseInt(gpusInput.value, 10);
    // @ts-ignore
    const failRatePerDay = parseFloat(failRateInput.value);
    // @ts-ignore
    const mttrHours = parseFloat(mttrInput.value) / 60;
    // @ts-ignore
    const gpuCost = parseFloat(costPerHourInput.value);

    // Target FLOPs: 4.30e23, at 1024 A100s ~ 33 effective days (792 compute hours)
    const baselineGpuHours = 1024 * 792;
    const requiredDays = (baselineGpuHours / gpus) / 24;

    const totalFailures = Math.round(failRatePerDay * requiredDays);
    const downtimeHours = totalFailures * mttrHours;
    const totalElapsedDays = requiredDays + (downtimeHours / 24);
    const downtimePct = (downtimeHours / (totalElapsedDays * 24)) * 100;
    const totalClusterCost = (gpus * totalElapsedDays * 24) * gpuCost;
    const wastedCost = (gpus * downtimeHours) * gpuCost;

    // Update labels
    const gpusVal = document.getElementById("sim-val-gpus");
    const failVal = document.getElementById("sim-val-fail");
    const mttrVal = document.getElementById("sim-val-mttr");
    const costVal = document.getElementById("sim-val-cost");

    if (gpusVal) gpusVal.textContent = `${gpus.toLocaleString()} GPUs`;
    if (failVal) failVal.textContent = `${failRatePerDay.toFixed(1)} / day`;
    // @ts-ignore
    if (mttrVal) mttrVal.textContent = `${mttrInput.value} min`;
    if (costVal) costVal.textContent = `$${gpuCost.toFixed(2)}/hr`;

    // Update metrics
    const outDays = document.getElementById("res-days");
    const outRestarts = document.getElementById("res-restarts");
    const outDowntime = document.getElementById("res-downtime");
    const outCost = document.getElementById("res-cost");
    const outWasted = document.getElementById("res-wasted");

    if (outDays) outDays.textContent = `${totalElapsedDays.toFixed(1)} days`;
    if (outRestarts) outRestarts.textContent = `${totalFailures} restarts`;
    if (outDowntime) outDowntime.textContent = `${downtimePct.toFixed(1)}% (${Math.round(downtimeHours)}h)`;
    if (outCost) outCost.textContent = `$${(totalClusterCost / 1e6).toFixed(2)}M`;
    if (outWasted) outWasted.textContent = `$${(wastedCost / 1e3).toFixed(1)}k wasted`;
  }

  [gpusInput, failRateInput, mttrInput, costPerHourInput].forEach(inp => {
    if (inp) inp.addEventListener("input", update);
  });

  update();
}

// Start once DOM is ready
if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initChronicles);
  } else {
    initChronicles();
  }
}
