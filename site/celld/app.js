// @ts-check
/**
 * celld & Durable Object Fleet Architecture — Simulation Engine & Interactive UI
 *
 * Models:
 * 1. S3 CAS Ownership & Monotonic Epoch Fencing (`cells/<cell>/ownership.json` + `ltx/e<epoch>/`)
 * 2. `fleet` (peer LTX streaming + fsync) vs `bucket` (S3 per-tx PUT) durability & RPO=0 failover
 * 3. Load balancing, peer tunnel routing (`X-Cell-Proxied: 1`), and hot-cell `503` queueing
 * 4. Fleet-wide Durable Object usage accounting & querying (Naive Central DO vs Credit Leases + Alarm Rollups)
 */

export class CasEpochSimulator {
  constructor() {
    this.reset();
  }

  reset() {
    this.bucketRecord = {
      cell: "tenant_acme",
      owner: "node-a",
      epoch: 1,
      etag: "etag-v1",
      maxTxid: 1,
      ltxFiles: ["cells/tenant_acme/ltx/e1/00000001-00000001.ltx"]
    };
    this.nodes = {
      "node-a": { id: "node-a", role: "owner", localEpoch: 1, localTxid: 1, peerReplicatedTxid: 1 },
      "node-b": { id: "node-b", role: "replica", localEpoch: 1, localTxid: 1, peerReplicatedTxid: 1 },
      "node-c": { id: "node-c", role: "standby", localEpoch: 1, localTxid: 0, peerReplicatedTxid: 0 }
    };
    this.logs = [
      "[boot] S3 PutObject If-None-Match:* -> cells/tenant_acme/ownership.json {owner:'node-a', epoch:1, etag:'etag-v1'}"
    ];
    return this.snapshot();
  }

  commitTransaction(durability = "fleet") {
    const ownerId = this.bucketRecord.owner;
    const ownerNode = this.nodes[ownerId];
    ownerNode.localTxid += 1;
    const tx = ownerNode.localTxid;
    const ep = ownerNode.localEpoch;
    const ltxPath = `cells/tenant_acme/ltx/e${ep}/0000000${tx}-0000000${tx}.ltx`;

    if (durability === "fleet") {
      // Replicate LTX frame to peer replica in ~1.8ms; async S3 upload
      for (const n of Object.values(this.nodes)) {
        if (n.id !== ownerId && n.role !== "fenced") {
          n.peerReplicatedTxid = tx;
          n.localEpoch = ep;
        }
      }
      this.bucketRecord.maxTxid = tx;
      this.bucketRecord.ltxFiles.push(ltxPath);
      this.logs.unshift(`[tx #${tx}] ${ownerId} committed SQLite tx -> streamed LTX to peer replica + disk fsync (1.8ms, RPO=0)`);
    } else {
      this.bucketRecord.maxTxid = tx;
      this.bucketRecord.ltxFiles.push(ltxPath);
      this.logs.unshift(`[tx #${tx}] ${ownerId} committed SQLite tx -> synchronous S3 PutObject ${ltxPath} (42.0ms, RPO=0)`);
    }
    return this.snapshot();
  }

  triggerPartitionTakeover() {
    const oldOwner = this.bucketRecord.owner;
    const newOwner = oldOwner === "node-a" ? "node-b" : "node-a";
    const nextEpoch = this.bucketRecord.epoch + 1;
    const nextEtag = `etag-v${nextEpoch}`;

    // Old owner becomes a zombie still thinking it owns the previous epoch
    this.nodes[oldOwner].role = "fenced";
    this.nodes[newOwner].role = "owner";
    this.nodes[newOwner].localEpoch = nextEpoch;
    this.nodes[newOwner].localTxid = this.bucketRecord.maxTxid;

    this.bucketRecord.owner = newOwner;
    this.bucketRecord.epoch = nextEpoch;
    this.bucketRecord.etag = nextEtag;

    this.logs.unshift(
      `[CAS Takeover] ${oldOwner} lease expired! ${newOwner} executed S3 PutObject If-Match -> bumped epoch=${nextEpoch} (${nextEtag}) and replayed LTX up to tx #${this.bucketRecord.maxTxid}.`
    );
    return this.snapshot();
  }

  attemptZombieWrite() {
    const fencedNode = Object.values(this.nodes).find((n) => n.role === "fenced");
    if (!fencedNode) {
      this.logs.unshift(`[Notice] No zombie node currently exists. Trigger a Partition & CAS Takeover first.`);
      return { rejected: false, state: this.snapshot() };
    }
    const staleEpoch = fencedNode.localEpoch;
    const activeEpoch = this.bucketRecord.epoch;
    this.logs.unshift(
      `[FENCED 412] Zombie ${fencedNode.id} attempted write under stale prefix ltx/e${staleEpoch}/ and If-Match old ETag -> REJECTED! Readers only hydrate ltx/e${activeEpoch}/.`
    );
    return { rejected: true, staleEpoch, activeEpoch, state: this.snapshot() };
  }

  snapshot() {
    return {
      bucketRecord: { ...this.bucketRecord, ltxFiles: [...this.bucketRecord.ltxFiles] },
      nodes: structuredClone(this.nodes),
      logs: [...this.logs]
    };
  }
}

/**
 * Computes load-balancing, peer-proxying, and hot-cell queueing metrics for a celld fleet.
 */
export function simulateLoadBalancing({
  nodes = 4,
  totalRps = 2400,
  hotCellSharePct = 35,
  routingMode = "random-l7", // "random-l7" | "affinity-header"
  durability = "fleet" // "fleet" | "bucket"
} = {}) {
  const writeLatencyMs = durability === "fleet" ? 2.0 : 38.0;
  // Single-writer cell max concurrency throughput before queueing/503
  const singleCellMaxRps = Math.round(1000 / writeLatencyMs) * 1.6;
  const hotCellRps = totalRps * (hotCellSharePct / 100);

  // Under random L7 routing, (nodes - 1) / nodes requests land on a non-owner node and tunnel over HTTP/2
  const proxiedRatio = routingMode === "affinity-header" ? 0.03 : (nodes - 1) / nodes;
  const proxiedRps = Math.round(totalRps * proxiedRatio);
  const tunnelOverheadMs = proxiedRatio * 0.9;
  const avgLatencyMs = writeLatencyMs + tunnelOverheadMs;

  const utilizationRatio = hotCellRps / singleCellMaxRps;
  const hotCell503Pct = utilizationRatio <= 1.0
    ? 0
    : Math.min(95, ((utilizationRatio - 1.0) / utilizationRatio) * 100);

  return {
    nodes,
    totalRps,
    hotCellRps: Math.round(hotCellRps),
    singleCellMaxRps: Math.round(singleCellMaxRps),
    proxiedRatio,
    proxiedRps,
    avgLatencyMs,
    hotCell503Pct,
    status: hotCell503Pct > 0 ? "hot-cell-throttled" : "healthy"
  };
}

/**
 * Compares Naive Per-Request Central Billing DO RPCs against Credit Leases + Write-Behind Alarm Batching.
 */
export function calculateFleetAccounting({
  fleetRps = 10000,
  activeCells = 2500,
  leaseBlockSize = 500,
  alarmIntervalSec = 30
} = {}) {
  const centralDoMaxRps = 800; // Single-threaded Durable Object RPC ceiling

  // Pattern A: Naive Synchronous Central Billing DO RPC on every event
  const naiveCentralRps = fleetRps;
  const naiveDroppedPct = naiveCentralRps <= centralDoMaxRps
    ? 0
    : ((naiveCentralRps - centralDoMaxRps) / naiveCentralRps) * 100;
  const naiveAddedLatencyMs = naiveCentralRps <= centralDoMaxRps ? 6.5 : 185.0;

  // Pattern B: Credit Leases + Write-Behind Alarm Rollups
  const leaseRefillRps = fleetRps / Math.max(1, leaseBlockSize);
  const alarmFlushRps = activeCells / Math.max(1, alarmIntervalSec);
  const optimizedCentralRps = leaseRefillRps + alarmFlushRps;
  const rpcReductionPct = ((naiveCentralRps - optimizedCentralRps) / Math.max(1, naiveCentralRps)) * 100;

  // Monthly RPC volume (30 days = 2,592,000 seconds)
  const secondsPerMonth = 2592000;
  const naiveMonthlyMillions = (naiveCentralRps * secondsPerMonth) / 1e6;
  const optimizedMonthlyMillions = (optimizedCentralRps * secondsPerMonth) / 1e6;

  return {
    fleetRps,
    activeCells,
    leaseBlockSize,
    alarmIntervalSec,
    naiveCentralRps: Math.round(naiveCentralRps),
    naiveDroppedPct,
    naiveAddedLatencyMs,
    optimizedCentralRps: Number(optimizedCentralRps.toFixed(1)),
    rpcReductionPct,
    naiveMonthlyMillions: Math.round(naiveMonthlyMillions),
    optimizedMonthlyMillions: Math.round(optimizedMonthlyMillions)
  };
}

// ============================================================================
// INTERACTIVE DOM WIRING
// ============================================================================

const $ = (id) => document.getElementById(id);

export function initCelldApp() {
  if (typeof document === "undefined") return;

  // 1. CAS & Epoch Fencing Simulator
  const casSim = new CasEpochSimulator();
  const durSelect = /** @type {HTMLSelectElement} */ ($("cas-durability"));

  function renderCas() {
    const snap = casSim.snapshot();
    const cardsEl = $("cas-nodes");
    if (cardsEl) {
      cardsEl.replaceChildren();
      for (const node of Object.values(snap.nodes)) {
        const card = document.createElement("div");
        card.className = `node-card ${node.role === "owner" ? "owner" : node.role === "fenced" ? "fenced" : ""}`;
        const badge = document.createElement("span");
        badge.className = "node-badge";
        badge.textContent = node.role.toUpperCase();
        const title = document.createElement("h4");
        title.textContent = node.id;
        const meta = document.createElement("div");
        meta.className = "small";
        meta.textContent = `Local Epoch: e${node.localEpoch} · SQLite TXID: #${node.localTxid} · Replicated TXID: #${node.peerReplicatedTxid}`;
        card.append(badge, title, meta);
        cardsEl.appendChild(card);
      }
    }

    const bucketEl = $("cas-bucket-state");
    if (bucketEl) {
      bucketEl.textContent =
        `S3 cells/${snap.bucketRecord.cell}/ownership.json → owner="${snap.bucketRecord.owner}", epoch=${snap.bucketRecord.epoch}, ETag="${snap.bucketRecord.etag}", latest LTX=${snap.bucketRecord.ltxFiles[snap.bucketRecord.ltxFiles.length - 1]}`;
    }

    const logEl = $("cas-log");
    if (logEl) {
      logEl.textContent = snap.logs.join("\n");
    }
  }

  if ($("btn-cas-commit")) {
    $("btn-cas-commit").addEventListener("click", () => {
      casSim.commitTransaction(durSelect ? durSelect.value : "fleet");
      renderCas();
    });
    $("btn-cas-partition").addEventListener("click", () => {
      casSim.triggerPartitionTakeover();
      renderCas();
    });
    $("btn-cas-zombie").addEventListener("click", () => {
      casSim.attemptZombieWrite();
      renderCas();
    });
    $("btn-cas-reset").addEventListener("click", () => {
      casSim.reset();
      renderCas();
    });
    renderCas();
  }

  // 2. Load Balancing & Hot-Cell Simulator
  const lbNodes = /** @type {HTMLInputElement} */ ($("lb-nodes"));
  const lbRps = /** @type {HTMLInputElement} */ ($("lb-rps"));
  const lbHot = /** @type {HTMLInputElement} */ ($("lb-hot"));
  const lbRoute = /** @type {HTMLSelectElement} */ ($("lb-route"));
  const lbDur = /** @type {HTMLSelectElement} */ ($("lb-dur"));

  function updateLb() {
    if (!lbNodes) return;
    $("lb-nodes-val").textContent = `${lbNodes.value} nodes`;
    $("lb-rps-val").textContent = `${Number(lbRps.value).toLocaleString()} req/s`;
    $("lb-hot-val").textContent = `${lbHot.value}%`;

    const res = simulateLoadBalancing({
      nodes: Number(lbNodes.value),
      totalRps: Number(lbRps.value),
      hotCellSharePct: Number(lbHot.value),
      routingMode: lbRoute.value,
      durability: lbDur.value
    });

    $("lb-out-proxied").textContent = `${res.proxiedRps.toLocaleString()} req/s (${(res.proxiedRatio * 100).toFixed(0)}%)`;
    $("lb-out-lat").textContent = `${res.avgLatencyMs.toFixed(1)} ms`;
    $("lb-out-cap").textContent = `${res.singleCellMaxRps.toLocaleString()} req/s`;
    const dropEl = $("lb-out-503");
    dropEl.textContent = `${res.hotCell503Pct.toFixed(1)}% 503 Busy`;
    dropEl.className = `metric-card-val ${res.hotCell503Pct > 0 ? "danger" : "good"}`;
  }

  [lbNodes, lbRps, lbHot, lbRoute, lbDur].forEach((el) => {
    if (el) el.addEventListener("input", updateLb);
  });
  updateLb();

  // 3. Fleet Usage Accounting & Querying Simulator
  const accRps = /** @type {HTMLInputElement} */ ($("acc-rps"));
  const accCells = /** @type {HTMLInputElement} */ ($("acc-cells"));
  const accLease = /** @type {HTMLInputElement} */ ($("acc-lease"));

  function updateAccounting() {
    if (!accRps) return;
    $("acc-rps-val").textContent = `${Number(accRps.value).toLocaleString()} req/s`;
    $("acc-cells-val").textContent = `${Number(accCells.value).toLocaleString()} DOs`;
    $("acc-lease-val").textContent = `${accLease.value} credits`;

    const res = calculateFleetAccounting({
      fleetRps: Number(accRps.value),
      activeCells: Number(accCells.value),
      leaseBlockSize: Number(accLease.value),
      alarmIntervalSec: 30
    });

    $("acc-out-naive-rps").textContent = `${res.naiveCentralRps.toLocaleString()} RPC/s`;
    $("acc-out-naive-drop").textContent = `${res.naiveDroppedPct.toFixed(1)}% throttled`;
    $("acc-out-opt-rps").textContent = `${res.optimizedCentralRps} RPC/s`;
    $("acc-out-saving").textContent = `${res.rpcReductionPct.toFixed(2)}% reduction`;
    $("acc-out-monthly").textContent = `${res.naiveMonthlyMillions.toLocaleString()}M → ${res.optimizedMonthlyMillions.toLocaleString()}M RPCs/mo`;
  }

  [accRps, accCells, accLease].forEach((el) => {
    if (el) el.addEventListener("input", updateAccounting);
  });
  updateAccounting();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initCelldApp);
  } else {
    initCelldApp();
  }
}
