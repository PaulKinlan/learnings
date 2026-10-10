import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  CasEpochSimulator,
  simulateLoadBalancing,
  calculateFleetAccounting
} from '../site/celld/app.js';

test('celld CasEpochSimulator enforces monotonic epoch fencing and rejects stale zombie writes', () => {
  const sim = new CasEpochSimulator();
  const s0 = sim.snapshot();
  assert.equal(s0.bucketRecord.owner, 'node-a');
  assert.equal(s0.bucketRecord.epoch, 1);

  const s1 = sim.commitTransaction('fleet');
  assert.equal(s1.bucketRecord.maxTxid, 2);
  assert.equal(s1.nodes['node-b'].peerReplicatedTxid, 2);

  const s2 = sim.triggerPartitionTakeover();
  assert.equal(s2.bucketRecord.owner, 'node-b');
  assert.equal(s2.bucketRecord.epoch, 2);
  assert.equal(s2.nodes['node-a'].role, 'fenced');

  const zombieAttempt = sim.attemptZombieWrite();
  assert.equal(zombieAttempt.rejected, true);
  assert.equal(zombieAttempt.staleEpoch, 1);
  assert.equal(zombieAttempt.activeEpoch, 2);
});

test('celld simulateLoadBalancing models peer tunneling and bucket vs fleet hot-cell saturation', () => {
  const fleetRes = simulateLoadBalancing({
    nodes: 4,
    totalRps: 2000,
    hotCellSharePct: 25,
    routingMode: 'random-l7',
    durability: 'fleet'
  });
  assert.equal(fleetRes.proxiedRatio, 0.75);
  assert.equal(fleetRes.hotCell503Pct, 0);

  const bucketRes = simulateLoadBalancing({
    nodes: 4,
    totalRps: 2000,
    hotCellSharePct: 25,
    routingMode: 'random-l7',
    durability: 'bucket'
  });
  assert.ok(bucketRes.hotCell503Pct > 50, 'Bucket mode should saturate single hot cell at 500 req/s');
});

test('celld calculateFleetAccounting shows >98% cross-DO RPC reduction with credit leases and alarms', () => {
  const res = calculateFleetAccounting({
    fleetRps: 10000,
    activeCells: 2500,
    leaseBlockSize: 500,
    alarmIntervalSec: 30
  });
  assert.ok(res.naiveDroppedPct > 85, 'Naive central BillingDO should bottleneck at 10,000 req/s');
  assert.ok(res.rpcReductionPct > 98, `Expected >98% RPC reduction, got ${res.rpcReductionPct}%`);
});

test('celld index.html defines valid speculationrules for prefetching siblings and prerendering hub (learnings-bjx)', () => {
  const htmlPath = new URL('../site/celld/index.html', import.meta.url);
  const celldHtml = readFileSync(htmlPath, 'utf8');
  const specRulesMatch = celldHtml.match(/<script type="speculationrules">([\s\S]*?)<\/script>/);
  assert.ok(specRulesMatch, 'speculationrules script block must exist in site/celld/index.html');
  const rules = JSON.parse(specRulesMatch[1]);
  assert.ok(Array.isArray(rules.prefetch), 'rules must declare prefetch rules');
  assert.ok(Array.isArray(rules.prerender), 'rules must declare prerender rules');
  assert.equal(rules.prefetch[0].source, 'list');
  assert.deepEqual(rules.prefetch[0].urls, [
    '../neural-networks/',
    '../decision-models/',
    '../opt-chronicles/'
  ]);
  assert.equal(rules.prerender[0].source, 'list');
  assert.deepEqual(rules.prerender[0].urls, ['../index.html']);
});
