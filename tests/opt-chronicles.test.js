import test from 'node:test';
import assert from 'node:assert/strict';
import { CHRONICLE_EVENTS, LOSS_DATA } from '../site/opt-chronicles/app.js';

test('OPT Chronicles: event dataset is structured and valid', () => {
  assert.ok(CHRONICLE_EVENTS.length >= 15, 'Should have comprehensive log of major events');

  for (const ev of CHRONICLE_EVENTS) {
    assert.ok(ev.id && ev.id.startsWith('e-'), `Invalid id: ${ev.id}`);
    assert.match(ev.date, /^\d{4}-\d{2}-\d{2}$/, `Invalid date: ${ev.date}`);
    assert.ok(['pre', 'phase-10', 'phase-27', 'phase-56', 'phase-100'].includes(ev.phase), `Invalid phase: ${ev.phase}`);
    assert.ok(['sev', 'hardware', 'stability', 'tune', 'milestone', 'auto'].includes(ev.category), `Invalid category: ${ev.category}`);
    assert.ok(ev.title && ev.title.length > 5, 'Title must be descriptive');
    assert.ok(ev.summary && ev.summary.length > 10, 'Summary must be descriptive');
    assert.ok(ev.quote && ev.quote.length > 15, 'Quote must carry verbatim excerpt');
    assert.ok(typeof ev.step === 'number' && ev.step >= 0, 'Step must be non-negative integer');
  }
});

test('OPT Chronicles: critical historical milestones are present', () => {
  const ids = CHRONICLE_EVENTS.map(e => e.id);
  assert.ok(ids.includes('e-plan-b'), 'Plan B must be recorded');
  assert.ok(ids.includes('e-dataset-glitch'), 'Dataset regex glitch must be recorded');
  assert.ok(ids.includes('e-run-11-launch'), 'Exp 11.0 launch must be recorded');
  assert.ok(ids.includes('e-relu-hotswap'), 'GeLU to ReLU hotswap must be recorded');
  assert.ok(ids.includes('e-run-12-pivot'), 'Exp 12.00 pivot must be recorded');
  assert.ok(ids.includes('e-thanksgiving-break'), 'Thanksgiving break hardware issues must be recorded');
  assert.ok(ids.includes('e-checkpoint-hangs'), '1.6TB checkpoint bottleneck must be recorded');
  assert.ok(ids.includes('e-omicron-cluster-deletion'), 'Omicron cluster deletion SEV must be recorded');
  assert.ok(ids.includes('e-automated-recovery-live'), 'Automated self-healing milestone must be recorded');
  assert.ok(ids.includes('e-run-complete'), '100% completion milestone must be recorded');
});

test('OPT Chronicles: loss trajectory datasets have aligned step lengths', () => {
  const len = LOSS_DATA.steps.length;
  assert.equal(LOSS_DATA.exp11.length, len, 'exp11 length must match steps');
  assert.equal(LOSS_DATA.exp12.length, len, 'exp12 length must match steps');
  assert.equal(LOSS_DATA.gpt3.length, len, 'gpt3 length must match steps');

  // Verify exp12 strictly decreases after step 500
  for (let i = 2; i < len; i++) {
    assert.ok(LOSS_DATA.exp12[i] < LOSS_DATA.exp12[i - 1], `exp12 should decrease at step ${LOSS_DATA.steps[i]}`);
  }
});
