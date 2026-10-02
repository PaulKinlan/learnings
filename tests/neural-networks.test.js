import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getWasmBackend,
  jsMatmul,
  benchmarkBackends,
  setBackendMode,
  generate2DDataset,
  MLPNetwork,
  generateGlyphDataset,
  TinyCNN,
  generateSequenceBatch,
  MicroTransformer,
  SPRITE_TARGETS,
  MicroDDPM,
  MaskedTextDiffusion,
  DecisionPointerHead,
  BLOCK_REGISTRY,
  registerBlockType,
  analyzeArchitecturePipeline
} from '../site/neural-networks/engine.js';

test('Raw WebAssembly f32 GEMM kernel compiles in-memory and matches Float32Array GEMM', () => {
  const wb = getWasmBackend();
  assert.equal(wb.available, true, 'WebAssembly GEMM module should compile and instantiate');
  assert.ok(wb.byteLength > 100 && wb.byteLength < 300, 'WASM binary should be compact');

  const A = new Float32Array([1, 2, 3, 4, 5, 6]); // 2x3
  const B = new Float32Array([7, 8, 9, 1, 2, 3]); // 3x2
  const outWasm = wb.matmul(A, B, 2, 3, 2);
  const outJs = jsMatmul(A, B, 2, 3, 2);

  assert.deepEqual(Array.from(outWasm), [31, 19, 85, 55]);
  assert.deepEqual(Array.from(outWasm), Array.from(outJs));

  const bench = benchmarkBackends(32, 5);
  assert.ok(bench.maxDiff < 1e-4, `Expected WASM and JS GEMM to agree, maxDiff=${bench.maxDiff}`);
});

test('MLPNetwork trains via analytical backprop on 2D XOR and decreases loss', () => {
  setBackendMode('wasm');
  const ds = generate2DDataset('xor', 80, 42);
  const mlp = new MLPNetwork([2, 8, 8, 1], { activation: 'gelu', seed: 42 });

  const first = mlp.trainStep(ds, { lr: 0.08, optimizer: 'adamw' });
  let last = first;
  for (let i = 0; i < 60; i++) {
    last = mlp.trainStep(ds, { lr: 0.08, optimizer: 'adamw' });
  }
  assert.ok(last.loss < first.loss * 0.5, `Expected XOR loss to drop significantly (${first.loss} -> ${last.loss})`);
  assert.ok(last.accuracy >= 0.85, `Expected XOR accuracy >= 0.85, got ${last.accuracy}`);
});

test('TinyCNN trains spatial 3x3 kernels and classifies 8x8 glyphs', () => {
  const samples = generateGlyphDataset(6, 77);
  const cnn = new TinyCNN(2026);
  const first = cnn.trainEpoch(samples, 0.08);
  let last = first;
  for (let e = 0; e < 25; e++) {
    last = cnn.trainEpoch(samples, 0.08);
  }
  assert.ok(last.loss < first.loss, `Expected CNN loss to decrease (${first.loss} -> ${last.loss})`);
  assert.ok(last.accuracy >= 0.85, `Expected CNN accuracy >= 0.85, got ${last.accuracy}`);
});

test('MicroTransformer learns sequence reversal and produces valid attention weights', () => {
  const batch = generateSequenceBatch('reverse', 4, 24, 99);
  const tf = new MicroTransformer({ vocabSize: 8, seqLen: 4, dModel: 16, numHeads: 2, causal: false, seed: 314 });
  const first = tf.trainEpoch(batch, 0.12);
  let last = first;
  for (let e = 0; e < 35; e++) {
    last = tf.trainEpoch(batch, 0.12);
  }
  assert.ok(last.loss < first.loss, `Expected Transformer loss to decrease (${first.loss} -> ${last.loss})`);

  const fwd = tf.forward([0, 1, 2, 3]);
  assert.equal(fwd.attnWeights.length, 2);
  // Each row of attention weights must sum to 1.0
  for (let r = 0; r < 4; r++) {
    const rowSum = fwd.attnWeights[0].slice(r * 4, r * 4 + 4).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(rowSum - 1.0) < 1e-4, `Attention row sum must be 1.0, got ${rowSum}`);
  }
});

test('MicroDDPM (continuous pixel diffusion) and MaskedTextDiffusion train and denoise', () => {
  const ddpm = new MicroDDPM({ dim: 36, steps: 12, hiddenDim: 32, seed: 512 });
  const target = SPRITE_TARGETS.smiley;
  const first = ddpm.trainStep(target, 0.04, 12);
  let last = first;
  for (let i = 0; i < 30; i++) {
    last = ddpm.trainStep(target, 0.04, 12);
  }
  assert.ok(last.loss < first.loss, `Expected DDPM noise-prediction MSE to drop (${first.loss} -> ${last.loss})`);
  const traj = ddpm.sampleTrajectory(808);
  assert.equal(traj.length, 13);

  const textDiff = new MaskedTextDiffusion(909);
  const tFirst = textDiff.trainStep(null, 0.2);
  let tLast = tFirst;
  for (let i = 0; i < 30; i++) {
    tLast = textDiff.trainStep(null, 0.2);
  }
  assert.ok(tLast.loss < tFirst.loss, `Expected MaskedTextDiffusion loss to drop (${tFirst.loss} -> ${tLast.loss})`);
  const textTraj = textDiff.denoiseTrajectory(0);
  const finalTokens = textTraj[textTraj.length - 1].tokens;
  assert.ok(!finalTokens.includes('[MASK]'), 'Final denoised sequence must unmask all tokens');
});

test('DecisionPointerHead and extensible Block Registry work end-to-end', () => {
  const head = new DecisionPointerHead(8, 404);
  const stateVec = new Float32Array([1, 0, 0.5, -0.5, 0.2, 0.1, -0.2, 0.8]);
  const opts = [
    new Float32Array([1, 0, 0.5, -0.5, 0.2, 0.1, -0.2, 0.8]),
    new Float32Array([-1, 0.2, -0.5, 0.5, -0.2, 0, 0.2, -0.8]),
    new Float32Array([0, 0.1, 0, 0, 0, 0, 0, 0])
  ];
  const res = head.decide(stateVec, opts, { temperature: 0.7, gateThreshold: 0.4 });
  assert.equal(res.bestIdx, 0);
  assert.ok(res.confidence > 0 && res.confidence <= 1);
  assert.ok(['act', 'review'].includes(res.recommendation));

  registerBlockType({
    type: 'ssm_mamba',
    label: 'Selective State-Space (Mamba SSM)',
    category: 'sequence',
    description: 'Linear-time O(T·D) selective state-space scan.',
    computeStats(shape) {
      return { outShape: { ...shape }, params: 3 * shape.d * shape.d, flops: 6 * shape.t * shape.d * shape.d };
    }
  });
  assert.ok(BLOCK_REGISTRY.has('ssm_mamba'));

  const analysis = analyzeArchitecturePipeline(['embedding', 'layernorm', 'attention', 'ssm_mamba', 'mlp_block', 'decision_head']);
  assert.equal(analysis.stages.length, 6);
  assert.ok(analysis.totalParams > 0);
  assert.ok(analysis.totalFlops > 0);
});
