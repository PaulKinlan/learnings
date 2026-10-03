import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getWasmBackend,
  jsMatmul,
  WGSL_GEMM_SHADER,
  getWebGPUStatus,
  webgpuMatmulAsync,
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
  generateTemporalSequenceBatch,
  MicroRNN,
  DeepResidualNetwork,
  BLOCK_REGISTRY,
  registerBlockType,
  analyzeArchitecturePipeline
} from '../site/neural-networks/engine.js';
import { HYPERPARAMETER_GUIDE, KERNEL_CODE_BLUEPRINTS } from '../site/neural-networks/curriculum.js';

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

test('MLPNetwork.traceStep and commitTrace produce node-level forward, backward, and weight update traces', () => {
  const mlp = new MLPNetwork([2, 4, 4, 1], { activation: 'gelu', seed: 101 });
  const traceTrain = mlp.traceStep([-0.55, 0.55], 1, { lr: 0.08, optimizer: 'adamw', mode: 'train' });

  assert.equal(traceTrain.nodesByLayer.length, 4, 'Should have 4 node layers (L0..L3)');
  assert.equal(traceTrain.nodesByLayer[0].length, 2);
  assert.equal(traceTrain.nodesByLayer[1].length, 4);
  assert.equal(traceTrain.nodesByLayer[2].length, 4);
  assert.equal(traceTrain.nodesByLayer[3].length, 1);
  assert.equal(traceTrain.microSteps.length, 9, '2-hidden-layer training trace has 9 micro-steps');
  assert.ok(traceTrain.lossAfter < traceTrain.lossBefore, `Single-sample update should reduce sample loss (${traceTrain.lossBefore} -> ${traceTrain.lossAfter})`);

  const committed = mlp.commitTrace(traceTrain);
  assert.equal(committed.step, 1);
  assert.ok(Math.abs(mlp.predictPoint(-0.55, 0.55) - traceTrain.predAfter) < 1e-5);

  const traceInfer = mlp.traceStep([-0.55, 0.55], 1, { mode: 'inference' });
  assert.equal(traceInfer.microSteps.length, 5, '2-hidden-layer inference trace has 5 micro-steps');
});

test('MicroRNN (Elman RNN vs Gated GRU) trains via BPTT and GRU retains long-horizon gradients', () => {
  const batch = generateTemporalSequenceBatch('memory_first', 10, 24, 303);
  const gru = new MicroRNN({ hiddenDim: 10, cellType: 'gru', seed: 1997 });
  const vanilla = new MicroRNN({ hiddenDim: 10, cellType: 'rnn', seed: 1997 });

  const gruInit = gru.trainEpoch(batch, 0.12);
  const vanInit = vanilla.trainEpoch(batch, 0.12);
  assert.equal(gruInit.temporalSteps.length, 10);
  assert.ok(
    gruInit.gradRetentionRatio > vanInit.gradRetentionRatio,
    `GRU should retain higher t=0 gradient ratio than Vanilla RNN (${gruInit.gradRetentionRatio} vs ${vanInit.gradRetentionRatio})`
  );

  let gruLast = gruInit;
  for (let e = 0; e < 30; e++) {
    gruLast = gru.trainEpoch(batch, 0.12);
  }
  assert.ok(gruLast.loss < gruInit.loss, `GRU BPTT loss should decrease (${gruInit.loss} -> ${gruLast.loss})`);
});

test('DeepResidualNetwork preserves Layer 1 gradient flow at 16 layers with Residual Highway + RMSNorm', () => {
  const ds = generate2DDataset('xor', 32, 77);
  const resNet = new DeepResidualNetwork({
    depth: 16,
    dim: 8,
    residual: true,
    norm: 'rmsnorm',
    initScheme: 'he',
    seed: 2015
  });
  const plainSmallNet = new DeepResidualNetwork({
    depth: 16,
    dim: 8,
    residual: false,
    norm: 'none',
    initScheme: 'small',
    seed: 2015
  });

  const resProfile = resNet.forwardAndBackward([-0.55, 0.55], 1);
  const plainProfile = plainSmallNet.forwardAndBackward([-0.55, 0.55], 1);

  assert.equal(resProfile.layerStats.length, 16);
  assert.ok(
    resProfile.layerStats[0].gradNorm > plainProfile.layerStats[0].gradNorm * 100,
    `16-layer ResNet Layer 1 grad (${resProfile.layerStats[0].gradNorm}) should be orders of magnitude stronger than plain unnormalized stack (${plainProfile.layerStats[0].gradNorm})`
  );

  const firstStep = resNet.trainStep(ds, 0.05);
  let lastStep = firstStep;
  for (let i = 0; i < 15; i++) {
    lastStep = resNet.trainStep(ds, 0.05);
  }
  assert.ok(lastStep.loss < firstStep.loss, `16-layer ResNet loss should decrease (${firstStep.loss} -> ${lastStep.loss})`);
});

test('WebGPU WGSL compute shader, fallback matmul, and curriculum blueprints are complete', async () => {
  assert.ok(WGSL_GEMM_SHADER.includes('@compute @workgroup_size(8, 8, 1)'));
  const A = new Float32Array([1, 2, 3, 4]);
  const B = new Float32Array([5, 6, 7, 8]);
  const C = await webgpuMatmulAsync(A, B, 2, 2, 2);
  assert.deepEqual(Array.from(C), [19, 22, 43, 50]);
  const status = getWebGPUStatus();
  assert.equal(status.checked, true);

  // Verify curriculum hyperparameter guide and all 8 kernel code blueprints (js, wasm, webgpu)
  assert.ok(HYPERPARAMETER_GUIDE.datasets.xor.title.includes('XOR'));
  assert.ok(HYPERPARAMETER_GUIDE.learningRate(0.05).title.includes('Sweet Spot'));
  for (const key of ['gemm', 'mlp_backprop', 'conv2d', 'rnn_bptt', 'deep_resnet', 'attention', 'diffusion', 'decision_head']) {
    const bp = KERNEL_CODE_BLUEPRINTS[key];
    assert.ok(bp, `Missing blueprint for ${key}`);
    assert.ok(bp.js.length > 50, `${key}.js should be populated`);
    assert.ok(bp.wasm.length > 50, `${key}.wasm should be populated`);
    assert.ok(bp.webgpu.length > 50, `${key}.webgpu should be populated`);
  }
});

