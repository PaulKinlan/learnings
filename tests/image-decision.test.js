import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BENCHMARK_META,
  JEV_IMAGE_BENCH_DATA,
  filterImageBench,
  sortImageBench
} from '../site/decision-models/image-bench-data.js';
import {
  PRESETS,
  validateImageSpec,
  decideImage,
  renderImageAnswers
} from '../site/decision-models/image-lab.js';
import { gate } from '../site/decision-models/core.js';

test('JevImageBench v0.1.5: dataset integrity & metadata', () => {
  assert.equal(BENCHMARK_META.name, 'JevImageBench');
  assert.equal(BENCHMARK_META.version, '0.1.5');
  assert.equal(BENCHMARK_META.totalSystems, 50);
  assert.equal(JEV_IMAGE_BENCH_DATA.length, 50, 'Must contain exactly 50 ranked systems');

  // Verify ranks are strictly 1..50
  const ranks = JEV_IMAGE_BENCH_DATA.map(m => m.rank);
  for (let i = 1; i <= 50; i++) {
    assert.ok(ranks.includes(i), `Rank ${i} must exist`);
  }
  const uniqueRanks = new Set(ranks);
  assert.equal(uniqueRanks.size, 50, 'All ranks must be unique');
});

test('JevImageBench v0.1.5: top Jev-class models and baselines scores match benchmark', () => {
  const find = name => JEV_IMAGE_BENCH_DATA.find(m => m.name.toLowerCase() === name.toLowerCase());

  // Top Jev-class models
  const wity = find('Wity-1');
  assert.ok(wity, 'Wity-1 must exist');
  assert.equal(wity.capabilityScore, 85.7);
  assert.equal(wity.access, 'Closed API');

  const jpt = find('JPT-9B');
  assert.ok(jpt, 'JPT-9B must exist');
  assert.equal(jpt.capabilityScore, 83.6);
  assert.equal(jpt.baseModel, 'Qwen3.5-9B');
  assert.equal(jpt.access, 'Open Weights');

  const imajev4b = find('Imajev-4B');
  assert.ok(imajev4b, 'Imajev-4B must exist');
  assert.equal(imajev4b.capabilityScore, 82.1);
  assert.equal(imajev4b.baseModel, 'Qwen3.5-4B');

  const shisa = find('shisa-de-1');
  assert.ok(shisa, 'shisa-de-1 must exist');
  assert.equal(shisa.capabilityScore, 82.1);
  assert.equal(shisa.baseModel, 'Gemma 4 26B-A4B');

  const neohorse = find('NeoHorse Jev 4B');
  assert.ok(neohorse, 'NeoHorse Jev 4B must exist');
  assert.equal(neohorse.capabilityScore, 82.0);
  assert.equal(neohorse.baseModel, 'NeoHorse-1-4B');

  const imajev9b = find('imajev 9B');
  assert.ok(imajev9b, 'imajev 9B must exist');
  assert.equal(imajev9b.capabilityScore, 82.0);
  assert.equal(imajev9b.baseModel, 'Qwen3.5-9B');

  const rune = find('Surogate Rune 26B-A4B v3');
  assert.ok(rune, 'Surogate Rune 26B-A4B v3 must exist');
  assert.equal(rune.capabilityScore, 81.3);
  assert.equal(rune.baseModel, 'gemma-4-26B-A4B-it');

  const visualJev = find('Visual-Jev 4B Answer-SFT');
  assert.ok(visualJev, 'Visual-Jev 4B Answer-SFT must exist');
  assert.equal(visualJev.capabilityScore, 74.8);
  assert.equal(visualJev.baseModel, 'Qwen3-VL-4B-Instruct');

  const imajev2b = find('imajev 2B');
  assert.ok(imajev2b, 'imajev 2B must exist');
  assert.equal(imajev2b.capabilityScore, 74.1);
  assert.equal(imajev2b.baseModel, 'Qwen3.5-2B');

  const mapika = find('Mapika decider-2b-vision');
  assert.ok(mapika, 'Mapika decider-2b-vision must exist');
  assert.equal(mapika.capabilityScore, 67.1);
  assert.equal(mapika.baseModel, 'Qwen3.5-2B');

  const jevify = find('Jevify Qwen3-VL-2B');
  assert.ok(jevify, 'Jevify Qwen3-VL-2B must exist');
  assert.equal(jevify.capabilityScore, 67.1);
  assert.equal(jevify.baseModel, 'Qwen3-VL-2B');

  const glance = find('Glance');
  assert.ok(glance, 'Glance must exist');
  assert.equal(glance.capabilityScore, 66.2);
  assert.equal(glance.baseModel, 'frozen Qwen3-VL-4B');

  const jevVision8b = find('Jev-Vision 8B');
  assert.ok(jevVision8b, 'Jev-Vision 8B must exist');
  assert.equal(jevVision8b.capabilityScore, 63.7);

  const jevSpatial = find('jev-spatial');
  assert.ok(jevSpatial, 'jev-spatial must exist');
  assert.equal(jevSpatial.capabilityScore, 62.5);
  assert.equal(jevSpatial.baseModel, 'Molmo2-ER');

  const qevi = find('Qevi-2B');
  assert.ok(qevi, 'Qevi-2B must exist');
  assert.equal(qevi.capabilityScore, 42.6);

  const jevision = find('JEVision');
  assert.ok(jevision, 'JEVision must exist');
  assert.equal(jevision.capabilityScore, 39.9);

  const playjev = find('PlayJev 0.8B');
  assert.ok(playjev, 'PlayJev 0.8B must exist');
  assert.equal(playjev.capabilityScore, 24.4);

  // Baselines outside budget
  const gpt56 = find('GPT-5.6 Luna');
  assert.ok(gpt56, 'GPT-5.6 Luna must exist');
  assert.equal(gpt56.capabilityScore, 93.8);

  const geminiFlashLite = find('Gemini 3.1 Flash Lite');
  assert.ok(geminiFlashLite, 'Gemini 3.1 Flash Lite must exist');
  assert.equal(geminiFlashLite.capabilityScore, 87.9);

  const gpt6 = find('GPT-6 Luna');
  assert.ok(gpt6, 'GPT-6 Luna must exist');
  assert.equal(gpt6.capabilityScore, 84.7);

  const geminiFlash = find('Gemini 3.8 Flash');
  assert.ok(geminiFlash, 'Gemini 3.8 Flash must exist');
  assert.equal(geminiFlash.capabilityScore, 80.7);
});

test('JevImageBench v0.1.5: schema and value ranges for all 50 entries', () => {
  for (const m of JEV_IMAGE_BENCH_DATA) {
    assert.ok(typeof m.rank === 'number' && m.rank >= 1 && m.rank <= 50, `Rank invalid: ${m.name}`);
    assert.ok(typeof m.name === 'string' && m.name.trim().length > 0, `Name invalid: ${m.name}`);
    assert.ok(typeof m.capabilityScore === 'number' && m.capabilityScore >= 0 && m.capabilityScore <= 100, `Score invalid: ${m.name}`);
    assert.ok(typeof m.intelligence === 'number' && m.intelligence >= 0 && m.intelligence <= 100, `Intel invalid: ${m.name}`);
    assert.ok(typeof m.calibration === 'number' && m.calibration >= 0 && m.calibration <= 100, `Calibration invalid: ${m.name}`);
    assert.ok(typeof m.speed === 'number' && m.speed > 0, `Speed invalid: ${m.name}`);
    assert.ok(typeof m.cost === 'number' && m.cost >= 0, `Cost invalid: ${m.name}`);
    assert.ok(['Open Weights', 'Closed API'].includes(m.access), `Access invalid: ${m.name}`);
    assert.ok(m.link.startsWith('http://') || m.link.startsWith('https://'), `Link invalid: ${m.name}`);
    assert.ok(['Jev-Class Vision', 'Baseline Multimodal'].includes(m.category), `Category invalid: ${m.name}`);
  }
});

test('JevImageBench filtering and sorting helpers', () => {
  // Access filtering
  const openOnly = filterImageBench(JEV_IMAGE_BENCH_DATA, { access: 'open' });
  assert.ok(openOnly.length > 0 && openOnly.every(m => m.access === 'Open Weights'));

  const closedOnly = filterImageBench(JEV_IMAGE_BENCH_DATA, { access: 'closed' });
  assert.ok(closedOnly.length > 0 && closedOnly.every(m => m.access === 'Closed API'));

  // Query filtering
  const qwenQuery = filterImageBench(JEV_IMAGE_BENCH_DATA, { query: 'qwen' });
  assert.ok(qwenQuery.length >= 5);
  assert.ok(qwenQuery.every(m => m.name.toLowerCase().includes('qwen') || m.baseModel.toLowerCase().includes('qwen')));

  // Category filtering
  const baselines = filterImageBench(JEV_IMAGE_BENCH_DATA, { category: 'baseline' });
  assert.ok(baselines.every(m => m.category === 'Baseline Multimodal'));

  // Sorting
  const sortedBySpeed = sortImageBench(JEV_IMAGE_BENCH_DATA, 'speed', true);
  for (let i = 0; i < sortedBySpeed.length - 1; i++) {
    assert.ok(sortedBySpeed[i].speed <= sortedBySpeed[i + 1].speed, 'Must be sorted ascending by speed');
  }

  const sortedByCost = sortImageBench(JEV_IMAGE_BENCH_DATA, 'cost', true);
  for (let i = 0; i < sortedByCost.length - 1; i++) {
    assert.ok(sortedByCost[i].cost <= sortedByCost[i + 1].cost, 'Must be sorted ascending by cost');
  }
});

test('Image decision schema validation: reject bad specs and payloads', () => {
  const validImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

  // Valid spec
  assert.doesNotThrow(() => {
    validateImageSpec({
      title: 'Valid test',
      questions: {
        q1: { type: 'noul', instructions: 'Is this an image?' }
      }
    }, validImage);
  });

  // Missing or non-string image
  assert.throws(() => {
    validateImageSpec({
      title: 'Valid test',
      questions: { q1: { type: 'noul', instructions: 'Is this an image?' } }
    }, '');
  }, /valid image/i);

  assert.throws(() => {
    validateImageSpec({
      title: 'Valid test',
      questions: { q1: { type: 'noul', instructions: 'Is this an image?' } }
    }, 'https://malicious.com/not-a-data-url.png');
  }, /valid image/i);

  // Missing title
  assert.throws(() => {
    validateImageSpec({
      title: '   ',
      questions: { q1: { type: 'noul', instructions: 'Test' } }
    }, validImage);
  }, /non-empty title/i);

  // Missing questions
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {}
    }, validImage);
  }, /at least one typed question/i);

  // Invalid question ID
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {
        'Bad-ID!': { type: 'noul', instructions: 'Test' }
      }
    }, validImage);
  }, /invalid question id/i);

  // Choice requires >= 2 criteria
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {
        pick: { type: 'choice', instructions: 'Choose', criteria: { only_one: 'desc' } }
      }
    }, validImage);
  }, /at least 2 named criteria/i);

  // Score requires 2..10 criteria
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {
        score: { type: 'score', instructions: 'Score', criteria: ['level 0'] }
      }
    }, validImage);
  }, /2 to 10 ordered criteria levels/i);
});

test('Decision readout formulas and probability distribution properties', async () => {
  const dummyImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

  for (const [key, preset] of Object.entries(PRESETS)) {
    const res = await decideImage({
      spec: preset.spec,
      imagePayload: dummyImage,
      engine: 'client'
    });

    assert.ok(res.data && res.data.answers, `Preset ${key} must return answers`);
    assert.ok(res.elapsed >= 0, `Preset ${key} must report elapsed ms`);

    for (const [qid, q] of Object.entries(preset.spec.questions)) {
      const a = res.data.answers[qid];
      assert.ok(a, `Answer for ${qid} in preset ${key} must exist`);
      assert.equal(a.type, q.type);

      if (q.type === 'noul') {
        assert.ok(a.noul >= 0 && a.noul <= 1, `Noul probability must be [0,1]`);
        assert.ok(a.confidence >= 0 && a.confidence <= 1, `Confidence must be [0,1]`);
        const g = gate(a, 0.8);
        assert.ok(['positive', 'negative', 'review'].includes(g));
      } else if (q.type === 'choice') {
        const optionKeys = Object.keys(q.criteria);
        assert.ok(optionKeys.includes(a.choice), `Selected choice must be in criteria`);
        assert.ok(a.confidence >= 0 && a.confidence <= 1, `Confidence must be [0,1]`);

        // Check probability distribution sums to 1.0
        const pSum = Object.values(a.probabilities).reduce((x, y) => x + y, 0);
        assert.ok(Math.abs(pSum - 1.0) < 1e-3, `Choice probabilities must sum to 1.0 (got ${pSum})`);

        const g = gate(a, 0.8);
        assert.ok(['accept', 'review'].includes(g));
      } else if (q.type === 'score') {
        assert.ok(a.score >= 0 && a.score <= q.criteria.length - 1, `Score must be within level indices`);
        assert.ok(a.confidence >= 0 && a.confidence <= 1, `Confidence must be [0,1]`);

        // Check probabilities sum to 1.0
        const pSum = Object.values(a.probabilities).reduce((x, y) => x + y, 0);
        assert.ok(Math.abs(pSum - 1.0) < 1e-3, `Score probabilities must sum to 1.0 (got ${pSum})`);

        const g = gate(a, 0.8);
        assert.ok(['accept', 'review'].includes(g));
      }
    }
  }
});

test('Preset visual test cases integrity', () => {
  const expectedPresets = ['ui', 'invoice', 'security', 'navigation'];
  for (const id of expectedPresets) {
    const p = PRESETS[id];
    assert.ok(p, `Preset ${id} must exist`);
    assert.ok(typeof p.name === 'string');
    assert.ok(typeof p.description === 'string');
    assert.ok(typeof p.draw === 'function');
    assert.ok(p.spec && p.spec.questions);
    assert.ok(p.answers);
  }
});
