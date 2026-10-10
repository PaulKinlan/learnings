import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  BENCHMARK_META,
  JEV_IMAGE_BENCH_DATA,
  filterImageBench,
  sortImageBench
} from '../site/decision-models/image-bench-data.js';

const SOURCE_EXTRACT = JSON.parse(
  readFileSync(new URL('./fixtures/image-jev-bench-source-extract.json', import.meta.url), 'utf8')
);
import {
  PRESETS,
  validateImageSpec,
  decideImage,
  renderImageAnswers
} from '../site/decision-models/image-lab.js';
import { gate, validateAnswers, validateSpec } from '../site/decision-models/core.js';

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

test('JevImageBench v0.1.5: comprehensive 50-row mechanical validation against source extract fixture', () => {
  assert.ok(SOURCE_EXTRACT.metadata, 'Fixture metadata header must exist');
  assert.equal(SOURCE_EXTRACT.metadata.sourceUrl, 'https://benchmarkheaven.com/image-jev-bench');
  assert.equal(SOURCE_EXTRACT.metadata.version, '0.1.5');
  assert.equal(SOURCE_EXTRACT.metadata.totalRows, 50);
  assert.ok(SOURCE_EXTRACT.metadata.fetchDate, 'Fetch date must exist in header metadata');

  assert.equal(JEV_IMAGE_BENCH_DATA.length, 50, 'image-bench-data.js must contain exactly 50 rows');
  assert.equal(SOURCE_EXTRACT.rows.length, 50, 'Source extract fixture must contain exactly 50 rows');

  for (let i = 0; i < 50; i++) {
    const dataRow = JEV_IMAGE_BENCH_DATA[i];
    const fixtureRow = SOURCE_EXTRACT.rows[i];

    assert.equal(dataRow.rank, fixtureRow.rank, `Row ${i + 1} rank mismatch (${dataRow.name})`);
    assert.equal(dataRow.name, fixtureRow.name, `Row ${i + 1} name mismatch`);
    assert.equal(dataRow.capabilityScore, fixtureRow.capability, `Row ${i + 1} capability mismatch (${dataRow.name})`);
    assert.equal(dataRow.intelligence, fixtureRow.intelligence, `Row ${i + 1} intelligence mismatch (${dataRow.name})`);
    assert.equal(dataRow.calibration, fixtureRow.calibration, `Row ${i + 1} calibration mismatch (${dataRow.name})`);
    assert.equal(dataRow.speed, fixtureRow.latency, `Row ${i + 1} latency/speed mismatch (${dataRow.name})`);
    assert.equal(dataRow.cost, fixtureRow.cost, `Row ${i + 1} cost mismatch (${dataRow.name})`);
    assert.equal(dataRow.link, fixtureRow.link, `Row ${i + 1} link mismatch (${dataRow.name})`);
  }
});

test('JevImageBench v0.1.5: top Jev-class models and baselines scores match benchmark', () => {
  const find = name => JEV_IMAGE_BENCH_DATA.find(m => m.name.toLowerCase() === name.toLowerCase());

  // Top Jev-class models
  const wity = find('Wity-1');
  assert.ok(wity, 'Wity-1 must exist');
  assert.equal(wity.capabilityScore, 85.7);
  assert.equal(wity.intelligence, 83.7);
  assert.equal(wity.calibration, 87.7);
  assert.equal(wity.speed, 0.71);
  assert.equal(wity.cost, 0.0074);
  assert.equal(wity.access, 'Closed API');

  const jpt = find('JPT-9B');
  assert.ok(jpt, 'JPT-9B must exist');
  assert.equal(jpt.capabilityScore, 83.6);
  assert.equal(jpt.intelligence, 75.7);
  assert.equal(jpt.calibration, 91.5);
  assert.equal(jpt.speed, 0.46);
  assert.equal(jpt.cost, 0.053);
  assert.equal(jpt.baseModel, 'Qwen3.5-9B');
  assert.equal(jpt.access, 'Open Weights');

  const imajev4b = find('Imajev-4B');
  assert.ok(imajev4b, 'Imajev-4B must exist');
  assert.equal(imajev4b.capabilityScore, 82.1);
  assert.equal(imajev4b.intelligence, 73.8);
  assert.equal(imajev4b.calibration, 90.5);
  assert.equal(imajev4b.speed, 0.35);
  assert.equal(imajev4b.cost, 0.020);
  assert.equal(imajev4b.baseModel, 'Qwen3.5-4B');

  const shisa = find('shisa-de-1');
  assert.ok(shisa, 'shisa-de-1 must exist');
  assert.equal(shisa.capabilityScore, 82.1);
  assert.equal(shisa.intelligence, 72.4);
  assert.equal(shisa.calibration, 91.8);
  assert.equal(shisa.speed, 0.48);
  assert.equal(shisa.cost, 0.056);
  assert.equal(shisa.baseModel, 'Gemma 4 26B-A4B');

  const neohorse = find('NeoHorse Jev 4B');
  assert.ok(neohorse, 'NeoHorse Jev 4B must exist');
  assert.equal(neohorse.capabilityScore, 82.0);
  assert.equal(neohorse.intelligence, 72.9);
  assert.equal(neohorse.calibration, 91.2);
  assert.equal(neohorse.speed, 0.43);
  assert.equal(neohorse.cost, 0.041);
  assert.equal(neohorse.baseModel, 'NeoHorse-1-4B');

  const imajev9b = find('imajev 9B');
  assert.ok(imajev9b, 'imajev 9B must exist');
  assert.equal(imajev9b.capabilityScore, 82.0);
  assert.equal(imajev9b.intelligence, 74.7);
  assert.equal(imajev9b.calibration, 89.3);
  assert.equal(imajev9b.speed, 0.53);
  assert.equal(imajev9b.cost, 0.063);
  assert.equal(imajev9b.baseModel, 'Qwen3.5-9B');

  const rune = find('Surogate Rune 26B-A4B v3');
  assert.ok(rune, 'Surogate Rune 26B-A4B v3 must exist');
  assert.equal(rune.capabilityScore, 81.3);
  assert.equal(rune.intelligence, 74.9);
  assert.equal(rune.calibration, 87.8);
  assert.equal(rune.speed, 0.48);
  assert.equal(rune.cost, 0.055);
  assert.equal(rune.baseModel, 'gemma-4-26B-A4B-it');

  const visualJev = find('Visual-Jev 4B Answer-SFT');
  assert.ok(visualJev, 'Visual-Jev 4B Answer-SFT must exist');
  assert.equal(visualJev.capabilityScore, 74.8);
  assert.equal(visualJev.intelligence, 65.4);
  assert.equal(visualJev.calibration, 84.1);
  assert.equal(visualJev.speed, 0.35);
  assert.equal(visualJev.cost, 0.036);
  assert.equal(visualJev.baseModel, 'Qwen3-VL-4B-Instruct');

  const imajev2b = find('imajev 2B');
  assert.ok(imajev2b, 'imajev 2B must exist');
  assert.equal(imajev2b.capabilityScore, 74.1);
  assert.equal(imajev2b.intelligence, 57.7);
  assert.equal(imajev2b.calibration, 90.5);
  assert.equal(imajev2b.speed, 0.34);
  assert.equal(imajev2b.cost, 0.034);
  assert.equal(imajev2b.baseModel, 'Qwen3.5-2B');

  const mapika = find('Mapika decider-2b-vision BF16');
  assert.ok(mapika, 'Mapika decider-2b-vision BF16 must exist');
  assert.equal(mapika.capabilityScore, 67.1);
  assert.equal(mapika.intelligence, 52.8);
  assert.equal(mapika.calibration, 81.5);
  assert.equal(mapika.speed, 0.33);
  assert.equal(mapika.cost, 0.026);
  assert.equal(mapika.baseModel, 'Qwen3.5-2B-Base');

  const jevify = find('Jevify Qwen3-VL-2B T2');
  assert.ok(jevify, 'Jevify Qwen3-VL-2B T2 must exist');
  assert.equal(jevify.capabilityScore, 67.1);
  assert.equal(jevify.intelligence, 47.3);
  assert.equal(jevify.calibration, 86.8);
  assert.equal(jevify.speed, 0.27);
  assert.equal(jevify.cost, 0.023);
  assert.equal(jevify.baseModel, 'Qwen3-VL-2B-Instruct');

  const glance = find('Glance');
  assert.ok(glance, 'Glance must exist');
  assert.equal(glance.capabilityScore, 66.2);
  assert.equal(glance.intelligence, 59.4);
  assert.equal(glance.calibration, 73.0);
  assert.equal(glance.speed, 0.33);
  assert.equal(glance.cost, 0.030);
  assert.equal(glance.baseModel, 'Qwen3-VL-4B');

  const jevVision8b = find('Jev-Vision 8B');
  assert.ok(jevVision8b, 'Jev-Vision 8B must exist');
  assert.equal(jevVision8b.capabilityScore, 63.7);
  assert.equal(jevVision8b.intelligence, 64.2);
  assert.equal(jevVision8b.calibration, 63.1);
  assert.equal(jevVision8b.speed, 0.45);
  assert.equal(jevVision8b.cost, 0.046);

  const jevSpatial = find('jev-spatial');
  assert.ok(jevSpatial, 'jev-spatial must exist');
  assert.equal(jevSpatial.capabilityScore, 62.5);
  assert.equal(jevSpatial.intelligence, 52.0);
  assert.equal(jevSpatial.calibration, 73.0);
  assert.equal(jevSpatial.speed, 0.27);
  assert.equal(jevSpatial.cost, 0.022);
  assert.equal(jevSpatial.baseModel, 'Molmo2-ER');

  const qevi = find('Qevi-2B');
  assert.ok(qevi, 'Qevi-2B must exist');
  assert.equal(qevi.capabilityScore, 42.6);
  assert.equal(qevi.intelligence, 29.5);
  assert.equal(qevi.calibration, 55.7);
  assert.equal(qevi.speed, 0.29);
  assert.equal(qevi.cost, 0.024);

  const jevision = find('JEVision');
  assert.ok(jevision, 'JEVision must exist');
  assert.equal(jevision.capabilityScore, 39.9);
  assert.equal(jevision.intelligence, 35.4);
  assert.equal(jevision.calibration, 44.4);
  assert.equal(jevision.speed, 0.49);
  assert.equal(jevision.cost, 0.057);

  const playjev = find('PlayJev 0.8B');
  assert.ok(playjev, 'PlayJev 0.8B must exist');
  assert.equal(playjev.capabilityScore, 24.4);
  assert.equal(playjev.intelligence, 4.4);
  assert.equal(playjev.calibration, 59.4);
  assert.equal(playjev.speed, 0.49);
  assert.equal(playjev.cost, 0.023);
  assert.equal(playjev.link, 'https://huggingface.co/OmniJev/PlayJev-0.8B/blob/a7348002b1e159add7037d6d50812cd4db2d96e9/README.md');

  const omnijev9b = find('OmniJev-Qwen3.5-9B-v4');
  assert.ok(omnijev9b, 'OmniJev-Qwen3.5-9B-v4 must exist');
  assert.equal(omnijev9b.capabilityScore, 60.9);
  assert.equal(omnijev9b.intelligence, 52.2);
  assert.equal(omnijev9b.calibration, 69.6);
  assert.equal(omnijev9b.speed, 0.27);
  assert.equal(omnijev9b.cost, 0.022);

  // Baselines outside budget
  const gpt56 = find('GPT-5.6 Luna');
  assert.ok(gpt56, 'GPT-5.6 Luna must exist');
  assert.equal(gpt56.capabilityScore, 93.8);
  assert.equal(gpt56.intelligence, 91.8);
  assert.equal(gpt56.calibration, 95.8);
  assert.equal(gpt56.speed, 2.88);
  assert.equal(gpt56.cost, 0.35);

  const geminiFlashLite = find('Gemini 3.1 Flash Lite');
  assert.ok(geminiFlashLite, 'Gemini 3.1 Flash Lite must exist');
  assert.equal(geminiFlashLite.capabilityScore, 87.9);
  assert.equal(geminiFlashLite.intelligence, 84.3);
  assert.equal(geminiFlashLite.calibration, 91.4);
  assert.equal(geminiFlashLite.speed, 1.80);
  assert.equal(geminiFlashLite.cost, 0.39);

  const gpt6 = find('GPT-6 Luna (low)');
  assert.ok(gpt6, 'GPT-6 Luna (low) must exist');
  assert.equal(gpt6.capabilityScore, 84.7);
  assert.equal(gpt6.intelligence, 81.1);
  assert.equal(gpt6.calibration, 88.3);
  assert.equal(gpt6.speed, 2.91);
  assert.equal(gpt6.cost, 0.22);

  const geminiFlash = find('Gemini 3.8 Flash');
  assert.ok(geminiFlash, 'Gemini 3.8 Flash must exist');
  assert.equal(geminiFlash.capabilityScore, 80.7);
  assert.equal(geminiFlash.intelligence, 88.5);
  assert.equal(geminiFlash.calibration, 73.0);
  assert.equal(geminiFlash.speed, 4.83);
  assert.equal(geminiFlash.cost, 2.06);
});

test('JevImageBench v0.1.5: zero invented systems exist in dataset', () => {
  const inventedNames = [
    'EyeJev',
    'PicJev',
    'Claude 3.7',
    'SigLIP',
    'Vision-Kev',
    'PocketJev',
    'FastJev',
    'CompactJev',
    'JevSight',
    'TinyJev',
    'UltraLight'
  ];

  for (const m of JEV_IMAGE_BENCH_DATA) {
    for (const banned of inventedNames) {
      assert.ok(
        !m.name.toLowerCase().includes(banned.toLowerCase()),
        `Invented system "${banned}" must NOT exist in benchmark data (found: ${m.name})`
      );
    }
  }

  // Verify all 50 models strictly match the 50 known benchmark systems
  const expectedSystems = new Set([
    'GPT-5.6 Luna',
    'Gemini 3.1 Flash Lite',
    'Wity-1',
    'GPT-6 Luna (low)',
    'JPT-9B',
    'Bonsai-2-27B v2 PQ2_0',
    'Imajev-4B',
    'shisa-de-1',
    'NeoHorse Jev 4B',
    'imajev 9B',
    'Surogate Rune 26B-A4B v3',
    'Autoloops - Gemma 4 31B IT',
    'Gemini 3.8 Flash',
    'JevAny-27B RLCR',
    'Jevify Gemma 4 26B-A4B',
    'JevAny-27B SFT',
    'JPT-4B',
    'Jev-Omni',
    'Winnow-12B',
    'AutoJev-27B',
    'Visual-Jev 4B Answer-SFT',
    'CUA-S1 4B',
    'Visual-Jev generic',
    'imajev 2B',
    'Reflex 4B',
    'Standard One 8B',
    'OmniJev 4B',
    'djev-distill-v4',
    'Mapika decider-2b-vision BF16',
    'Jevify Qwen3-VL-2B T2',
    'djev-spark NVFP4',
    'Glance',
    'diffusiongemma-26b djev v10',
    'OmniJev 2B',
    'djev-dev BF16',
    'Jev-Vision 8B',
    'vjev-vision',
    'jev-spatial',
    'OmniJev-Qwen3.5-9B-v4',
    'Standard One 3B',
    'JPT-0.8B',
    'Jevify Gemma 4 E4B',
    'OmniJev 0.8B',
    'Gevva E2B multimodal',
    'Gevva E4B',
    'Qevi-2B',
    'JEVision',
    'OpenJev 4B NLI v5',
    'OpenJev 4B NLI v2',
    'PlayJev 0.8B'
  ]);

  assert.equal(JEV_IMAGE_BENCH_DATA.length, 50);
  for (const m of JEV_IMAGE_BENCH_DATA) {
    assert.ok(expectedSystems.has(m.name), `Unexpected model found: ${m.name}`);
  }
});

test('JevImageBench v0.1.5: range validation matches benchmark bounds', () => {
  const speeds = JEV_IMAGE_BENCH_DATA.map(m => m.speed);
  const costs = JEV_IMAGE_BENCH_DATA.map(m => m.cost);

  const minSpeed = Math.min(...speeds);
  const maxSpeed = Math.max(...speeds);
  assert.equal(minSpeed, 0.26, 'Min speed should be 0.26s (Jevify Gemma 4 E4B)');
  assert.equal(maxSpeed, 4.83, 'Max speed should be 4.83s (Gemini 3.8 Flash)');

  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);
  assert.equal(minCost, 0.0074, 'Min cost should be $0.0074/1k (Wity-1)');
  assert.equal(maxCost, 2.06, 'Max cost should be $2.06/1k (Gemini 3.8 Flash)');

  // Jev-class subset bounds (38 systems)
  const jevClass = JEV_IMAGE_BENCH_DATA.filter(m => m.category === 'Jev-Class Vision');
  assert.equal(jevClass.length, 38, 'Must have exactly 38 Jev-class models');
  const jevSpeeds = jevClass.map(m => m.speed);
  const jevCosts = jevClass.map(m => m.cost);
  assert.ok(Math.min(...jevSpeeds) >= 0.26 && Math.max(...jevSpeeds) <= 0.71);
  assert.ok(Math.min(...jevCosts) >= 0.0074 && Math.max(...jevCosts) <= 0.063);
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
    if (m.link) {
      assert.ok(m.link.startsWith('http://') || m.link.startsWith('https://'), `Link invalid: ${m.name}`);
    } else {
      assert.equal(typeof m.link, 'string', `Link must be a string: ${m.name}`);
    }
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
  }, /Title must be non-empty text/);

  // Missing questions
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {}
    }, validImage);
  }, /At least one question is required/);

  // Invalid question ID
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {
        'Bad-ID!': { type: 'noul', instructions: 'Test' }
      }
    }, validImage);
  }, /Use simple lowercase question IDs/);

  // Choice requires >= 2 criteria
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {
        pick: { type: 'choice', instructions: 'Choose', criteria: { only_one: 'desc' } }
      }
    }, validImage);
  }, /Choice requires 2–255 named options/);

  // Score requires 2..10 criteria
  assert.throws(() => {
    validateImageSpec({
      title: 'Title',
      questions: {
        score: { type: 'score', instructions: 'Score', criteria: ['level 0'] }
      }
    }, validImage);
  }, /Score requires 2–10 ordered levels/);
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

test('Simulated demo mode: carries explicit simulation disclosure in metadata and UI', async () => {
  const dummyImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

  const res = await decideImage({
    spec: PRESETS.ui.spec,
    imagePayload: dummyImage,
    engine: 'client'
  });

  // Verify explicit simulation disclosure in metadata
  assert.equal(res.isSimulation, true, 'isSimulation must be true');
  assert.equal(res.data.simulated, true, 'data.simulated must be true');
  assert.ok(res.source.includes('Simulated Fixture') || res.source.includes('Demo Mode'));
  assert.ok(res.data.model.includes('Simulated Fixture') || res.data.model.includes('Demo Mode'));

  // Must NOT claim in-browser model weights or calibrated execution
  assert.ok(!res.data.model.includes('Imajev-4B'), 'Must not claim Imajev-4B attribution for simulation fixture');
  assert.ok(!res.data.model.includes('in-browser'), 'Must not claim in-browser model weights');
  assert.ok(!res.source.includes('in-tab'), 'Must not claim in-tab model execution');

  // Verify exact simulation notice text
  const expectedNotice = 'Notice: UI and schema validation simulation. Connect a local vision server (e.g. llama.cpp / vLLM / Kev serve) or multimodal API endpoint below to run actual model weights.';
  assert.equal(res.simulationNotice, expectedNotice);
  assert.equal(res.data.simulationNotice, expectedNotice);

  // Verify UI rendering of simulation disclosure badge
  const dummyTarget = answerTarget();

  renderImageAnswers(dummyTarget, res, 0.8);
  const badge = dummyTarget.children.find(c => c.id === 'simulation-badge');
  assert.ok(badge, 'Simulation disclosure badge must be rendered in UI answers');
  assert.ok(badge.textContent.includes('Notice: UI and schema validation simulation.'));
  assert.ok(badge.textContent.includes('Connect a local vision server'));
});

const DUMMY_IMAGE = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

test('image lab clears the prior run on JSON, validation, provider and timeout failures (learnings-63h)', async () => {
  const { serve } = await import('../scripts/serve.mjs');
  const { launch } = await import('./lib/cdp.mjs');
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: ['/usr/bin/chromium'] });
    await page.goto(url + 'decision-models/image-lab.html');
    await page.waitFor(() => document.querySelector('#preview-image').src.startsWith('data:image/'));
    const run = async prefix => {
      await page.click('#run-decision');
      await page.waitFor(p => document.querySelector('#status').textContent.startsWith(p), { args: [prefix], label: prefix });
    };
    const emptyResult = async prefix => {
      await run(prefix);
      const state = await page.evaluate(() => ({
        status: document.querySelector('#status').textContent,
        answers: document.querySelector('#answers').textContent,
        count: document.querySelectorAll('#answers .answer').length,
        rawHidden: document.querySelector('#raw-details').hidden
      }));
      assert.equal(state.answers, '', `${state.status}: previous answers must be gone`);
      assert.equal(state.count, 0);
      assert.equal(state.rawHidden, true);
    };
    await run('Decision complete');
    const first = await page.evaluate(() => document.querySelector('#answers').textContent);
    assert.match(first, /is_blockedp\(yes\): 0\.965/);
    assert.match(first, /actionSelected: retry/);
    assert.match(first, /Confidence score: 0\.912/);
    assert.match(first, /severityExpected score: 1\.94/);

    await page.type('#questions', '{');
    await emptyResult('JSON Error:');
    await page.click('#preset-ui');
    await run('Decision complete');
    await page.type('#questions', JSON.stringify({ questions: {} }));
    await emptyResult('Error: Title must be non-empty text.');

    const selectEngine = async value => page.evaluate(v => {
      const select = document.querySelector('#engine');
      select.value = v;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }, value);
    const previousSuccess = async () => {
      await selectEngine('client');
      await page.click('#preset-ui');
      await run('Decision complete');
      assert.equal(await page.evaluate(() => document.querySelector('#answers').textContent), first);
      await selectEngine('local');
    };
    await previousSuccess();
    await page.evaluate(() => { window.fetch = async () => ({ ok: true, json: async () => ({ model: 'bad-provider', answers: {} }) }); });
    await emptyResult('Error:'); // validateAnswers refuses missing provider answers
    await page.evaluate(() => { document.querySelector('#threshold').dispatchEvent(new Event('input')); });
    assert.equal(await page.evaluate(() => document.querySelector('#answers').textContent), '', 'threshold must not restore the stale result');

    await previousSuccess();
    await page.evaluate(() => { window.fetch = async () => { throw new TypeError('fixture fetch failure'); }; });
    await emptyResult('Error: fixture fetch failure');
    await previousSuccess();
    await page.evaluate(() => { window.fetch = async () => { throw new DOMException('fixture timeout', 'TimeoutError'); }; });
    await emptyResult('Error: fixture timeout');

    await selectEngine('client');
    await page.click('#preset-ui');
    await run('Decision complete');
    assert.equal(await page.evaluate(() => document.querySelector('#answers').textContent), first, 'successful fixture readout unchanged after failures');
  } finally {
    if (page) await page.close();
    await new Promise((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
  }
});

test('image lab cuts off a provider that never answers at the 45s boundary (learnings-eku)', async () => {
  const { serve } = await import('../scripts/serve.mjs');
  const { launch } = await import('./lib/cdp.mjs');
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: ['/usr/bin/chromium'] });
    await page.goto(url + 'decision-models/image-lab.html');
    await page.waitFor(() => document.querySelector('#preview-image').src.startsWith('data:image/'));

    // A successful run first, so the timeout case has real answers it could fail to clear.
    await page.click('#run-decision');
    await page.waitFor(() => document.querySelector('#status').textContent.startsWith('Decision complete'));
    assert.match(await page.evaluate(() => document.querySelector('#answers').textContent), /actionSelected: retry/);

    // The run handler's own signal is what is under test, so the interval it asks for is recorded
    // and only the wait is compressed. The provider then accepts the request and never answers, and
    // the abort that ends it is the platform's (the injected signal's reason), not a thrown fixture.
    await page.evaluate(() => {
      window.__timeoutIntervals = [];
      const realTimeout = AbortSignal.timeout.bind(AbortSignal);
      AbortSignal.timeout = ms => { window.__timeoutIntervals.push(ms); return realTimeout(50); };
      window.fetch = (resource, init) => new Promise((resolve, reject) => {
        // No signal means no boundary: with the wiring reverted this never settles, which is the hang.
        if (!init.signal) return;
        init.signal.addEventListener('abort', () => reject(init.signal.reason));
      });
    });
    await page.evaluate(() => {
      const select = document.querySelector('#engine');
      select.value = 'local';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });

    await page.click('#run-decision');
    await page.waitFor(() => window.__timeoutIntervals.length > 0, { label: 'the run handler to wire a timeout signal' });
    assert.deepEqual(
      await page.evaluate(() => window.__timeoutIntervals),
      [45000],
      "the run must wire the sibling decision lab's 45s boundary"
    );
    await page.waitFor(() => document.querySelector('#status').textContent.startsWith('Error'), { label: 'timeout error' });

    const state = await page.evaluate(() => ({
      status: document.querySelector('#status').textContent,
      answers: document.querySelector('#answers').textContent,
      count: document.querySelectorAll('#answers .answer').length,
      rawHidden: document.querySelector('#raw-details').hidden,
      runDisabled: document.querySelector('#run-decision').disabled
    }));
    assert.equal(state.status, 'Error: Request cancelled or timed out. No automatic retry.');
    assert.equal(state.answers, '', 'a timed-out run must not leave the previous readout on screen');
    assert.equal(state.count, 0);
    assert.equal(state.rawHidden, true);
    assert.equal(state.runDisabled, false, 'the timed-out run must terminate and re-enable the run button');
  } finally {
    if (page) await page.close();
    await new Promise((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
  }
});

test('decideImage cuts off a provider that never answers for direct callers that pass no signal (learnings-ndz)', async (t) => {
  // The lab wires its own signal, but decideImage is exported and the eku review flagged the gap:
  // a caller that forgets it hangs exactly as the lab did. This drives decideImage directly with
  // no signal. The interval asked for is recorded and only the wait is compressed, so the 45s
  // boundary is exercised rather than a thrown fixture, matching the eku regression's convention.
  const intervals = [];
  const realTimeout = AbortSignal.timeout.bind(AbortSignal);
  AbortSignal.timeout = ms => { intervals.push(ms); return realTimeout(50); };
  t.after(() => { AbortSignal.timeout = realTimeout; });

  const originalFetch = globalThis.fetch;
  // The provider accepts the request and never answers; the abort that ends it is the default
  // signal's own reason, not a fixture. With no signal wired this never settles, which is the hang.
  globalThis.fetch = (resource, init) => new Promise((resolve, reject) => {
    if (!init.signal) return;
    init.signal.addEventListener('abort', () => reject(init.signal.reason));
  });
  t.after(() => { globalThis.fetch = originalFetch; });

  // Bounded, so the reverted mutation reports a failed assertion instead of hanging the suite.
  let hangTimer;
  const hung = new Promise(resolve => {
    hangTimer = setTimeout(() => resolve({ state: 'hung' }), 5000);
    hangTimer.unref();
  });
  t.after(() => clearTimeout(hangTimer));

  const outcome = await Promise.race([
    decideImage({
      spec: PRESETS.ui.spec,
      imagePayload: DUMMY_IMAGE,
      engine: 'local',
      endpoint: 'http://127.0.0.1:8009/v1/systemone',
      model: 'jpt-9b'
    }).then(
      () => ({ state: 'resolved' }),
      err => ({ state: 'rejected', message: err.message })
    ),
    hung
  ]);

  assert.deepEqual(
    outcome,
    { state: 'rejected', message: 'Request cancelled or timed out. No automatic retry.' },
    "a direct caller that passes no signal must be cut off on decideImage's own boundary, not hang"
  );
  assert.deepEqual(intervals, [45000], "decideImage's default must be the sibling decision lab's 45s boundary");
});

// renderImageAnswers only needs replaceChildren/append, so a recording stub stands in for the DOM
// and the rendered tree can be asserted on directly.
const answerTarget = () => ({
  children: [],
  replaceChildren() { this.children = []; },
  append(...items) { this.children.push(...items); }
});

// A contract-shaped vision provider response for PRESETS.ui.spec (noul + choice + score).
// The image lab posts the reader's image to a local or hosted provider and renders whatever
// comes back, so this is the payload every provider-path test starts from and mutates.
const visionResponse = () => ({
  model: 'stub-vision-server',
  answers: {
    is_blocked: { type: 'noul', noul: 0.93, confidence: 0.93 },
    action: {
      type: 'choice',
      choice: 'retry',
      confidence: 0.88,
      probabilities: { retry: 0.88, cancel: 0.06, dismiss: 0.04, unclear: 0.02 }
    },
    severity: {
      type: 'score',
      score: 1.9,
      confidence: 0.9,
      probabilities: { '0': 0.02, '1': 0.08, '2': 0.9 }
    }
  }
});

// Stub the network so a test controls exactly what the provider returns. Returns the URLs called.
function stubProvider(payload, t) {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, status: 200, json: async () => structuredClone(payload) };
  };
  t.after(() => { globalThis.fetch = original; });
  return calls;
}

const runEngine = (engine, spec = PRESETS.ui.spec) => decideImage({
  spec,
  imagePayload: DUMMY_IMAGE,
  engine,
  endpoint: 'http://127.0.0.1:8009/v1/systemone',
  key: 'synthetic-test-not-a-real-key',
  model: engine === 'local' ? 'jpt-9b' : 'wity-1'
});

test('Local vision endpoint uses the core.js loopback contract (learnings-8u9)', async (t) => {
  const dummyImage = DUMMY_IMAGE;
  const spec = PRESETS.ui.spec;
  const run = endpoint => decideImage({ spec, imagePayload: dummyImage, engine: 'local', endpoint, model: 'jpt-9b' });

  const original = globalThis.fetch;
  let calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    // Contract-shaped, because the response is now gated by core.js validateAnswers:
    // this test pins the endpoint contract, so it must not trip the answer gate.
    return { ok: true, json: async () => visionResponse() };
  };
  t.after(() => { globalThis.fetch = original; });

  // Refused, each naming the constraint that failed. Under the old check only u.hostname was tested,
  // so every one of these was POSTed the base64 image and the questions.
  const refused = [
    ['http://127.0.0.1:8009/admin', /path must be exactly \/v1\/systemone, got \/admin/],
    ['http://127.0.0.1:8009/v1/systemone/../admin', /path must be exactly \/v1\/systemone, got \/v1\/admin/],
    ['http://127.0.0.1:8009/v1/systemone?to=https://evil.test', /must not carry a query string/],
    ['http://127.0.0.1:8009/v1/systemone#frag', /must not carry a fragment/],
    ['http://user:secret@127.0.0.1:8009/v1/systemone', /must not embed credentials/],
    ['http://localhost.evil.test/v1/systemone', /host must be exactly localhost, 127\.0\.0\.1 or \[::1\], got localhost\.evil\.test/],
    ['http://127.0.0.1.evil.test/v1/systemone', /host must be exactly localhost, 127\.0\.0\.1 or \[::1\], got 127\.0\.0\.1\.evil\.test/],
    ['https://evil.test/v1/systemone', /host must be exactly localhost/],
    ['javascript:alert(1)', /protocol must be http or https/],
    ['not a url at all', /Enter a loopback URL/]
  ];
  for (const [endpoint, reason] of refused) {
    await t.test(`refuses ${endpoint} and says which constraint failed`, async () => {
      calls = [];
      await assert.rejects(run(endpoint), reason);
      assert.deepEqual(calls, [], 'a refused endpoint must not be sent the image');
    });
  }

  // Accepted-set, pinned as a regression test instead of left implicit.
  // Any loopback port is accepted deliberately: the endpoint is reader-supplied so a reader can point the page
  // at their own local vision server (Ollama 11434, llama.cpp 8080, LM Studio 1234, vLLM 8000).
  // https on loopback is accepted too - a hostile local TLS server fails on the certificate before any image is posted.
  const accepted = [
    ['http://127.0.0.1:8009/v1/systemone', 'http://127.0.0.1:8009/v1/systemone'],
    ['http://localhost:11434/v1/systemone', 'http://localhost:11434/v1/systemone'],
    ['http://127.0.0.1:8080/v1/systemone', 'http://127.0.0.1:8080/v1/systemone'],
    ['http://[::1]:1234/v1/systemone', 'http://[::1]:1234/v1/systemone'],
    ['https://127.0.0.1:8009/v1/systemone', 'https://127.0.0.1:8009/v1/systemone']
  ];
  for (const [endpoint, href] of accepted) {
    await t.test(`accepts ${endpoint} and POSTs to ${href}`, async () => {
      calls = [];
      const res = await run(endpoint);
      assert.equal(calls.length, 1, 'the documented endpoint shape must still reach the local server');
      assert.equal(calls[0].url, href, 'must POST to the normalised href');
      assert.equal(calls[0].options.method, 'POST');
      assert.equal(JSON.parse(calls[0].options.body).image, dummyImage, 'the base64 image is the payload at risk');
      assert.deepEqual(Object.keys(JSON.parse(calls[0].options.body).questions), Object.keys(spec.questions));
      assert.equal(res.source, 'Local Vision Decision Server');
    });
  }
});

// The two engines below post the reader's image to a provider and used to hand `await res.json()`
// straight to renderImageAnswers. They now pass it through the same validator core.js decide() uses,
// so there is one convention for externally derived answers rather than an image-specific second one.
test('local and api engines refuse malformed provider answers with the core.js message (learnings-ytm)', async (t) => {
  const refused = [
    ['an out-of-set choice', () => { const r = visionResponse(); r.answers.action.choice = 'rm_rf'; return r; }, /Out-of-set choice: action\./],
    ['a confidence outside zero to one', () => { const r = visionResponse(); r.answers.action.confidence = 99; return r; }, /Invalid confidence: action\./],
    ['a non-finite probability', () => { const r = visionResponse(); r.answers.action.probabilities.retry = Infinity; return r; }, /Invalid distribution: action\./],
    ['a distribution over the wrong option set', () => { const r = visionResponse(); r.answers.action.probabilities = { retry: 1 }; return r; }, /Invalid distribution: action\./],
    ['a distribution that does not sum to one', () => { const r = visionResponse(); r.answers.action.probabilities = { retry: 0.9, cancel: 0.9, dismiss: 0.9, unclear: 0.9 }; return r; }, /Distribution does not sum to one: action\./],
    ['a non-numeric probability', () => { const r = visionResponse(); r.answers.is_blocked.noul = 'very likely'; return r; }, /Invalid probability: is_blocked\./],
    ['a wrong answer type', () => { const r = visionResponse(); r.answers.is_blocked.type = 'choice'; return r; }, /Missing or wrong answer type: is_blocked\./],
    ['a missing answer', () => { const r = visionResponse(); delete r.answers.severity; return r; }, /Missing or wrong answer type: severity\./],
    ['a score outside the level range', () => { const r = visionResponse(); r.answers.severity.score = 99; return r; }, /Invalid score: severity\./],
    ['an empty answers object', () => ({ model: 'stub-vision-server', answers: {} }), /Missing or wrong answer type: is_blocked\./],
    ['no answers object at all', () => ({ model: 'stub-vision-server' }), /Provider did not return an answers object\./]
  ];

  for (const engine of ['local', 'api']) {
    for (const [name, build, message] of refused) {
      await t.test(`${engine} engine refuses ${name}`, async (st) => {
        const calls = stubProvider(build(), st);
        await assert.rejects(runEngine(engine), message);
        assert.equal(calls.length, 1, 'the refusal is about the response, so the request still reaches the provider');
      });
    }
  }
});

// The other half of the gate: a well-formed provider response must still render in full.
test('local and api engines still render a valid provider response (learnings-ytm)', async (t) => {
  const payload = visionResponse();
  const expected = {
    local: 'http://127.0.0.1:8009/v1/systemone',
    api: 'https://api.typesafe.ai/v1/systemone'
  };

  for (const engine of ['local', 'api']) {
    await t.test(`${engine} engine renders every question with its gate verdict`, async (st) => {
      const calls = stubProvider(payload, st);
      const res = await runEngine(engine);

      assert.equal(calls.length, 1, 'a valid run still makes exactly one provider request');
      assert.equal(calls[0].url, expected[engine]);
      assert.deepEqual(res.data, payload, 'a valid response passes through unchanged');
      assert.notEqual(res.isSimulation, true, 'a provider run must not claim to be a simulation');
      assert.equal(res.source, engine === 'local' ? 'Local Vision Decision Server' : 'TypeSafe API (wity-1)');

      const target = answerTarget();
      renderImageAnswers(target, res, 0.8);
      assert.equal(target.children.length, 3, 'one rendered answer per question, and no simulation badge');
      assert.equal(target.children.find(c => c.id === 'simulation-badge'), undefined);
      assert.deepEqual(
        target.children.map(c => c.textContent.match(/Gate verdict: \w+/)[0]),
        ['Gate verdict: positive', 'Gate verdict: accept', 'Gate verdict: accept']
      );
      assert.ok(target.children[0].textContent.includes('p(yes): 0.930'));
      assert.ok(target.children[1].textContent.includes('Selected: retry'));
      assert.ok(target.children[2].textContent.includes('Expected score: 1.90'));
    });
  }
});

// The demo engine is deliberately NOT gated: it is the disclosed offline path that must work with no
// server at all, so a fixture must never be refused by a provider-side validator. This pins that it
// still renders, and that the fixtures stay contract-shaped so both readouts share one convention.
test('simulated demo path stays ungated and renders offline (learnings-ytm)', async (t) => {
  const customSpec = {
    title: 'Custom image triage',
    questions: {
      blocked: { type: 'noul', instructions: 'Is the user blocked?' },
      act: { type: 'choice', instructions: 'Which action?', criteria: { go: 'Proceed', stop: 'Halt' } },
      lvl: { type: 'score', instructions: 'How severe?', criteria: ['low', 'mid', 'high'] }
    }
  };
  const runs = [...Object.entries(PRESETS).map(([key, p]) => [`preset ${key}`, p.spec]), ['custom image with no preset match', customSpec]];

  for (const [name, spec] of runs) {
    await t.test(`demo engine renders ${name} with no server and no validation refusal`, async () => {
      const res = await decideImage({ spec, imagePayload: DUMMY_IMAGE, engine: 'client' });
      assert.equal(res.isSimulation, true);
      assert.equal(res.data.simulated, true);

      const target = answerTarget();
      renderImageAnswers(target, res, 0.8);
      const badge = target.children.find(c => c.id === 'simulation-badge');
      assert.ok(badge, 'the simulation disclosure badge still renders');
      assert.equal(target.children.length, Object.keys(spec.questions).length + 1, 'disclosure badge plus one answer per question');
      assert.deepEqual(Object.keys(res.data.answers), Object.keys(spec.questions), 'every question is answered offline');
    });
  }

  for (const [key, preset] of Object.entries(PRESETS)) {
    assert.doesNotThrow(
      () => validateAnswers({ answers: preset.answers }, preset.spec.questions),
      `preset ${key} fixture answers must stay contract-shaped, so the demo and a validated provider run read the same`
    );
  }
});

test('simulated demo path: a custom spec reusing a preset title with different options produces synthesised answers matching ITS OWN options, NOT the preset ones (learnings-mnw)', async (t) => {
  const customSpec = {
    title: PRESETS.ui.spec.title, // Reusing the title
    description: 'Custom description',
    questions: {
      action: {
        type: 'choice',
        instructions: 'Which button?',
        criteria: {
          new_option: 'A new option',
          another: 'Another option'
        }
      }
    }
  };

  const res = await decideImage({
    spec: customSpec,
    imagePayload: DUMMY_IMAGE,
    engine: 'client'
  });

  assert.equal(res.isSimulation, true);
  const actionAnswer = res.data.answers.action;
  assert.ok(actionAnswer);
  
  // The answer must be drawn from ITS OWN options, not 'retry', 'cancel' etc.
  assert.ok(['new_option', 'another'].includes(actionAnswer.choice), `Expected choice to be drawn from custom spec options, got ${actionAnswer.choice}`);
});

test('simulated demo path: preset -> fixture invariant holds regardless of key order, and does not over-match (learnings-e0q2)', async () => {
  for (const [id, preset] of Object.entries(PRESETS)) {
    // (a) preset's own spec
    const resA = await decideImage({
      spec: preset.spec,
      imagePayload: DUMMY_IMAGE,
      engine: 'client'
    });
    assert.equal(
      JSON.stringify(resA.data.answers),
      JSON.stringify(preset.answers),
      `Preset ${id}: own spec must resolve to curated fixture answers`
    );

    // (b) a REORDERED COPY of the spec
    // Reorder top-level question keys
    const reorderedQuestions = {};
    const qKeys = Object.keys(preset.spec.questions).reverse();
    for (const qKey of qKeys) {
      const q = structuredClone(preset.spec.questions[qKey]);
      // Reorder choice criteria keys if applicable
      if (q.type === 'choice') {
        const cKeys = Object.keys(q.criteria).reverse();
        const reorderedCriteria = {};
        for (const cKey of cKeys) {
          reorderedCriteria[cKey] = q.criteria[cKey];
        }
        q.criteria = reorderedCriteria;
      }
      reorderedQuestions[qKey] = q;
    }
    
    const reorderedSpec = {
      title: preset.spec.title,
      description: preset.spec.description,
      questions: reorderedQuestions
    };

    const resB = await decideImage({
      spec: reorderedSpec,
      imagePayload: DUMMY_IMAGE,
      engine: 'client'
    });
    assert.equal(
      JSON.stringify(resB.data.answers),
      JSON.stringify(preset.answers),
      `Preset ${id}: reordered copy must resolve to curated fixture answers`
    );
  }

  // (c) NEGATIVE CONTROL: the ui preset's own question/option KEYS, but ONE primitive value
  // altered (a criterion label). Key-order canonicalization preserves values, so the changed
  // value breaks the match and the spec synthesises. A value-blind canonicalizer that discards
  // every primitive value would collapse this to the ui preset's keys and return the curated
  // fixture, so this control catches exactly that over-match.
  const alteredSpec = structuredClone(PRESETS.ui.spec);
  alteredSpec.questions.action.criteria.retry =
    'Click Retry Payment to attempt card re-authorization (altered label)';

  const resC = await decideImage({
    spec: alteredSpec,
    imagePayload: DUMMY_IMAGE,
    engine: 'client'
  });
  
  // Must synthesise (not match any preset's answers)
  for (const preset of Object.values(PRESETS)) {
    assert.notEqual(
      JSON.stringify(resC.data.answers),
      JSON.stringify(preset.answers),
      `Negative control must synthesise, not match preset ${preset.id}`
    );
  }
  // And it must synthesise from its own options (the altered spec keeps the ui option keys)
  assert.ok(
    ['retry', 'cancel', 'dismiss', 'unclear'].includes(resC.data.answers.action.choice),
    'Negative control choice must come from its own options'
  );
});

// learnings-9k2: the questions textarea is JSON.parse'd straight into decideImage() -> validateImageSpec(),
// so a prototype-named question id arrives as an ordinary own key (the object-literal `__proto__: {}`
// form would set the prototype and never reach the question loop at all). core.js:validateSpec refuses
// 'constructor', 'prototype' and '__proto__' by name; 'constructor' and 'prototype' still satisfy the
// character regex, so without the same list here the image lab accepts a schema the sibling lab refuses.
test('Image lab schema: prototype-named question ids are refused like core.js:validateSpec (learnings-9k2)', () => {
  const protoIdSpec = JSON.parse(
    '{"title":"Proto key triage","questions":{"__proto__":{"type":"noul","instructions":"Is the path clear?"}}}',
  );
  assert.ok(
    Object.hasOwn(protoIdSpec.questions, '__proto__'),
    'probe: JSON.parse keys __proto__ as an own question, which is how it reaches this loop',
  );
  assert.throws(
    () => validateImageSpec(protoIdSpec, DUMMY_IMAGE),
    /Use simple lowercase question IDs/,
    'a __proto__ question id is refused instead of being handed to every id-keyed map downstream',
  );

  for (const id of ['constructor', 'prototype']) {
    assert.throws(
      () =>
        validateImageSpec(
          { title: 'Proto key triage', questions: { [id]: { type: 'noul', instructions: 'Is the path clear?' } } },
          DUMMY_IMAGE,
        ),
      /Use simple lowercase question IDs/,
      `'${id}' is refused by core.js:validateSpec and must be refused here too`,
    );
  }

  // Positive control: an ordinary id in the same shape still validates, so the list cannot grow into
  // an over-broad refusal that breaks the lab's own presets.
  assert.doesNotThrow(() =>
    validateImageSpec(
      { title: 'Proto key triage', questions: { is_blocked: { type: 'noul', instructions: 'Is the path clear?' } } },
      DUMMY_IMAGE,
    ),
  );
});

// learnings-9k2: choice option labels are caller-supplied keys, and core.js asks only that a label be
// non-empty text (text(k,'Option')), so a prototype-named label is a legal option that has to be held
// safely rather than reserved. On a plain {} accumulator the canonicalizer's `acc['__proto__'] = ...`
// hits the inherited accessor, which ignores a string value: the option vanishes from the lookup key,
// a spec with one extra option canonicalizes into the four-option ui preset's exact shape, and
// decideImage() returns that preset's curated fixture answers for a spec the caller never described.
test('simulated demo path: a __proto__ option label survives canonicalization and cannot collapse a spec into a preset fixture (learnings-9k2)', async () => {
  // JSON round-trip is the textarea path: JSON.parse defines `__proto__` as an ordinary own key.
  const criteria = JSON.parse(
    JSON.stringify({ ...PRESETS.ui.spec.questions.action.criteria, ['__proto__']: 'Escalate to a human' }),
  );
  assert.ok(Object.hasOwn(criteria, '__proto__'), 'probe: the fifth option is an own key, as the textarea JSON yields it');

  const spec = structuredClone(PRESETS.ui.spec);
  spec.questions.action.criteria = criteria;

  assert.doesNotThrow(() => validateImageSpec(spec, DUMMY_IMAGE), 'a prototype-named option label is a valid option label');

  const res = await decideImage({ spec, imagePayload: DUMMY_IMAGE, engine: 'client' });

  for (const [key, preset] of Object.entries(PRESETS)) {
    assert.notEqual(
      JSON.stringify(res.data.answers),
      JSON.stringify(preset.answers),
      `a spec with an extra __proto__ option is not the ${key} preset: the label must stay in the lookup key`,
    );
  }

  const probabilities = res.data.answers.action.probabilities;
  assert.deepEqual(
    Object.keys(probabilities).sort(),
    ['__proto__', 'cancel', 'dismiss', 'retry', 'unclear'],
    'the synthesised answers describe all five options the caller named',
  );
  assert.ok(Object.hasOwn(probabilities, '__proto__'), 'the __proto__ option is answered like any other option');
  const total = Object.values(probabilities).reduce((sum, p) => sum + p, 0);
  assert.ok(Math.abs(total - 1) < 1e-9, `the distribution still sums to one over all five options (got ${total})`);
});

// learnings-9k2: core.js:validateSpec runs text(k,'Option') over every choice criteria key, so an empty
// label is refused there. The image lab only counted the keys, which accepted a spec the sibling lab
// rejects, so the label check mirrors core.js and leaves ordinary labels untouched.
test('Image lab schema: choice option labels must be non-empty text, as core.js requires (learnings-9k2)', () => {
  assert.throws(
    () =>
      validateImageSpec(
        {
          title: 'Proto key triage',
          questions: { pick: { type: 'choice', instructions: 'Choose', criteria: { '': 'unlabelled option', go: 'Proceed' } } },
        },
        DUMMY_IMAGE,
      ),
    /Option must be non-empty text/,
  );

  assert.doesNotThrow(() =>
    validateImageSpec(
      {
        title: 'Proto key triage',
        questions: { pick: { type: 'choice', instructions: 'Choose', criteria: { go: 'Proceed', stop: 'Halt' } } },
      },
      DUMMY_IMAGE,
    ),
  );
});

// learnings-x6p: validateImageSpec delegates to core.js validateSpec, eliminating divergence
// where validateImageSpec previously accepted specs core.js rejected:
//   (a) choice question with > 255 options (core: 'Choice requires 2–255 named options in this workbench.')
//   (b) noul question with criteria keys other than true/false (e.g. {yes, no})
//   (c) question with unknown extra fields (core: 'Unexpected specification field...')
//   and checks option descriptions and score levels are non-empty text.
test('Image lab schema: validateImageSpec rejects divergent cases the same way validateSpec does (learnings-x6p)', () => {
  const dummyImage = DUMMY_IMAGE;
  const getError = fn => {
    let err;
    assert.throws(() => {
      try {
        fn();
      } catch (e) {
        err = e;
        throw e;
      }
    });
    return err;
  };

  // Case (a): Choice question with 300 options (> 255)
  const choice300 = {};
  for (let i = 0; i < 300; i++) choice300[`option_${i}`] = `Description for option ${i}`;
  const specChoice300 = {
    title: 'Choice upper bound test',
    description: 'Test choice options count',
    state: 'State',
    questions: {
      action: {
        type: 'choice',
        instructions: 'Choose an option',
        criteria: choice300
      }
    }
  };
  const coreErrA = getError(() => validateSpec(specChoice300));
  const imageErrA = getError(() => validateImageSpec(specChoice300, dummyImage));
  assert.equal(imageErrA.message, coreErrA.message);
  assert.match(imageErrA.message, /Choice requires 2–255 named options/);

  // Case (b): Noul question whose criteria keys are {yes, no} instead of {true, false}
  const specNoulYesNo = {
    title: 'Noul criteria keys test',
    description: 'Test noul criteria schema',
    state: 'State',
    questions: {
      is_valid: {
        type: 'noul',
        instructions: 'Is this valid?',
        criteria: { yes: 'It is valid', no: 'It is invalid' }
      }
    }
  };
  const coreErrB = getError(() => validateSpec(specNoulYesNo));
  const imageErrB = getError(() => validateImageSpec(specNoulYesNo, dummyImage));
  assert.equal(imageErrB.message, coreErrB.message);
  assert.match(imageErrB.message, /Unexpected specification field/);

  // Case (c): Question with an unknown extra field
  const specExtraField = {
    title: 'Extra field test',
    description: 'Test unknown question field rejection',
    state: 'State',
    questions: {
      triage: {
        type: 'noul',
        instructions: 'Should we triage?',
        unexpected_extra_property: true
      }
    }
  };
  const coreErrC = getError(() => validateSpec(specExtraField));
  const imageErrC = getError(() => validateImageSpec(specExtraField, dummyImage));
  assert.equal(imageErrC.message, coreErrC.message);
  assert.match(imageErrC.message, /Unexpected specification field/);

  // Additional divergences mentioned in review:
  // Choice option descriptions must be non-empty text
  const specChoiceEmptyDesc = {
    title: 'Choice empty desc test',
    description: 'Test option description non-empty text',
    state: 'State',
    questions: {
      action: {
        type: 'choice',
        instructions: 'Choose',
        criteria: { opt1: 'Valid description', opt2: '   ' }
      }
    }
  };
  const coreErrDesc = getError(() => validateSpec(specChoiceEmptyDesc));
  const imageErrDesc = getError(() => validateImageSpec(specChoiceEmptyDesc, dummyImage));
  assert.equal(imageErrDesc.message, coreErrDesc.message);
  assert.match(imageErrDesc.message, /Option description must be non-empty text/);

  // Score levels must be non-empty text
  const specScoreNumericLevel = {
    title: 'Score level non-text test',
    description: 'Test score level must be text',
    state: 'State',
    questions: {
      severity: {
        type: 'score',
        instructions: 'Rate severity',
        criteria: ['Low', 2, 'High']
      }
    }
  };
  const coreErrScore = getError(() => validateSpec(specScoreNumericLevel));
  const imageErrScore = getError(() => validateImageSpec(specScoreNumericLevel, dummyImage));
  assert.equal(imageErrScore.message, coreErrScore.message);
  assert.match(imageErrScore.message, /Score level must be non-empty text/);

  // Case (a): description and/or state omitted -> validateImageSpec accepts (defaults applied)
  assert.doesNotThrow(() => validateImageSpec({
    title: 'Image-only spec without state or description',
    questions: {
      q1: { type: 'noul', instructions: 'Is this an image?' }
    }
  }, dummyImage), 'omitting description and state applies image defaults and validates');

  assert.doesNotThrow(() => validateImageSpec({
    title: 'Omitted state with custom description',
    description: 'Custom description',
    questions: {
      q1: { type: 'noul', instructions: 'Is this an image?' }
    }
  }, dummyImage), 'omitting only state applies state default and keeps description');

  assert.doesNotThrow(() => validateImageSpec({
    title: 'Omitted description with custom state',
    state: 'Custom state',
    questions: {
      q1: { type: 'noul', instructions: 'Is this an image?' }
    }
  }, dummyImage), 'omitting only description applies description default and keeps state');

  // Case (b): description: null and state: null (and empty-string variants) -> validateImageSpec rejects exactly as validateSpec does
  const baseSpec = {
    title: 'Null/empty field triage',
    description: 'Valid description',
    state: 'Valid state',
    questions: {
      q1: { type: 'noul', instructions: 'Is this an image?' }
    }
  };

  for (const field of ['description', 'state']) {
    const expectedRegex = new RegExp(`${field === 'description' ? 'Description' : 'State'} must be non-empty text`);

    // null variant: present-but-null must NOT default, must be rejected like core validateSpec
    const specNull = { ...baseSpec, [field]: null };
    const coreErrNull = getError(() => validateSpec(specNull));
    const imageErrNull = getError(() => validateImageSpec(specNull, dummyImage));
    assert.equal(imageErrNull.message, coreErrNull.message, `${field}: null must be rejected identically to core validateSpec`);
    assert.match(imageErrNull.message, expectedRegex);

    // empty string variant: present-but-empty must be rejected like core validateSpec
    const specEmpty = { ...baseSpec, [field]: '' };
    const coreErrEmpty = getError(() => validateSpec(specEmpty));
    const imageErrEmpty = getError(() => validateImageSpec(specEmpty, dummyImage));
    assert.equal(imageErrEmpty.message, coreErrEmpty.message, `${field}: empty string must be rejected identically to core validateSpec`);
    assert.match(imageErrEmpty.message, expectedRegex);

    // whitespace-only variant
    const specWhitespace = { ...baseSpec, [field]: '   ' };
    const coreErrWhitespace = getError(() => validateSpec(specWhitespace));
    const imageErrWhitespace = getError(() => validateImageSpec(specWhitespace, dummyImage));
    assert.equal(imageErrWhitespace.message, coreErrWhitespace.message, `${field}: whitespace string must be rejected identically to core validateSpec`);
    assert.match(imageErrWhitespace.message, expectedRegex);
  }

  // Unknown top-level field rejection
  assert.throws(
    () => validateImageSpec({
      title: 'Extra top-level field',
      unknownTopLevel: 'invalid',
      questions: {
        q1: { type: 'noul', instructions: 'Is this an image?' }
      }
    }, dummyImage),
    /Unexpected specification field/
  );
});

test('image lab: drop zone styles, CSP compliance, keyboard activation, and non-image rejection (learnings-2w9, learnings-ae0, learnings-1ee)', async () => {
  const { serve } = await import('../scripts/serve.mjs');
  const { launch } = await import('./lib/cdp.mjs');
  const { server, url } = await serve();
  let page;
  try {
    page = await launch();
    await page.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `
        window.__cspViolations = [];
        addEventListener('securitypolicyviolation', e => {
          window.__cspViolations.push({
            directive: e.violatedDirective,
            blockedURI: e.blockedURI
          });
        });
      `
    });

    await page.goto(url + 'decision-models/image-lab.html');
    await page.waitFor(() => document.querySelector('#preview-image').src.startsWith('data:image/'));

    // 1. learnings-2w9: CSP compliance and file input hidden from drop zone
    const violations = await page.evaluate(() => window.__cspViolations);
    assert.equal(violations.length, 0, `Expected 0 CSP violations, got: ${JSON.stringify(violations)}`);

    const measurements1280 = await page.evaluate(() => {
      const input = document.querySelector('#image-file');
      const dropZone = document.querySelector('#drop-zone');
      const inputRect = input.getBoundingClientRect();
      const dropRect = dropZone.getBoundingClientRect();
      return {
        inputDisplay: window.getComputedStyle(input).display,
        inputWidth: inputRect.width,
        inputHeight: inputRect.height,
        dropBorder: window.getComputedStyle(dropZone).borderStyle
      };
    });
    assert.equal(measurements1280.inputDisplay, 'none');
    assert.equal(measurements1280.inputWidth, 0);
    assert.equal(measurements1280.inputHeight, 0);
    assert.equal(measurements1280.dropBorder, 'dashed');

    // 390px mobile check
    await page.emulateViewport({ width: 390, height: 844, mobile: true, scale: 1 });
    const measurements390 = await page.evaluate(() => {
      const input = document.querySelector('#image-file');
      const inputRect = input.getBoundingClientRect();
      return {
        inputDisplay: window.getComputedStyle(input).display,
        inputWidth: inputRect.width,
        inputHeight: inputRect.height
      };
    });
    assert.equal(measurements390.inputDisplay, 'none');
    assert.equal(measurements390.inputWidth, 0);
    assert.equal(measurements390.inputHeight, 0);

    // 2. learnings-ae0: Keyboard activation on role="button" drop zone
    await page.emulateViewport({ width: 1280, height: 900, mobile: false, scale: 1 });
    const keyResults = await page.evaluate(() => {
      const dropZone = document.querySelector('#drop-zone');
      const input = document.querySelector('#image-file');
      let clicks = 0;
      input.addEventListener('click', e => {
        clicks++;
        e.preventDefault();
      });

      const enterEv = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
      dropZone.dispatchEvent(enterEv);
      const afterEnter = clicks;

      const spaceEv = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
      dropZone.dispatchEvent(spaceEv);
      const afterSpace = clicks;
      const spacePrevented = spaceEv.defaultPrevented;

      const escEv = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
      dropZone.dispatchEvent(escEv);
      const afterEsc = clicks;

      return { afterEnter, afterSpace, afterEsc, spacePrevented };
    });
    assert.equal(keyResults.afterEnter, 1, 'Enter key must trigger file input click');
    assert.equal(keyResults.afterSpace, 2, 'Space key must trigger file input click');
    assert.equal(keyResults.afterEsc, 2, 'Other keys must not trigger file input click');
    assert.equal(keyResults.spacePrevented, true, 'Space keydown must prevent default scrolling');

    // 3. learnings-1ee: Non-image file handling and clearing
    const nonImageInputStatus = await page.evaluate(() => {
      const status = document.querySelector('#status');
      const input = document.querySelector('#image-file');
      const dt = new DataTransfer();
      dt.items.add(new File(['data'], 'test.txt', { type: 'text/plain' }));
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
      return status.textContent;
    });
    assert.match(nonImageInputStatus, /test\.txt.*is not an image/i);

    // Choosing a valid image clears the error
    await page.evaluate(() => {
      const input = document.querySelector('#image-file');
      const dt = new DataTransfer();
      dt.items.add(new File(['fake-png-bytes'], 'valid.png', { type: 'image/png' }));
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitFor(() => document.querySelector('#status').textContent.includes('valid.png'));
    const validStatus = await page.evaluate(() => document.querySelector('#status').textContent);
    assert.match(validStatus, /valid\.png/);
    assert.doesNotMatch(validStatus, /not an image/);

    // Dropping a non-image file shows error
    const nonImageDropStatus = await page.evaluate(() => {
      const status = document.querySelector('#status');
      const dropZone = document.querySelector('#drop-zone');
      const dt = new DataTransfer();
      dt.items.add(new File(['pdf'], 'doc.pdf', { type: 'application/pdf' }));
      const dropEvent = new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: dt });
      dropZone.dispatchEvent(dropEvent);
      return status.textContent;
    });
    assert.match(nonImageDropStatus, /doc\.pdf.*is not an image/i);

    // Presets clear the error
    await page.click('#preset-ui');
    const presetStatus = await page.evaluate(() => document.querySelector('#status').textContent);
    assert.match(presetStatus, /Preset loaded/);
    assert.doesNotMatch(presetStatus, /not an image/);
  } finally {
    if (page) await page.close();
    server.close();
  }
});

