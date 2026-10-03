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
  const dummyTarget = {
    children: [],
    replaceChildren() { this.children = []; },
    append(...items) { this.children.push(...items); }
  };

  renderImageAnswers(dummyTarget, res, 0.8);
  const badge = dummyTarget.children.find(c => c.id === 'simulation-badge');
  assert.ok(badge, 'Simulation disclosure badge must be rendered in UI answers');
  assert.ok(badge.textContent.includes('Notice: UI and schema validation simulation.'));
  assert.ok(badge.textContent.includes('Connect a local vision server'));
});

