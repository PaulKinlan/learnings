import { decide, gate } from './core.js';
import { loadLaya } from './laya-engine.js';
import { loadKev } from './kev-engine.js';
import {
  choose as kevChoose,
  expectedLevel as kevExpectedLevel,
  noulProbability as kevNoulProbability,
} from './kev-pack.js';

export const $ = id => document.getElementById(id);

let layaSession = null;
let kevSession = null;

export function settings() {
  const p = $('provider') ? $('provider').value : 'laya';
  return {
    provider: p,
    key: $('decision-key') ? $('decision-key').value : '',
    model: $('model') ? $('model').value : '',
    endpoint: $('endpoint') ? $('endpoint').value : '',
    transport: $('transport') ? $('transport').value : 'direct'
  };
}

export function setupSettings(onChange = () => {}) {
  const transportEl = $('transport-field');
  if (transportEl) transportEl.hidden = !['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
  $('transport')?.addEventListener('change', () => onChange('transport'));

  const update = () => {
    const p = $('provider').value;
    const jevField = $('jev-key-field');
    const kevField = $('kev-field');
    const synthField = $('synthetic-field');
    const modelInput = $('model');
    const note = $('provider-note');

    if (jevField) jevField.hidden = p !== 'jev';
    if (kevField) kevField.hidden = true; // Kev runs in-browser
    if (synthField) synthField.hidden = true; // No synthetic mode
    if (modelInput) {
      modelInput.disabled = p !== 'jev';
      modelInput.value = p === 'laya' ? 'laya-multilingual' : p === 'kev' ? 'kev-0.6b' : 'jev-1.13.0';
    }

    if (note) {
      note.textContent = p === 'laya'
        ? 'Laya runs on-device in this tab via LiteRT.js (mmBERT-base). Weights download once (~680 MB total); zero data leaves your browser.'
        : p === 'kev'
        ? 'Kev runs on-device in this tab via ONNX Runtime Web. Weights download once (~375 MB); zero data leaves your browser.'
        : 'Jev sends state and questions directly to api.typesafe.ai with your API key. Incurs provider usage.';
    }
    onChange();
  };

  $('provider')?.addEventListener('change', update);
  const thresh = $('threshold');
  if (thresh) {
    thresh.addEventListener('input', () => {
      const val = $('threshold-value');
      if (val) val.textContent = Number(thresh.value).toFixed(2);
      onChange('threshold');
    });
  }

  const clearBtn = $('clear-keys');
  if (clearBtn) {
    clearBtn.onclick = () => {
      document.querySelectorAll('input[type=password]').forEach(i => i.value = '');
      const note = $('provider-note');
      if (note) note.textContent = 'Keys cleared from memory. In-flight requests cannot be recalled; use Cancel to stop waiting.';
    };
  }

  $('model')?.addEventListener('input', () => onChange('model'));
  update();
}

export async function runSpec(spec, signal, onProgress = null) {
  const cfg = settings();
  if (cfg.provider === 'jev' && !cfg.key?.trim()) {
    throw new Error('Enter a Jev API key first.');
  }
  if (globalThis.__MOCK_DECIDE__) {
    return globalThis.__MOCK_DECIDE__(spec, cfg);
  }
  if (cfg.provider === 'laya') {
    if (!layaSession) {
      layaSession = await loadLaya({
        onProgress: (stage, received, total) => onProgress?.(stage, received, total),
        ...(globalThis.__LAYA_URLS ? { urls: globalThis.__LAYA_URLS } : {})
      });
    }
    const started = performance.now();
    const layaQuestions = {};
    for (const [id, q] of Object.entries(spec.questions)) {
      layaQuestions[id] = {
        t: q.type,
        ins: q.instructions,
        crit: q.criteria ?? {}
      };
    }
    const report = await layaSession.decideAll(spec.state, layaQuestions);
    const elapsed = performance.now() - started;
    return {
      data: {
        model: layaSession.checkpoint,
        answers: report.answers
      },
      elapsed,
      source: 'Laya (LiteRT.js in-browser)',
      requestedModel: 'laya-multilingual'
    };
  }

  if (cfg.provider === 'kev') {
    if (!kevSession) {
      kevSession = await loadKev({
        onProgress: (info) => onProgress?.('weights', info?.loaded || 0, info?.total || 0),
        ...(globalThis.__KEV_URLS ? { urls: globalThis.__KEV_URLS } : {})
      });
    }
    const started = performance.now();
    const kevQuestions = Object.values(spec.questions).map(q => {
      if (q.type === 'choice') {
        const labels = Object.keys(q.criteria);
        return {
          instruction: q.instructions,
          options: labels.map(l => q.criteria[l] ? `${l}: ${q.criteria[l]}` : l)
        };
      }
      if (q.type === 'score') {
        return {
          instruction: q.instructions,
          options: q.criteria
        };
      }
      return {
        instruction: q.instructions,
        options: ['no', 'yes']
      };
    });
    const { distributions, tokens, ms, flat } = await kevSession.decidePacked(spec.state, kevQuestions);
    const elapsed = performance.now() - started;
    const answers = {};
    Object.keys(spec.questions).forEach((id, i) => {
      const q = spec.questions[id];
      const dist = distributions[i];
      if (q.type === 'choice') {
        const labels = Object.keys(q.criteria);
        const best = kevChoose(dist);
        answers[id] = {
          type: 'choice',
          choice: labels[best],
          probabilities: Object.fromEntries(labels.map((l, o) => [l, dist[o]])),
          confidence: dist[best]
        };
      } else if (q.type === 'score') {
        answers[id] = {
          type: 'score',
          score: kevExpectedLevel(dist),
          probabilities: Object.fromEntries(q.criteria.map((_, o) => [String(o), dist[o]])),
          confidence: Math.max(...dist)
        };
      } else {
        const pYes = kevNoulProbability(dist);
        answers[id] = {
          type: 'noul',
          noul: pYes,
          confidence: Math.max(...dist)
        };
      }
    });
    return {
      data: {
        model: kevSession.checkpoint,
        answers
      },
      elapsed,
      source: 'Kev (ONNX Runtime Web in-browser)',
      requestedModel: 'kev-0.6b'
    };
  }

  // Jev API
  return decide(spec, cfg, signal);
}

export function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}

export function renderAnswers(target, result) {
  target.replaceChildren();
  for (const [id, a] of Object.entries(result.data.answers)) {
    const box = el('section', undefined, 'answer');
    box.append(
      el('h3', id),
      el('p', a.type === 'choice' ? `Selected: ${a.choice}` : a.type === 'score' ? `Expected level: ${a.score.toFixed(2)}` : `p(yes): ${a.noul.toFixed(3)}`)
    );
    const ps = a.type === 'noul' ? { yes: a.noul, no: 1 - a.noul } : a.probabilities;
    for (const [label, value] of Object.entries(ps)) {
      const row = el('div', undefined, 'bar');
      const meter = el('meter');
      meter.min = 0;
      meter.max = 1;
      meter.value = value;
      meter.setAttribute('aria-label', label + ' probability');
      row.append(el('span', label), meter, el('output', (value * 100).toFixed(1) + '%'));
      box.append(row);
    }
    if (a.type !== 'noul' && a.confidence !== undefined) {
      box.append(el('p', `Returned confidence: ${Number(a.confidence).toFixed(3)} (not the winning probability)`));
    }
    box.append(el('p', `Gate: ${gate(a, Number($('threshold').value))}`, 'gate'));
    target.append(box);
  }
}

export function download(data, name) {
  const u = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = el('a');
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}
