// The playground driver: four demos, two engines, everything local.
//
// Laya answers in the host-contract's own dictionary (probabilities included); Kev answers
// in kev-pack's softmaxed per-option distributions. The demo definitions live in the page's
// textareas in Laya's {t, ins, crit} schema and are translated for Kev at run time — the
// options Kev sees are the same rendered strings Laya's contract renders (laya-pack's
// renderOptions), so both engines read the same words.

import { loadLaya } from "./laya-engine.js";
import { loadKev, KevSession } from "./kev-engine.js";
import { Classifier } from "./classifier-api.js";
import { validateAnswers } from "./core.js";
import { renderOptions, choiceConfidence } from "./laya-pack.js";
import {
  choose as kevChoose,
  expectedLevel as kevExpectedLevel,
  noulProbability as kevNoulProbability,
  NOUL_OPTIONS,
} from "./kev-pack.js";

// Ensure window.Classifier polyfill is available on the page
Classifier.install();

const $ = (id) => document.getElementById(id);
const engines = {
  laya: {
    label: "Laya",
    checkpointNote: "multilingual mmBERT on LiteRT.js (WASM/XNNPack), ~650 MB of verified downloads",
    // window.__LAYA_URLS is a test seam: when set (by a smoke driver), loadLaya uses those
    // asset URLs instead of the hub's. Unset in normal use, so the hub is the source.
    load: (onProgress) => loadLaya({
      onProgress: (stage, received, total) => onProgress(stage, received, total),
      ...(globalThis.__LAYA_URLS ? { urls: globalThis.__LAYA_URLS } : {}),
    }),
  },
  kev: {
    label: "Kev",
    checkpointNote: "0.6 B decision model on ONNX Runtime Web (CPU), ~375 MB download",
    load: (onProgress) => loadKev({
      onProgress: ({ stage, file, loaded, total }) => onProgress(stage || "weights", loaded, total, file),
      ...(globalThis.__KEV_URLS ? { urls: globalThis.__KEV_URLS } : {}),
    }),
  },
  jev: {
    label: "Jev",
    checkpointNote: "Hosted API (api.typesafe.ai), needs your API key",
    load: async () => {
      const key = $("jev-key")?.value?.trim();
      if (!key) throw new Error("Enter your Jev API key in the field next to Engine.");
      return {
        checkpoint: "api.typesafe.ai/v1/systemone (hosted Jev)",
        decideAll: async (state, questions) => {
          // A null-prototype object so a question id of `__proto__` is stored as an ordinary own
          // property. On a normal object, tsQuestions['__proto__'] = {...} triggers the inherited
          // accessor and reassigns the prototype instead of adding a key, silently dropping the
          // question from the wire body before it ever reaches validateAnswers().
          const tsQuestions = Object.create(null);
          for (const [id, q] of Object.entries(questions)) {
            tsQuestions[id] = {
              type: q.t === "choice" ? "choice" : q.t === "score" ? "score" : "noul",
              instructions: q.ins,
              criteria: q.crit || {},
            };
          }
          // Invariant I2: a UI-blocking hosted call carries a bounded AbortSignal. The sibling
          // labs wire the same 45s boundary (core.js decide()/generate(), image-lab.js
          // decideImage()), so a provider that accepts the connection and never answers still
          // ends the run instead of leaving the Run button disabled with no way out.
          const signal = AbortSignal.timeout(45000);
          let res;
          try {
            res = await fetch("https://api.typesafe.ai/v1/systemone", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${key}`,
              },
              body: JSON.stringify({
                model: "jev-1.13.0",
                state,
                questions: tsQuestions,
              }),
              signal,
            });
          } catch (error) {
            // core.js post() reports an aborted signal as a timeout, so the playground says the
            // same thing when a provider accepts the connection and then never answers.
            if (signal.aborted) throw new Error("Request cancelled or timed out. No automatic retry.");
            throw error;
          }
          if (!res.ok) throw new Error(`Jev API returned HTTP ${res.status}`);
          // Known, pinned limitation: res.json() (JSON.parse) collapses duplicate object keys to
          // the last occurrence, per ECMAScript and RFC 8259 §4 ("The names within an object
          // SHOULD be unique"). A provider that emits {"tool":null,"tool":<valid>} therefore
          // reaches validateAnswers() as a single last-wins `tool` answer, not as a duplicate.
          // That residual is accepted and pinned by a browser test rather than detected with a
          // raw-text re-scan: a regex key scan can false-positive on `"` inside string values and
          // reject a legitimate response (regressing the passing Jev path), and a correct
          // duplicate detector is a second JSON tokenizer — not a cheap pre-check.
          const json = await res.json();
          // Invariants I1/I11: hosted-provider JSON is externally derived, so it passes the same
          // validator core.js decide() and image-lab.js decideImage() apply before rendering.
          const data = validateAnswers(json, tsQuestions);
          return {
            answers: data.answers,
            usage: json.usage || { input_tokens: 0 },
          };
        },
      };
    },
  },
};

let activeEngine = null; // { name, session }

function engineName() {
  return $("engine").value;
}

function setProgress(text) {
  $("progress").textContent = text;
}

async function ensureEngine() {
  const name = engineName();
  if (activeEngine?.name === name) return activeEngine.session;
  document.querySelectorAll(".engine-name").forEach((el) => (el.textContent = engines[name].label));
  $("load").disabled = true;
  const started = performance.now();
  const spec = engines[name];
  setProgress(`Loading ${spec.label}: ${spec.checkpointNote}…`);
  try {
    const session = await spec.load((stage, received, total, file) => {
      const mb = (n) => (n / 1048576).toFixed(1);
      setProgress(
        total
          ? `${spec.label} · ${stage} · ${mb(received)} of ${mb(total)} MB${file ? ` · ${file}` : ""}`
          : `${spec.label} · ${stage}…`,
      );
    });
    activeEngine = { name, session };
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    setProgress(
      name === "jev"
        ? `${spec.label} ready in ${seconds}s — ${session.checkpoint}. Requests use your API key.`
        : `${spec.label} ready in ${seconds}s — ${session.checkpoint}. Nothing has been sent anywhere.`,
    );
    $("demos").hidden = false;
    $("backend").textContent = backendLine(name, session);
    return session;
  } catch (error) {
    setProgress(`Could not load ${spec.label}: ${error.message}`);
    throw error;
  } finally {
    $("load").disabled = false;
  }
}

function backendLine(name, session) {
  if (name === "kev") {
    const report = session.backendReport();
    return (
      `Requested ${report.requested}. ` +
      (report.providers.length ? `Session reports: ${report.providers.join(", ")}. ` : "Session did not report its providers. ") +
      `WebGPU is ${report.webgpuPresent ? "present" : "not present"} in this browser and no WebGPU binary is vendored, so it was not used. ` +
      `Cross-origin isolation is ${report.crossOriginIsolated ? "on" : "off, so wasm runs without shared-memory threads"}.`
    );
  }
  return `${session.checkpoint}. No WebGPU path is offered by this page and none is claimed.`;
}

// ── running one demo ─────────────────────────────────────────────────────────

function parseQuestions(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`the questions are not valid JSON: ${error.message}`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || !Object.keys(parsed).length)
    throw new Error("the questions must be a non-empty JSON object of question-id to question");
  return parsed;
}

const NOUL_DEFAULT_CRIT = {};

/** Render one question's answer per engine, as a small DOM block. */
function renderAnswer(qid, q, answer, meta) {
  const block = document.createElement("div");
  block.className = "answer";
  const title = document.createElement("h4");
  title.textContent = `${qid} — ${q.ins}`;
  block.append(title);

  const bar = (label, probability, highlight) => {
    const row = document.createElement("p");
    row.className = "small";
    const pct = (probability * 100).toFixed(1);
    row.textContent = `${highlight ? "→ " : ""}${label}: ${pct}%`;
    if (highlight) row.style.fontWeight = "bold";
    return row;
  };

  const actProb = answer.action?.act_probability ?? "—";

  if (answer.type === "choice") {
    block.append(bar(`choice: ${answer.choice}`, answer.probabilities[answer.choice], true));
    for (const [label, p] of Object.entries(answer.probabilities)) {
      if (label !== answer.choice) block.append(bar(label, p, false));
    }
    block.append(Object.assign(document.createElement("p"), {
      className: "small",
      textContent: `confidence ${answer.confidence} · act ${actProb}`,
    }));
  } else if (answer.type === "score") {
    const legend = answer.legend ?? Object.fromEntries((Array.isArray(q.crit) ? q.crit : []).map((c, o) => [String(o), c]));
    block.append(bar(`score: ${answer.score} (${nearestLegend(answer.score, legend)})`, Math.min(Number(answer.confidence) || 0, 1), true));
    for (const [index, p] of Object.entries(answer.probabilities)) {
      block.append(bar(`${index} — ${legend[index] ?? index}`, p, false));
    }
    block.append(Object.assign(document.createElement("p"), {
      className: "small",
      textContent: `confidence ${answer.confidence} · act ${actProb}`,
    }));
  } else if (answer.type === "noul") {
    block.append(bar("yes", answer.noul, answer.noul >= 0.5));
    block.append(bar("no", 1 - answer.noul, answer.noul < 0.5));
    block.append(Object.assign(document.createElement("p"), {
      className: "small",
      textContent:
        `confidence ${answer.confidence} · act ${actProb}. ` +
        `This is a probability, not a boolean — judge the ranking before any threshold.`,
    }));
  }

  block.append(Object.assign(document.createElement("p"), { className: "small", textContent: meta }));
  return block;
}

function nearestLegend(score, legend) {
  const keys = Object.keys(legend || {});
  if (!keys.length) return String(score);
  const index = Math.max(0, Math.min(keys.length - 1, Math.round(score)));
  return legend[index] ?? String(score);
}

/** Translate a Laya-schema question to Kev's {instruction, options} shape. */
function kevQuestionFor(q) {
  if (q.t === "noul") return { instruction: q.ins, options: [...NOUL_OPTIONS] };
  return { instruction: q.ins, options: renderOptions({ t: q.t, crit: q.crit }) };
}

async function runDemo(demoEl, session, engine) {
  const results = demoEl.querySelector(".results");
  results.replaceChildren();
  const state = demoEl.querySelector(".state-text").value.trim();
  let questions;
  try {
    questions = parseQuestions(demoEl.querySelector(".state-questions").value);
  } catch (error) {
    results.append(Object.assign(document.createElement("p"), { className: "status", textContent: error.message }));
    return;
  }
  if (!state) {
    results.append(Object.assign(document.createElement("p"), { className: "status", textContent: "The state must not be empty." }));
    return;
  }

  const button = demoEl.querySelector(".run-demo");
  button.disabled = true;
  const started = performance.now();
  try {
    if (engine === "laya" || engine === "jev") {
      const report = await session.decideAll(state, questions);
      const ms = Math.round(performance.now() - started);
      for (const [qid, answer] of Object.entries(report.answers)) {
        results.append(
          renderAnswer(qid, questions[qid], answer, `${session.checkpoint} · ${ms} ms total · ${report.usage.input_tokens} input tokens`),
        );
      }
    } else {
      const kevQuestions = Object.values(questions).map(kevQuestionFor);
      const { distributions, tokens, ms, flat } = await session.decidePacked(state, kevQuestions);
      if (flat) {
        results.append(Object.assign(document.createElement("p"), {
          className: "status",
          textContent: "The graph returned one uniform score — the delimiter readout failed, and any answer printed here would be invented. This is the failure mode the delimiter check exists to name.",
        }));
      }
      Object.keys(questions).forEach((qid, i) => {
        const q = questions[qid];
        const dist = distributions[i];
        let answer;
        if (q.t === "choice") {
          const labels = Object.keys(q.crit);
          const best = kevChoose(dist);
          answer = {
            type: "choice",
            choice: labels[best],
            probabilities: Object.fromEntries(labels.map((l, o) => [l, Math.round(dist[o] * 10000) / 10000])),
            confidence: choiceConfidence(dist, labels.length).toFixed(4),
            action: { act_probability: "—" },
          };
        } else if (q.t === "score") {
          answer = {
            type: "score",
            score: Math.round(kevExpectedLevel(dist) * 10000) / 10000,
            legend: Object.fromEntries(q.crit.map((c, o) => [String(o), c])),
            probabilities: Object.fromEntries(q.crit.map((_, o) => [String(o), Math.round(dist[o] * 10000) / 10000])),
            confidence: choiceConfidence(dist, q.crit.length).toFixed(4),
            action: { act_probability: "—" },
          };
        } else {
          answer = {
            type: "noul",
            noul: Math.round(kevNoulProbability(dist) * 10000) / 10000,
            confidence: Math.max(...dist).toFixed(4),
            action: { act_probability: "—" },
          };
        }
        results.append(renderAnswer(qid, q, answer, `${session.checkpoint} · ${ms} ms · ${tokens} tokens in one packed pass`));
      });
    }
  } catch (error) {
    results.append(Object.assign(document.createElement("p"), { className: "status", textContent: `Run failed: ${error.message}` }));
  } finally {
    button.disabled = false;
  }
}

// ── wiring ───────────────────────────────────────────────────────────────────

$("load").addEventListener("click", () => {
  ensureEngine().catch(() => {});
});
$("engine").addEventListener("change", () => {
  const name = engineName();
  if ($("jev-options")) $("jev-options").hidden = name !== "jev";
  document.querySelectorAll(".engine-name").forEach((el) => (el.textContent = engines[name].label));
  if (activeEngine && activeEngine.name !== name) {
    setProgress(`${engines[name].label} is selected; ${activeEngine.name === name ? "already loaded" : "loads on the next run"}.`);
  }
});
document.querySelectorAll(".run-demo").forEach((button) => {
  button.addEventListener("click", async () => {
    const session = await ensureEngine().catch(() => null);
    if (session) await runDemo(button.closest(".demo"), session, engineName());
  });
});

async function runClassifierDemo() {
  const input = $("classifier-input").value;
  const schemaText = $("classifier-schema").value;
  const results = $("classifier-results");
  const codeBlock = $("classifier-code");
  const button = $("run-classifier");

  let schema;
  try {
    schema = JSON.parse(schemaText);
  } catch (e) {
    results.replaceChildren(Object.assign(document.createElement("p"), {
      className: "status",
      textContent: `Invalid schema JSON: ${e.message}`,
    }));
    return;
  }

  button.disabled = true;
  results.replaceChildren(Object.assign(document.createElement("p"), {
    className: "status",
    textContent: "Running inference via window.Classifier…",
  }));

  const code = `// W3C Chrome Built-in Classifier API
const schema = ${JSON.stringify(schema, null, 2)};

const classifier = await Classifier.create(schema);
const results = await classifier.classify(${JSON.stringify(input)});

console.log(results);
classifier.destroy();`;
  if (codeBlock) codeBlock.textContent = code;

  const started = performance.now();
  try {
    Classifier.install();
    const classifier = await Classifier.create(schema);
    const decisions = await classifier.classify(input);
    const elapsed = Math.round(performance.now() - started);

    results.replaceChildren();
    for (const [qid, d] of Object.entries(decisions)) {
      const box = document.createElement("div");
      box.className = "answer";
      const h4 = document.createElement("h4");
      h4.textContent = `${qid} → Selected: ${d.label} (confidence: ${d.confidence.toFixed(3)})`;
      box.append(h4);

      if (d.expectedScore !== undefined) {
        const sc = document.createElement("p");
        sc.textContent = `Expected score: ${d.expectedScore.toFixed(2)}`;
        box.append(sc);
      }

      for (const p of d.probabilities) {
        const row = document.createElement("p");
        row.className = "bar";
        const labelSpan = document.createElement("span");
        labelSpan.textContent = p.label + (p.description ? `: ${p.description}` : "");
        const meter = document.createElement("meter");
        meter.min = 0;
        meter.max = 1;
        meter.value = p.probability;
        const out = document.createElement("output");
        out.textContent = `${(p.probability * 100).toFixed(1)}%`;
        row.append(labelSpan, meter, out);
        box.append(row);
      }
      results.append(box);
    }
    const meta = document.createElement("p");
    meta.className = "small";
    meta.textContent = `window.Classifier · ${elapsed} ms total wall time`;
    results.append(meta);
    classifier.destroy();
  } catch (err) {
    results.replaceChildren(Object.assign(document.createElement("p"), {
      className: "status",
      textContent: `Classifier error: ${err.message}`,
    }));
  } finally {
    button.disabled = false;
  }
}

$("run-classifier")?.addEventListener("click", runClassifierDemo);

// Kev exposes its class for tests that need it.
export { KevSession, Classifier };
