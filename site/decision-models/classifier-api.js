// Chrome Built-in AI Classifier API proposal implementation
// Follows the explainer (https://github.com/michaelwasserman/classifier-api)
// as polyfilled by web-ai.studio using the Laya decision encoder on LiteRT.js.
//
// Key methods:
//   Classifier.availability(schema)
//   Classifier.create(options) -> ClassifierSession
//   session.classify(input, options) -> Record<string, ClassifierDecision>
//   session.measureContextUsage(input, options) -> number
//   session.destroy()
//
// Question types:
//   - 'binary' (or 'boolean'): returns true/false label, probability, confidence
//   - 'categorical' (or 'choice'): returns winning label, probabilities, confidence
//   - 'ordinal' (or 'score'): returns winning label, expectedScore, probabilities, confidence

import { loadLaya } from "./laya-engine.js";

let cachedLayaSession = null;
let loadingLayaPromise = null;

async function getLaya(onProgress, urls) {
  if (cachedLayaSession) return cachedLayaSession;
  if (!loadingLayaPromise) {
    loadingLayaPromise = loadLaya({
      onProgress: (stage, received, total) => {
        const loaded = total ? received / total : 0;
        onProgress?.(loaded);
      },
      urls: urls ?? (globalThis.__LAYA_URLS || {}),
    })
      .then((session) => {
        cachedLayaSession = session;
        loadingLayaPromise = null;
        return session;
      })
      .catch((err) => {
        loadingLayaPromise = null;
        throw err;
      });
  }
  return loadingLayaPromise;
}

/**
 * Map Classifier questions (the caller's schema) into Laya's {t, ins, crit} contract.
 *
 * A null-prototype map, because the id is caller-supplied and `Classifier.create()` validates
 * only that it is a non-empty string. On a normal object, `map['__proto__'] = {...}` hits the
 * inherited accessor and reassigns the prototype instead of adding a key, so the question the
 * caller asked is dropped before `classify()`'s Object.entries() ever runs it. JSON.parse (the
 * playground hands the schema textarea's text straight here) yields `__proto__` as an ordinary
 * own key, so the id does reach this loop.
 */
export function toLayaQuestions(questions, context) {
  const layaQuestions = Object.create(null);
  for (const q of questions) {
    const prompt = context ? `${context}\n${q.prompt || ""}` : q.prompt || "";
    const qtype = (q.type || "categorical").toLowerCase();

    if (qtype === "binary" || qtype === "boolean") {
      layaQuestions[q.id] = {
        t: "noul",
        ins: prompt,
        crit: {},
        meta: { originalType: "binary", q },
      };
    } else if (qtype === "ordinal" || qtype === "score") {
      const optionsList = Array.isArray(q.options) ? q.options : [];
      const crit = optionsList.map((o) => (typeof o === "string" ? o : o.description || o.label));
      layaQuestions[q.id] = {
        t: "score",
        ins: prompt,
        crit,
        meta: { originalType: "ordinal", q, optionsList },
      };
    } else {
      // Categorical / choice
      const optionsList = Array.isArray(q.options) ? q.options : [];
      const crit = {};
      for (const o of optionsList) {
        const label = typeof o === "string" ? o : o.label;
        const desc = typeof o === "string" ? o : o.description || o.label;
        crit[label] = desc;
      }
      layaQuestions[q.id] = {
        t: "choice",
        ins: prompt,
        crit,
        meta: { originalType: "categorical", q, optionsList },
      };
    }
  }

  return layaQuestions;
}

export class Classifier {
  constructor(session, schema, layaQuestions) {
    this._session = session;
    this._schema = schema;
    this._layaQuestions = layaQuestions;
    this._contextWindow = session.window;
    this._destroyed = false;
  }

  static async availability(options = {}) {
    // If native Classifier is available on window, defer to it
    if (
      typeof window !== "undefined" &&
      window.Classifier &&
      window.Classifier !== Classifier &&
      typeof window.Classifier.availability === "function"
    ) {
      try {
        return await window.Classifier.availability(options);
      } catch {
        // Fall back to polyfill availability
      }
    }
    // With Laya on LiteRT.js, available once downloaded or ready in memory
    return cachedLayaSession ? "available" : "downloadable";
  }

  static async create(options) {
    if (!options || typeof options !== "object") {
      throw new TypeError("Failed to execute 'create' on 'Classifier': parameter 1 is not a dictionary.");
    }
    const { questions, context, monitor, signal, urls } = options;
    if (!Array.isArray(questions) || questions.length === 0) {
      throw new TypeError("Failed to execute 'create' on 'Classifier': 'questions' must be a non-empty array.");
    }
    if (signal !== undefined && !(signal instanceof AbortSignal)) {
      throw new TypeError("Failed to read the 'signal' property: The provided value is not an AbortSignal.");
    }
    signal?.throwIfAborted();

    // Validate questions before starting download
    for (const q of questions) {
      if (!q || typeof q !== "object" || !q.id || typeof q.id !== "string") {
        throw new TypeError("Each question must have a string 'id'.");
      }
    }

    const monitorTarget = typeof EventTarget !== "undefined" ? new EventTarget() : null;
    if (typeof monitor === "function" && monitorTarget) {
      monitor(monitorTarget);
    }

    const onProgress = (loaded) => {
      if (!monitorTarget) return;
      const event = typeof ProgressEvent !== "undefined"
        ? new ProgressEvent("downloadprogress", { lengthComputable: true, loaded, total: 1 })
        : { type: "downloadprogress", lengthComputable: true, loaded, total: 1 };
      monitorTarget.dispatchEvent?.(event);
    };

    const session = await getLaya(onProgress, urls);
    signal?.throwIfAborted();

    // Map Classifier questions to Laya format
    return new Classifier(session, options, toLayaQuestions(questions, context));
  }

  get contextWindow() {
    return this._contextWindow;
  }

  get contextUsage() {
    return 0;
  }

  async classify(input, options = {}) {
    this._assertAlive("classify");
    const { signal, context } = options;
    if (signal !== undefined && !(signal instanceof AbortSignal)) {
      throw new TypeError("Failed to read the 'signal' property: The provided value is not an AbortSignal.");
    }
    signal?.throwIfAborted();

    const stateText = context ? `${context}\n${String(input)}` : String(input);
    // Null-prototype for the same reason as toLayaQuestions(): every key here is a
    // caller-supplied question id, and a `__proto__` id on a normal object would reassign the
    // prototype instead of adding the decision, dropping it from the returned map.
    const results = Object.create(null);

    for (const [qid, qConfig] of Object.entries(this._layaQuestions)) {
      signal?.throwIfAborted();
      const { answer } = await this._session.decide(stateText, {
        t: qConfig.t,
        ins: qConfig.ins,
        crit: qConfig.crit,
      });

      const originalType = qConfig.meta.originalType;

      if (originalType === "binary") {
        const pTrue = answer.noul;
        const pFalse = Number((1 - pTrue).toFixed(6));
        const conf = answer.confidence ?? (1 - Math.abs(pTrue - 0.5) * 2);
        results[qid] = {
          id: qid,
          label: pTrue >= 0.5 ? "true" : "false",
          probability: pTrue,
          confidence: conf,
          probabilities: [
            { label: "true", probability: pTrue },
            { label: "false", probability: pFalse },
          ],
        };
      } else if (originalType === "ordinal") {
        const optionsList = qConfig.meta.optionsList;
        const probs = optionsList.map((o, idx) => {
          const label = typeof o === "string" ? o : o.label;
          const description = typeof o === "string" ? undefined : o.description;
          const p = answer.probabilities[String(idx)] ?? (answer.probabilities[label] || 0);
          return { label, description, probability: p };
        });

        // Compute expected score
        const levels = optionsList.map((o) => Number(typeof o === "string" ? o : o.label));
        const numeric = levels.every((n) => Number.isFinite(n));
        let expectedScore = answer.score;
        if (numeric) {
          expectedScore = probs.reduce((acc, p, idx) => acc + levels[idx] * p.probability, 0);
        }

        const sorted = [...probs].sort((a, b) => b.probability - a.probability);
        const winningLabel = sorted[0]?.label || String(Math.round(answer.score));

        results[qid] = {
          id: qid,
          label: winningLabel,
          confidence: answer.confidence,
          expectedScore: Number(expectedScore.toFixed(3)),
          probabilities: probs,
        };
      } else {
        // Categorical
        const optionsList = qConfig.meta.optionsList;
        const probs = optionsList.map((o) => {
          const label = typeof o === "string" ? o : o.label;
          const description = typeof o === "string" ? undefined : o.description;
          const p = answer.probabilities[label] ?? 0;
          return { label, description, probability: p };
        });

        results[qid] = {
          id: qid,
          label: answer.choice,
          confidence: answer.confidence,
          probabilities: probs,
        };
      }
    }

    return results;
  }

  async measureContextUsage(input, options = {}) {
    this._assertAlive("measureContextUsage");
    const stateText = options?.context ? `${options.context}\n${String(input)}` : String(input);
    const tokenized = this._session.encode(stateText);
    return tokenized.length;
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    this._session = null;
    this._layaQuestions = null;
  }

  _assertAlive(method) {
    if (this._destroyed) {
      throw new DOMException(
        `Failed to execute '${method}' on 'Classifier': The classifier session has been destroyed.`,
        "InvalidStateError",
      );
    }
  }

  /**
   * Installs window.Classifier polyfill globally if not already native.
   */
  static install() {
    if (typeof window === "undefined") return false;
    if ("Classifier" in window && window.Classifier !== undefined && window.Classifier !== Classifier) {
      return false;
    }
    Object.defineProperty(window, "Classifier", {
      value: Classifier,
      writable: true,
      configurable: true,
      enumerable: false,
    });
    window.webai = window.webai || {};
    window.webai.classifier = window.webai.classifier || {};
    window.webai.classifier.isPolyfilled = true;
    window.webai.classifier.diagnose = async () => ({
      isPolyfilled: true,
      hasClassifier: true,
      runtime: "Laya (LiteRT.js mmBERT)",
      status: cachedLayaSession ? "cached" : "downloadable",
    });
    return true;
  }
}
