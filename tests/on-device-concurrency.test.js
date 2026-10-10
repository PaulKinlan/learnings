// tests/on-device-concurrency.test.js
// Verification of on-device exercise runner serialization and ORT Web wasm EP non-reentrancy.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  evaluateExercises,
  EXERCISES,
  readout,
} from "../site/decision-models/on-device.js";

// Canonical mock distribution response for choice / noul / score exercises
function mockDistributionFor(exercise) {
  if (exercise.kind === "noul") {
    // [p(no), p(yes)]
    return exercise.expect === "yes" ? [0.05, 0.95] : [0.95, 0.05];
  }
  if (exercise.kind === "score") {
    // 5-point scale: [very negative, negative, neutral, positive, very positive]
    return exercise.expect.includes("below")
      ? [0.8, 0.15, 0.05, 0.0, 0.0]
      : [0.0, 0.0, 0.05, 0.15, 0.8];
  }
  // choice: options array
  const opts = exercise.question.options;
  const idx = opts.indexOf(exercise.expect);
  const dist = new Array(opts.length).fill(0.01);
  if (idx !== -1) dist[idx] = 0.9;
  return dist;
}

test("on-device runner: evaluateExercises serializes dispatches strictly without concurrency overlap (learnings-fzk)", async () => {
  let inFlight = 0;
  let maxInFlight = 0;
  const dispatchOrder = [];
  const completionOrder = [];

  const mockDecide = async (state, questions) => {
    inFlight++;
    maxInFlight = Math.max(maxInFlight, inFlight);
    const exerciseIndex = dispatchOrder.length;
    dispatchOrder.push(exerciseIndex);

    // Yield execution to the microtask queue so that if any caller had dispatched
    // concurrently via Promise.all, inFlight would increment beyond 1.
    await Promise.resolve();

    const exercise = EXERCISES[exerciseIndex];
    const dist = mockDistributionFor(exercise);

    completionOrder.push(exerciseIndex);
    inFlight--;

    return {
      distributions: [dist],
      ms: 15.5,
      tokens: 42,
      flat: false,
    };
  };

  const notifiedExercises = [];
  const results = await evaluateExercises(EXERCISES, mockDecide, ({ exercise, result, read }) => {
    notifiedExercises.push(exercise.name);
    assert.equal(inFlight, 0, "onResult callback is invoked only after the pass finishes and before next starts");
    assert.ok(result.ms > 0, "result includes latency");
    assert.ok(read.pass, `exercise ${exercise.name} should pass mock distribution`);
  });

  // Pinning strict serialization
  assert.equal(maxInFlight, 1, "decisions must be executed strictly serialized (max in flight must be 1)");
  assert.equal(results.length, EXERCISES.length, "all exercises must be evaluated");
  assert.equal(notifiedExercises.length, EXERCISES.length, "every exercise fires the onResult callback");

  // Pinning deterministic ordering
  assert.deepEqual(dispatchOrder, [...Array(EXERCISES.length).keys()], "dispatch order must match EXERCISES index order");
  assert.deepEqual(completionOrder, dispatchOrder, "completion order must match dispatch order");
  for (let i = 0; i < EXERCISES.length; i++) {
    assert.equal(results[i].exercise.name, EXERCISES[i].name, `result ${i} preserves exercise name`);
    assert.equal(notifiedExercises[i], EXERCISES[i].name, `notification ${i} preserves exercise name`);
  }
});

test("ORT Web wasm EP non-reentrancy contract: vendored JSEP wrapper enforces single active session mutex", () => {
  const jsepMjsPath = resolve("site/vendor/ort-wasm-simd-threaded.jsep.mjs");
  const source = readFileSync(jsepMjsPath, "utf8");

  // Verify the exact non-reentrant mutex in the vendored JSEP wasm runtime
  assert.ok(
    source.includes('throw Error("Session already started")'),
    "ort-wasm-simd-threaded.jsep.mjs contains explicit 'Session already started' mutex check",
  );
  assert.ok(
    source.includes('throw Error("Session mismatch")'),
    "ort-wasm-simd-threaded.jsep.mjs contains explicit 'Session mismatch' validation",
  );
  assert.ok(
    source.includes('f._OrtRun=da(f._OrtRun)'),
    "_OrtRun is wrapped by da() concurrency guard",
  );
});

