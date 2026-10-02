// @ts-check
/**
 * Neural Networks From First Principles — Raw JS (Float32Array) & Raw WebAssembly Engine
 *
 * Implements:
 * 1. In-memory compiled WebAssembly (wasm32) f32 GEMM kernel + Float32Array fallback
 * 2. Cooperative main-thread yielding via scheduler.yield() with 50ms frame deadline
 * 3. Perceptrons, Multi-Layer Perceptrons (MLPs) & analytical Backpropagation
 * 4. 2D Convolutional Neural Networks (Conv2D, ReLU, MaxPool2D, Linear)
 * 5. Micro-Transformer & Multi-Head Scaled Dot-Product Self-Attention
 * 6. Diffusion Models: Continuous 2D/Sprite DDPM & Discrete Masked Text Diffusion
 * 7. Decision Models & Single-Pass Pointer Readout Heads
 * 8. Composable, extensible Block Registry & Architecture Builder
 */

// ============================================================================
// 1. DETERMINISTIC PRNG & MATH HELPERS
// ============================================================================

export function createRng(seed = 42) {
  let state = (seed >>> 0) || 1;
  return {
    next() {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      return (state >>> 0) / 4294967296;
    },
    normal(mean = 0, std = 1) {
      const u1 = Math.max(1e-7, this.next());
      const u2 = this.next();
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      return mean + z * std;
    }
  };
}

/**
 * Cooperative main-thread yielding budget controller (50ms window).
 */
export function createYieldController(budgetMs = 50) {
  let deadline = performance.now() + budgetMs;
  return {
    async maybeYield() {
      if (performance.now() >= deadline) {
        // TODO(baseline/api.Scheduler.yield): Use scheduler.yield() directly without setTimeout fallback.
        if ("scheduler" in globalThis && "yield" in globalThis.scheduler) {
          await globalThis.scheduler.yield();
        } else {
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
        deadline = performance.now() + budgetMs;
        return true;
      }
      return false;
    },
    reset() {
      deadline = performance.now() + budgetMs;
    }
  };
}

// ============================================================================
// 2. RAW WEBASSEMBLY (wasm32) + RAW JS (Float32Array) GEMM BACKEND
// ============================================================================

/**
 * Builds a minimal, self-contained WebAssembly binary module exporting:
 * - `memory`: WebAssembly.Memory (initial 16 pages = 1 MB)
 * - `gemm_f32(aPtr, bPtr, cPtr, M, K, N)`: C[M,N] = A[M,K] * B[K,N]
 */
function uleb128(n) {
  const out = [];
  do {
    let byte = n & 0x7f;
    n >>>= 7;
    if (n !== 0) byte |= 0x80;
    out.push(byte);
  } while (n !== 0);
  return out;
}

export function buildWasmGemmBytes() {
  const body = [
    // Local declarations: 3x i32 (locals 6,7,8: i,j,k), 1x f32 (local 9: sum)
    0x02, 0x03, 0x7f, 0x01, 0x7d,
    // i = 0
    0x41, 0x00, 0x21, 0x06,
    0x02, 0x40, // block $break_i
      0x03, 0x40, // loop $loop_i
        0x20, 0x06, 0x20, 0x03, 0x4f, 0x0d, 0x01, // br_if $break_i (i >= M)
        // j = 0
        0x41, 0x00, 0x21, 0x07,
        0x02, 0x40, // block $break_j
          0x03, 0x40, // loop $loop_j
            0x20, 0x07, 0x20, 0x05, 0x4f, 0x0d, 0x01, // br_if $break_j (j >= N)
            // sum = 0.0
            0x43, 0x00, 0x00, 0x00, 0x00, 0x21, 0x09,
            // k = 0
            0x41, 0x00, 0x21, 0x08,
            0x02, 0x40, // block $break_k
              0x03, 0x40, // loop $loop_k
                0x20, 0x08, 0x20, 0x04, 0x4f, 0x0d, 0x01, // br_if $break_k (k >= K)
                // sum += A[i*K + k] * B[k*N + j]
                0x20, 0x09,
                // A addr: aPtr + ((i * K + k) << 2)
                0x20, 0x00, 0x20, 0x06, 0x20, 0x04, 0x6c, 0x20, 0x08, 0x6a, 0x41, 0x02, 0x74, 0x6a,
                0x2a, 0x02, 0x00, // f32.load
                // B addr: bPtr + ((k * N + j) << 2)
                0x20, 0x01, 0x20, 0x08, 0x20, 0x05, 0x6c, 0x20, 0x07, 0x6a, 0x41, 0x02, 0x74, 0x6a,
                0x2a, 0x02, 0x00, // f32.load
                0x94, // f32.mul
                0x92, // f32.add
                0x21, 0x09, // local.set $sum
                // k++
                0x20, 0x08, 0x41, 0x01, 0x6a, 0x21, 0x08,
                0x0c, 0x00, // br $loop_k
              0x0b,
            0x0b,
            // C[i*N + j] = sum
            0x20, 0x02, 0x20, 0x06, 0x20, 0x05, 0x6c, 0x20, 0x07, 0x6a, 0x41, 0x02, 0x74, 0x6a,
            0x20, 0x09,
            0x38, 0x02, 0x00, // f32.store
            // j++
            0x20, 0x07, 0x41, 0x01, 0x6a, 0x21, 0x07,
            0x0c, 0x00, // br $loop_j
          0x0b,
        0x0b,
        // i++
        0x20, 0x06, 0x41, 0x01, 0x6a, 0x21, 0x06,
        0x0c, 0x00, // br $loop_i
      0x0b,
    0x0b,
    0x0b // end function
  ];

  const codeSectionPayload = [0x01, ...uleb128(body.length), ...body];
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // magic "\0asm"
    0x01, 0x00, 0x00, 0x00, // version 1
    // Type section: 1 func type (i32, i32, i32, i32, i32, i32) -> ()
    0x01, 0x0a, 0x01, 0x60, 0x06, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00,
    // Function section: 1 function of type 0
    0x03, 0x02, 0x01, 0x00,
    // Memory section: 1 memory, min 16 pages (1 MB)
    0x05, 0x03, 0x01, 0x00, 0x10,
    // Export section: "memory" (mem 0) and "gemm_f32" (func 0)
    0x07, 0x15, 0x02,
      0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00,
      0x08, 0x67, 0x65, 0x6d, 0x6d, 0x5f, 0x66, 0x33, 0x32, 0x00, 0x00,
    // Code section
    0x0a, ...uleb128(codeSectionPayload.length), ...codeSectionPayload
  ]);
}

let wasmInstanceCache = null;

export function getWasmBackend() {
  if (wasmInstanceCache) return wasmInstanceCache;
  try {
    const bytes = buildWasmGemmBytes();
    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    const memory = /** @type {WebAssembly.Memory} */ (inst.exports.memory);
    const gemm_f32 = /** @type {Function} */ (inst.exports.gemm_f32);
    wasmInstanceCache = {
      available: true,
      byteLength: bytes.byteLength,
      matmul(A, B, M, K, N, out = new Float32Array(M * N)) {
        const neededBytes = (M * K + K * N + M * N) * 4;
        if (neededBytes > memory.buffer.byteLength) {
          const extraPages = Math.ceil((neededBytes - memory.buffer.byteLength) / 65536);
          memory.grow(extraPages);
        }
        const f32 = new Float32Array(memory.buffer);
        const aOffset = 0;
        const bOffset = M * K;
        const cOffset = bOffset + K * N;
        f32.set(A, aOffset);
        f32.set(B, bOffset);
        gemm_f32(aOffset * 4, bOffset * 4, cOffset * 4, M, K, N);
        out.set(f32.subarray(cOffset, cOffset + M * N));
        return out;
      }
    };
  } catch {
    wasmInstanceCache = {
      available: false,
      byteLength: 0,
      matmul: jsMatmul
    };
  }
  return wasmInstanceCache;
}

export function jsMatmul(A, B, M, K, N, out = new Float32Array(M * N)) {
  out.fill(0);
  for (let i = 0; i < M; i++) {
    const iK = i * K;
    const iN = i * N;
    for (let k = 0; k < K; k++) {
      const a = A[iK + k];
      const kN = k * N;
      for (let j = 0; j < N; j++) {
        out[iN + j] += a * B[kN + j];
      }
    }
  }
  return out;
}

let activeBackendMode = "wasm"; // "wasm" | "js"

export function setBackendMode(mode) {
  activeBackendMode = mode === "js" ? "js" : "wasm";
  return activeBackendMode;
}

export function getBackendMode() {
  return activeBackendMode;
}

export function matmul(A, B, M, K, N, out = new Float32Array(M * N)) {
  if (activeBackendMode === "wasm") {
    const wb = getWasmBackend();
    if (wb.available) return wb.matmul(A, B, M, K, N, out);
  }
  return jsMatmul(A, B, M, K, N, out);
}

/**
 * Benchmarks raw WASM vs raw JS Float32Array GEMM for an MxKxN matrix multiply.
 */
export function benchmarkBackends(size = 64, iterations = 25) {
  const rng = createRng(1337);
  const A = new Float32Array(size * size);
  const B = new Float32Array(size * size);
  const outWasm = new Float32Array(size * size);
  const outJs = new Float32Array(size * size);
  for (let i = 0; i < A.length; i++) {
    A[i] = rng.normal(0, 0.5);
    B[i] = rng.normal(0, 0.5);
  }
  const wb = getWasmBackend();
  // Warmup
  wb.matmul(A, B, size, size, size, outWasm);
  jsMatmul(A, B, size, size, size, outJs);

  const t0 = performance.now();
  for (let it = 0; it < iterations; it++) {
    wb.matmul(A, B, size, size, size, outWasm);
  }
  const wasmMs = Math.max(0.05, performance.now() - t0);

  const t1 = performance.now();
  for (let it = 0; it < iterations; it++) {
    jsMatmul(A, B, size, size, size, outJs);
  }
  const jsMs = Math.max(0.05, performance.now() - t1);

  let maxDiff = 0;
  for (let i = 0; i < outWasm.length; i++) {
    const d = Math.abs(outWasm[i] - outJs[i]);
    if (d > maxDiff) maxDiff = d;
  }

  const totalFlops = 2 * size * size * size * iterations;
  return {
    size,
    iterations,
    wasmMs,
    jsMs,
    wasmGflops: (totalFlops / (wasmMs * 1e6)),
    jsGflops: (totalFlops / (jsMs * 1e6)),
    maxDiff,
    wasmBytes: wb.byteLength
  };
}

// ============================================================================
// 3. ACTIVATIONS & LOSS FUNCTIONS
// ============================================================================

export const ACTIVATIONS = {
  relu: {
    label: "ReLU max(0, x)",
    fn(x) { return x > 0 ? x : 0; },
    grad(x) { return x > 0 ? 1 : 0; }
  },
  gelu: {
    label: "GELU x·σ(1.702x)",
    fn(x) {
      const s = 1 / (1 + Math.exp(-1.702 * x));
      return x * s;
    },
    grad(x) {
      const s = 1 / (1 + Math.exp(-1.702 * x));
      return s + x * 1.702 * s * (1 - s);
    }
  },
  tanh: {
    label: "Tanh(x)",
    fn(x) { return Math.tanh(x); },
    grad(x) {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
  },
  sigmoid: {
    label: "Sigmoid σ(x)",
    fn(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); },
    grad(x) {
      const s = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
      return s * (1 - s);
    }
  }
};

export function softmaxInPlace(logits, offset = 0, length = logits.length, temperature = 1.0) {
  let maxVal = -Infinity;
  const invT = 1.0 / Math.max(1e-4, temperature);
  for (let i = 0; i < length; i++) {
    const v = logits[offset + i] * invT;
    if (v > maxVal) maxVal = v;
  }
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const e = Math.exp(logits[offset + i] * invT - maxVal);
    logits[offset + i] = e;
    sum += e;
  }
  const invSum = 1.0 / (sum || 1);
  for (let i = 0; i < length; i++) {
    logits[offset + i] *= invSum;
  }
  return logits;
}

export function l2Norm(arr) {
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) sumSq += arr[i] * arr[i];
  return Math.sqrt(sumSq);
}

// ============================================================================
// 4. CHAPTER 1: PERCEPTRONS, FEEDFORWARD MLPs & BACKPROPAGATION
// ============================================================================

export function generate2DDataset(kind = "xor", count = 120, seed = 42) {
  const rng = createRng(seed);
  const X = new Float32Array(count * 2);
  const Y = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    let x1 = 0;
    let x2 = 0;
    let label = 0;

    if (kind === "xor") {
      const q = i % 4;
      x1 = (q & 1 ? 0.55 : -0.55) + rng.normal(0, 0.18);
      x2 = (q & 2 ? 0.55 : -0.55) + rng.normal(0, 0.18);
      label = ((q === 1 || q === 2) ? 1 : 0);
    } else if (kind === "circle") {
      const inner = i % 2 === 0;
      const r = inner ? rng.next() * 0.38 : 0.58 + rng.next() * 0.28;
      const angle = rng.next() * Math.PI * 2;
      x1 = r * Math.cos(angle);
      x2 = r * Math.sin(angle);
      label = inner ? 1 : 0;
    } else if (kind === "spiral") {
      const arm = i % 2;
      const frac = Math.floor(i / 2) / Math.max(1, count / 2);
      const r = 0.12 + frac * 0.78;
      const theta = frac * 2.6 * Math.PI + (arm ? Math.PI : 0) + rng.normal(0, 0.09);
      x1 = r * Math.cos(theta);
      x2 = r * Math.sin(theta);
      label = arm;
    } else {
      // moons
      const upper = i % 2 === 0;
      const angle = rng.next() * Math.PI;
      if (upper) {
        x1 = Math.cos(angle) * 0.65 - 0.2 + rng.normal(0, 0.08);
        x2 = Math.sin(angle) * 0.65 - 0.1 + rng.normal(0, 0.08);
        label = 0;
      } else {
        x1 = (1 - Math.cos(angle)) * 0.65 - 0.45 + rng.normal(0, 0.08);
        x2 = (-Math.sin(angle) * 0.65) + 0.25 + rng.normal(0, 0.08);
        label = 1;
      }
    }

    X[i * 2] = Math.max(-1, Math.min(1, x1));
    X[i * 2 + 1] = Math.max(-1, Math.min(1, x2));
    Y[i] = label;
  }

  return { X, Y, count };
}

export class MLPNetwork {
  /**
   * @param {number[]} layerSizes e.g. [2, 8, 8, 1]
   * @param {{ activation?: keyof typeof ACTIVATIONS, seed?: number }} [opts]
   */
  constructor(layerSizes = [2, 8, 8, 1], opts = {}) {
    this.layerSizes = [...layerSizes];
    this.activationName = opts.activation || "gelu";
    this.rng = createRng(opts.seed || 101);
    this.stepCount = 0;
    this.layers = [];

    for (let l = 0; l < layerSizes.length - 1; l++) {
      const inDim = layerSizes[l];
      const outDim = layerSizes[l + 1];
      const scale = Math.sqrt(2.0 / (inDim + outDim));
      const W = new Float32Array(inDim * outDim);
      const b = new Float32Array(outDim);
      for (let i = 0; i < W.length; i++) W[i] = this.rng.normal(0, scale);
      this.layers.push({
        inDim,
        outDim,
        W,
        b,
        dW: new Float32Array(inDim * outDim),
        db: new Float32Array(outDim),
        mW: new Float32Array(inDim * outDim),
        vW: new Float32Array(inDim * outDim),
        mb: new Float32Array(outDim),
        vb: new Float32Array(outDim),
        gradNorm: 0,
        actNorm: 0
      });
    }
  }

  /**
   * Forward pass on batch X [N, inDim].
   */
  forward(X, N) {
    const act = ACTIVATIONS[this.activationName] || ACTIVATIONS.gelu;
    const caches = [{ a: X, z: X }];
    let curA = X;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const isLast = l === this.layers.length - 1;
      const Z = matmul(curA, layer.W, N, layer.inDim, layer.outDim);
      const A = new Float32Array(N * layer.outDim);

      for (let i = 0; i < N; i++) {
        const rowOffset = i * layer.outDim;
        for (let j = 0; j < layer.outDim; j++) {
          const zVal = Z[rowOffset + j] + layer.b[j];
          Z[rowOffset + j] = zVal;
          A[rowOffset + j] = isLast ? ACTIVATIONS.sigmoid.fn(zVal) : act.fn(zVal);
        }
      }
      layer.actNorm = l2Norm(A) / Math.sqrt(N);
      caches.push({ z: Z, a: A });
      curA = A;
    }

    return { probs: curA, caches };
  }

  /**
   * Executes one full forward + analytical backward + optimizer step on dataset {X, Y, count}.
   */
  trainStep(dataset, { lr = 0.05, optimizer = "adamw", weightDecay = 1e-3 } = {}) {
    const { X, Y, count: N } = dataset;
    const act = ACTIVATIONS[this.activationName] || ACTIVATIONS.gelu;
    const { probs, caches } = this.forward(X, N);

    // Binary Cross-Entropy loss + accuracy
    let loss = 0;
    let correct = 0;
    // dZ for output layer (sigmoid + BCE simplifies cleanly to (p - y) / N)
    let dZ = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const p = Math.max(1e-6, Math.min(1 - 1e-6, probs[i]));
      const y = Y[i];
      loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      if ((p >= 0.5 ? 1 : 0) === y) correct++;
      dZ[i] = (p - y) / N;
    }
    loss /= N;

    // Backpropagate layer by layer from output to input
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const prevA = caches[l].a; // [N, inDim]
      const outDim = layer.outDim;
      const inDim = layer.inDim;

      layer.dW.fill(0);
      layer.db.fill(0);

      for (let n = 0; n < N; n++) {
        const nIn = n * inDim;
        const nOut = n * outDim;
        for (let j = 0; j < outDim; j++) {
          const dz = dZ[nOut + j];
          layer.db[j] += dz;
          for (let i = 0; i < inDim; i++) {
            layer.dW[i * outDim + j] += prevA[nIn + i] * dz;
          }
        }
      }

      layer.gradNorm = l2Norm(layer.dW);

      if (l > 0) {
        const prevZ = caches[l].z;
        const dPrevZ = new Float32Array(N * inDim);
        for (let n = 0; n < N; n++) {
          const nIn = n * inDim;
          const nOut = n * outDim;
          for (let i = 0; i < inDim; i++) {
            let sum = 0;
            for (let j = 0; j < outDim; j++) {
              sum += dZ[nOut + j] * layer.W[i * outDim + j];
            }
            dPrevZ[nIn + i] = sum * act.grad(prevZ[nIn + i]);
          }
        }
        dZ = dPrevZ;
      }
    }

    // Apply Optimizer Update
    this.stepCount++;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(beta1, this.stepCount);
    const bc2 = 1 - Math.pow(beta2, this.stepCount);

    for (const layer of this.layers) {
      for (let i = 0; i < layer.W.length; i++) {
        const g = layer.dW[i];
        if (optimizer === "sgd") {
          layer.W[i] -= lr * (g + weightDecay * layer.W[i]);
        } else if (optimizer === "momentum") {
          layer.mW[i] = beta1 * layer.mW[i] + g;
          layer.W[i] -= lr * (layer.mW[i] + weightDecay * layer.W[i]);
        } else {
          // AdamW
          layer.mW[i] = beta1 * layer.mW[i] + (1 - beta1) * g;
          layer.vW[i] = beta2 * layer.vW[i] + (1 - beta2) * g * g;
          const mHat = layer.mW[i] / bc1;
          const vHat = layer.vW[i] / bc2;
          layer.W[i] = layer.W[i] * (1 - lr * weightDecay) - lr * (mHat / (Math.sqrt(vHat) + eps));
        }
      }
      for (let j = 0; j < layer.b.length; j++) {
        const g = layer.db[j];
        if (optimizer === "sgd") {
          layer.b[j] -= lr * g;
        } else if (optimizer === "momentum") {
          layer.mb[j] = beta1 * layer.mb[j] + g;
          layer.b[j] -= lr * layer.mb[j];
        } else {
          layer.mb[j] = beta1 * layer.mb[j] + (1 - beta1) * g;
          layer.vb[j] = beta2 * layer.vb[j] + (1 - beta2) * g * g;
          const mHat = layer.mb[j] / bc1;
          const vHat = layer.vb[j] / bc2;
          layer.b[j] -= lr * (mHat / (Math.sqrt(vHat) + eps));
        }
      }
    }

    return {
      step: this.stepCount,
      loss,
      accuracy: correct / N,
      gradNorms: this.layers.map((l) => l.gradNorm),
      actNorms: this.layers.map((l) => l.actNorm)
    };
  }

  predictPoint(x1, x2) {
    const input = new Float32Array([x1, x2]);
    return this.forward(input, 1).probs[0];
  }
}

// ============================================================================
// 5. CHAPTER 2: CONVOLUTIONAL NEURAL NETWORKS (CNNs)
// ============================================================================

export const GLYPH_CLASSES = [
  { id: 0, name: "Vertical Bar |", pattern: [
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0
  ] },
  { id: 1, name: "Horizontal Bar —", pattern: [
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
  ] },
  { id: 2, name: "Diagonal Slash /", pattern: [
    0,0,0,0,0,0,1,1,
    0,0,0,0,0,1,1,0,
    0,0,0,0,1,1,0,0,
    0,0,0,1,1,0,0,0,
    0,0,1,1,0,0,0,0,
    0,1,1,0,0,0,0,0,
    1,1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0
  ] },
  { id: 3, name: "Hollow Box □", pattern: [
    0,1,1,1,1,1,1,0,
    0,1,1,1,1,1,1,0,
    0,1,1,0,0,1,1,0,
    0,1,1,0,0,1,1,0,
    0,1,1,0,0,1,1,0,
    0,1,1,0,0,1,1,0,
    0,1,1,1,1,1,1,0,
    0,1,1,1,1,1,1,0
  ] }
];

export function generateGlyphDataset(samplesPerClass = 10, seed = 77) {
  const rng = createRng(seed);
  const samples = [];
  for (let c = 0; c < GLYPH_CLASSES.length; c++) {
    const base = GLYPH_CLASSES[c].pattern;
    for (let s = 0; s < samplesPerClass; s++) {
      const img = new Float32Array(64);
      const shiftX = s === 0 ? 0 : (Math.floor(rng.next() * 3) - 1);
      const shiftY = s === 0 ? 0 : (Math.floor(rng.next() * 3) - 1);
      for (let r = 0; r < 8; r++) {
        for (let col = 0; col < 8; col++) {
          const sr = Math.max(0, Math.min(7, r - shiftY));
          const sc = Math.max(0, Math.min(7, col - shiftX));
          const noise = s === 0 ? 0 : (rng.next() - 0.5) * 0.15;
          img[r * 8 + col] = Math.max(0, Math.min(1, base[sr * 8 + sc] + noise));
        }
      }
      samples.push({ img, label: c });
    }
  }
  return samples;
}

/**
 * First-principles 2D CNN:
 * Input (8x8) -> Conv2D(4 filters, 3x3, stride 1 -> 4x6x6) -> ReLU -> MaxPool2D(2x2 -> 4x3x3=36) -> Linear(36 -> 4)
 */
export class TinyCNN {
  constructor(seed = 2026) {
    this.rng = createRng(seed);
    this.numFilters = 4;
    this.kernelSize = 3;
    this.numClasses = 4;
    this.stepCount = 0;

    // Initialize 4 filters with interpretable directional seeds + small noise
    this.convW = new Float32Array(this.numFilters * 9);
    this.convB = new Float32Array(this.numFilters);
    for (let i = 0; i < this.convW.length; i++) {
      this.convW[i] = this.rng.normal(0, 0.35);
    }

    // Linear head: 36 -> 4
    this.fcW = new Float32Array(36 * 4);
    this.fcB = new Float32Array(4);
    for (let i = 0; i < this.fcW.length; i++) {
      this.fcW[i] = this.rng.normal(0, Math.sqrt(2 / 36));
    }
  }

  forward(img) {
    // 1. Conv2D (8x8 -> 4 x 6x6) + ReLU
    const convZ = new Float32Array(4 * 36);
    const convA = new Float32Array(4 * 36);
    for (let f = 0; f < 4; f++) {
      const fOff = f * 9;
      const outOff = f * 36;
      const bias = this.convB[f];
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          let sum = bias;
          for (let kr = 0; kr < 3; kr++) {
            for (let kc = 0; kc < 3; kc++) {
              sum += img[(r + kr) * 8 + (c + kc)] * this.convW[fOff + kr * 3 + kc];
            }
          }
          const idx = outOff + r * 6 + c;
          convZ[idx] = sum;
          convA[idx] = sum > 0 ? sum : 0;
        }
      }
    }

    // 2. MaxPool2D 2x2 stride 2 (4 x 6x6 -> 4 x 3x3 = 36)
    const pooled = new Float32Array(36);
    const poolArgmax = new Int32Array(36);
    for (let f = 0; f < 4; f++) {
      const inOff = f * 36;
      const pOff = f * 9;
      for (let pr = 0; pr < 3; pr++) {
        for (let pc = 0; pc < 3; pc++) {
          let bestVal = -Infinity;
          let bestIdx = 0;
          for (let dr = 0; dr < 2; dr++) {
            for (let dc = 0; dc < 2; dc++) {
              const idx = inOff + (pr * 2 + dr) * 6 + (pc * 2 + dc);
              if (convA[idx] > bestVal) {
                bestVal = convA[idx];
                bestIdx = idx;
              }
            }
          }
          const outIdx = pOff + pr * 3 + pc;
          pooled[outIdx] = bestVal;
          poolArgmax[outIdx] = bestIdx;
        }
      }
    }

    // 3. Linear Head (36 -> 4) via matmul
    const logits = matmul(pooled, this.fcW, 1, 36, 4);
    for (let c = 0; c < 4; c++) logits[c] += this.fcB[c];
    const probs = new Float32Array(logits);
    softmaxInPlace(probs, 0, 4);

    return { convZ, convA, pooled, poolArgmax, logits, probs };
  }

  trainEpoch(samples, lr = 0.06) {
    const dConvW = new Float32Array(this.convW.length);
    const dConvB = new Float32Array(this.convB.length);
    const dFcW = new Float32Array(this.fcW.length);
    const dFcB = new Float32Array(this.fcB.length);

    let totalLoss = 0;
    let correct = 0;
    const N = samples.length;

    for (const { img, label } of samples) {
      const { convZ, pooled, poolArgmax, probs } = this.forward(img);
      totalLoss += -Math.log(Math.max(1e-7, probs[label]));
      let pred = 0;
      for (let c = 1; c < 4; c++) if (probs[c] > probs[pred]) pred = c;
      if (pred === label) correct++;

      // dLogits = (probs - oneHot(label)) / N
      const dLogits = new Float32Array(4);
      for (let c = 0; c < 4; c++) {
        dLogits[c] = (probs[c] - (c === label ? 1 : 0)) / N;
        dFcB[c] += dLogits[c];
      }

      // Backprop through FC
      const dPooled = new Float32Array(36);
      for (let i = 0; i < 36; i++) {
        for (let c = 0; c < 4; c++) {
          dFcW[i * 4 + c] += pooled[i] * dLogits[c];
          dPooled[i] += this.fcW[i * 4 + c] * dLogits[c];
        }
      }

      // Backprop through MaxPool2D + ReLU
      const dConvZ = new Float32Array(4 * 36);
      for (let i = 0; i < 36; i++) {
        const winIdx = poolArgmax[i];
        if (convZ[winIdx] > 0) {
          dConvZ[winIdx] += dPooled[i];
        }
      }

      // Backprop through Conv2D
      for (let f = 0; f < 4; f++) {
        const fOff = f * 9;
        const outOff = f * 36;
        for (let r = 0; r < 6; r++) {
          for (let c = 0; c < 6; c++) {
            const dz = dConvZ[outOff + r * 6 + c];
            if (dz === 0) continue;
            dConvB[f] += dz;
            for (let kr = 0; kr < 3; kr++) {
              for (let kc = 0; kc < 3; kc++) {
                dConvW[fOff + kr * 3 + kc] += img[(r + kr) * 8 + (c + kc)] * dz;
              }
            }
          }
        }
      }
    }

    for (let i = 0; i < this.convW.length; i++) this.convW[i] -= lr * dConvW[i];
    for (let i = 0; i < this.convB.length; i++) this.convB[i] -= lr * dConvB[i];
    for (let i = 0; i < this.fcW.length; i++) this.fcW[i] -= lr * dFcW[i];
    for (let i = 0; i < this.fcB.length; i++) this.fcB[i] -= lr * dFcB[i];

    this.stepCount++;
    return {
      step: this.stepCount,
      loss: totalLoss / N,
      accuracy: correct / N
    };
  }
}

// ============================================================================
// 6. CHAPTER 3: TRANSFORMERS & SCALED DOT-PRODUCT SELF-ATTENTION
// ============================================================================

export const VOCAB_TOKENS = ["A", "B", "C", "D", "E", "F", "G", "H"];

export function generateSequenceBatch(task = "reverse", seqLen = 4, count = 32, seed = 99) {
  const rng = createRng(seed);
  const pairs = [];
  for (let n = 0; n < count; n++) {
    const input = [];
    for (let t = 0; t < seqLen; t++) {
      input.push(Math.floor(rng.next() * VOCAB_TOKENS.length));
    }
    let target;
    if (task === "reverse") {
      target = [...input].reverse();
    } else if (task === "sort") {
      target = [...input].sort((a, b) => a - b);
    } else {
      // copy
      target = [...input];
    }
    pairs.push({ input, target });
  }
  return pairs;
}

/**
 * First-principles Micro-Transformer with Multi-Head Scaled Dot-Product Attention:
 * Token Embed + Positional Embed -> Multi-Head Attention (2 heads) -> Residual -> Linear Output Head
 */
export class MicroTransformer {
  constructor({ vocabSize = 8, seqLen = 4, dModel = 16, numHeads = 2, causal = false, seed = 314 } = {}) {
    this.vocabSize = vocabSize;
    this.seqLen = seqLen;
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.headDim = dModel / numHeads;
    this.causal = causal;
    this.stepCount = 0;
    const rng = createRng(seed);

    const scale = Math.sqrt(2.0 / dModel);
    this.tokEmbed = new Float32Array(vocabSize * dModel);
    this.posEmbed = new Float32Array(seqLen * dModel);
    this.Wq = new Float32Array(dModel * dModel);
    this.Wk = new Float32Array(dModel * dModel);
    this.Wv = new Float32Array(dModel * dModel);
    this.Wo = new Float32Array(dModel * dModel);
    this.Wout = new Float32Array(dModel * vocabSize);
    this.bout = new Float32Array(vocabSize);

    for (const arr of [this.tokEmbed, this.posEmbed, this.Wq, this.Wk, this.Wv, this.Wo, this.Wout]) {
      for (let i = 0; i < arr.length; i++) arr[i] = rng.normal(0, scale * 0.6);
    }
  }

  forward(tokens) {
    const T = tokens.length;
    const D = this.dModel;
    const H = this.numHeads;
    const Dh = this.headDim;
    const V = this.vocabSize;

    // 1. Embeddings X [T, D]
    const X = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const tok = tokens[t];
      for (let d = 0; d < D; d++) {
        X[t * D + d] = this.tokEmbed[tok * D + d] + this.posEmbed[t * D + d];
      }
    }

    // 2. Q, K, V projections [T, D]
    const Q = matmul(X, this.Wq, T, D, D);
    const K = matmul(X, this.Wk, T, D, D);
    const Val = matmul(X, this.Wv, T, D, D);

    // 3. Multi-Head Scaled Dot-Product Attention
    const attnWeights = []; // [H][T * T]
    const headContext = new Float32Array(T * D);
    const invSqrtDh = 1.0 / Math.sqrt(Dh);

    for (let h = 0; h < H; h++) {
      const hOffset = h * Dh;
      const weights = new Float32Array(T * T);
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (this.causal && j > i) {
            weights[i * T + j] = -1e9;
            continue;
          }
          let dot = 0;
          for (let d = 0; d < Dh; d++) {
            dot += Q[i * D + hOffset + d] * K[j * D + hOffset + d];
          }
          weights[i * T + j] = dot * invSqrtDh;
        }
        softmaxInPlace(weights, i * T, T);
      }
      attnWeights.push(weights);

      // Multiply weights [T, T] by V_h [T, Dh]
      for (let i = 0; i < T; i++) {
        for (let d = 0; d < Dh; d++) {
          let sum = 0;
          for (let j = 0; j < T; j++) {
            sum += weights[i * T + j] * Val[j * D + hOffset + d];
          }
          headContext[i * D + hOffset + d] = sum;
        }
      }
    }

    // 4. Output projection + Residual connection
    const attnOut = matmul(headContext, this.Wo, T, D, D);
    const hidden = new Float32Array(T * D);
    for (let i = 0; i < hidden.length; i++) {
      hidden[i] = X[i] + attnOut[i];
    }

    // 5. Logits & Softmax probabilities [T, V]
    const logits = matmul(hidden, this.Wout, T, D, V);
    const probs = new Float32Array(T * V);
    const predictions = [];
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < V; v++) {
        logits[t * V + v] += this.bout[v];
        probs[t * V + v] = logits[t * V + v];
      }
      softmaxInPlace(probs, t * V, V);
      let bestV = 0;
      for (let v = 1; v < V; v++) {
        if (probs[t * V + v] > probs[t * V + bestV]) bestV = v;
      }
      predictions.push(bestV);
    }

    return { X, Q, K, Val, attnWeights, headContext, hidden, logits, probs, predictions };
  }

  trainEpoch(batch, lr = 0.08) {
    const T = this.seqLen;
    const D = this.dModel;
    const H = this.numHeads;
    const Dh = this.headDim;
    const V = this.vocabSize;
    const N = batch.length;
    const invSqrtDh = 1.0 / Math.sqrt(Dh);

    const dTok = new Float32Array(this.tokEmbed.length);
    const dPos = new Float32Array(this.posEmbed.length);
    const dWq = new Float32Array(this.Wq.length);
    const dWk = new Float32Array(this.Wk.length);
    const dWv = new Float32Array(this.Wv.length);
    const dWo = new Float32Array(this.Wo.length);
    const dWout = new Float32Array(this.Wout.length);
    const dbout = new Float32Array(this.bout.length);

    let totalLoss = 0;
    let totalTokens = 0;
    let correctTokens = 0;

    for (const { input, target } of batch) {
      const { X, Q, K, Val, attnWeights, headContext, hidden, probs, predictions } = this.forward(input);

      const dLogits = new Float32Array(T * V);
      for (let t = 0; t < T; t++) {
        const y = target[t];
        totalLoss += -Math.log(Math.max(1e-7, probs[t * V + y]));
        totalTokens++;
        if (predictions[t] === y) correctTokens++;
        for (let v = 0; v < V; v++) {
          const g = (probs[t * V + v] - (v === y ? 1 : 0)) / (N * T);
          dLogits[t * V + v] = g;
          dbout[v] += g;
        }
      }

      // dHidden = dLogits * Wout^T, dWout += hidden^T * dLogits
      const dHidden = new Float32Array(T * D);
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) {
          let sum = 0;
          for (let v = 0; v < V; v++) {
            const g = dLogits[t * V + v];
            dWout[d * V + v] += hidden[t * D + d] * g;
            sum += this.Wout[d * V + v] * g;
          }
          dHidden[t * D + d] = sum;
        }
      }

      // Residual split: dX = dHidden, dAttnOut = dHidden
      const dX = new Float32Array(dHidden);
      const dHeadContext = new Float32Array(T * D);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < D; i++) {
          let sum = 0;
          for (let j = 0; j < D; j++) {
            const g = dHidden[t * D + j];
            dWo[i * D + j] += headContext[t * D + i] * g;
            sum += this.Wo[i * D + j] * g;
          }
          dHeadContext[t * D + i] = sum;
        }
      }

      // Backprop through Multi-Head Attention
      const dQ = new Float32Array(T * D);
      const dK = new Float32Array(T * D);
      const dVal = new Float32Array(T * D);

      for (let h = 0; h < H; h++) {
        const hOffset = h * Dh;
        const W = attnWeights[h];
        const dW = new Float32Array(T * T);

        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            let sum = 0;
            for (let d = 0; d < Dh; d++) {
              const dhc = dHeadContext[i * D + hOffset + d];
              sum += dhc * Val[j * D + hOffset + d];
              dVal[j * D + hOffset + d] += W[i * T + j] * dhc;
            }
            dW[i * T + j] = sum;
          }

          // Softmax Jacobian per row i: dScore_j = W_j * (dW_j - sum_m W_m * dW_m)
          let rowDot = 0;
          for (let j = 0; j < T; j++) rowDot += W[i * T + j] * dW[i * T + j];
          for (let j = 0; j < T; j++) {
            if (this.causal && j > i) continue;
            const dScore = W[i * T + j] * (dW[i * T + j] - rowDot) * invSqrtDh;
            for (let d = 0; d < Dh; d++) {
              dQ[i * D + hOffset + d] += dScore * K[j * D + hOffset + d];
              dK[j * D + hOffset + d] += dScore * Q[i * D + hOffset + d];
            }
          }
        }
      }

      // Backprop Q, K, Val to X and Wq, Wk, Wv
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < D; i++) {
          const xVal = X[t * D + i];
          let sumX = 0;
          for (let j = 0; j < D; j++) {
            const dq = dQ[t * D + j];
            const dk = dK[t * D + j];
            const dv = dVal[t * D + j];
            dWq[i * D + j] += xVal * dq;
            dWk[i * D + j] += xVal * dk;
            dWv[i * D + j] += xVal * dv;
            sumX += this.Wq[i * D + j] * dq + this.Wk[i * D + j] * dk + this.Wv[i * D + j] * dv;
          }
          dX[t * D + i] += sumX;
        }
      }

      // Accumulate into token and positional embeddings
      for (let t = 0; t < T; t++) {
        const tok = input[t];
        for (let d = 0; d < D; d++) {
          const gx = dX[t * D + d];
          dTok[tok * D + d] += gx;
          dPos[t * D + d] += gx;
        }
      }
    }

    const applyUpdate = (param, grad) => {
      for (let i = 0; i < param.length; i++) param[i] -= lr * grad[i];
    };
    applyUpdate(this.tokEmbed, dTok);
    applyUpdate(this.posEmbed, dPos);
    applyUpdate(this.Wq, dWq);
    applyUpdate(this.Wk, dWk);
    applyUpdate(this.Wv, dWv);
    applyUpdate(this.Wo, dWo);
    applyUpdate(this.Wout, dWout);
    applyUpdate(this.bout, dbout);

    this.stepCount++;
    return {
      step: this.stepCount,
      loss: totalLoss / totalTokens,
      accuracy: correctTokens / totalTokens
    };
  }
}

// ============================================================================
// 7. CHAPTER 4: DIFFUSION MODELS (IMAGE/PIXEL DDPM + MASKED TEXT DIFFUSION)
// ============================================================================

export const SPRITE_TARGETS = {
  smiley: new Float32Array([
    0, 1, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1,
    1, 0, 1, 1, 0, 1,
    1, 1, 0, 0, 1, 1,
    0, 1, 1, 1, 1, 0
  ]),
  cross: new Float32Array([
    0, 0, 1, 1, 0, 0,
    0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 0, 0,
    0, 0, 1, 1, 0, 0
  ]),
  diamond: new Float32Array([
    0, 0, 1, 1, 0, 0,
    0, 1, 1, 1, 1, 0,
    1, 1, 0, 0, 1, 1,
    1, 1, 0, 0, 1, 1,
    0, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 0, 0
  ])
};

/**
 * Continuous Pixel/Image DDPM (Denoising Diffusion Probabilistic Model) on a 6x6 canvas (36 dims).
 * Learns noise prediction \epsilon_\theta(x_t, t) across T = 16 diffusion steps.
 */
export class MicroDDPM {
  constructor({ dim = 36, steps = 16, hiddenDim = 48, seed = 512 } = {}) {
    this.dim = dim;
    this.steps = steps;
    this.hiddenDim = hiddenDim;
    this.stepCount = 0;
    this.rng = createRng(seed);

    // Linear variance schedule \beta_t \in [0.02, 0.28]
    this.betas = new Float32Array(steps + 1);
    this.alphas = new Float32Array(steps + 1);
    this.alphaBars = new Float32Array(steps + 1);
    this.alphaBars[0] = 1.0;

    for (let t = 1; t <= steps; t++) {
      const b = 0.02 + ((t - 1) / Math.max(1, steps - 1)) * 0.24;
      this.betas[t] = b;
      this.alphas[t] = 1 - b;
      this.alphaBars[t] = this.alphaBars[t - 1] * this.alphas[t];
    }

    // 2-layer MLP denoiser taking [x_t (dim), sin(t), cos(t), t/T] -> dim
    const inDim = dim + 3;
    this.inDim = inDim;
    this.W1 = new Float32Array(inDim * hiddenDim);
    this.b1 = new Float32Array(hiddenDim);
    this.W2 = new Float32Array(hiddenDim * dim);
    this.b2 = new Float32Array(dim);

    for (let i = 0; i < this.W1.length; i++) this.W1[i] = this.rng.normal(0, Math.sqrt(2 / inDim));
    for (let i = 0; i < this.W2.length; i++) this.W2[i] = this.rng.normal(0, Math.sqrt(2 / hiddenDim) * 0.5);
  }

  /**
   * Closed-form forward noise addition q(x_t | x_0).
   */
  qSample(x0, t, noise = null) {
    const eps = noise || new Float32Array(this.dim);
    if (!noise) {
      for (let i = 0; i < this.dim; i++) eps[i] = this.rng.normal(0, 1);
    }
    const s1 = Math.sqrt(this.alphaBars[t]);
    const s2 = Math.sqrt(1 - this.alphaBars[t]);
    const xt = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      // Scale clean pixels [0,1] to [-1,1]
      const clean = x0[i] * 2 - 1;
      xt[i] = s1 * clean + s2 * eps[i];
    }
    return { xt, eps };
  }

  predictNoise(xt, t) {
    const D = this.dim;
    const H = this.hiddenDim;
    const inp = new Float32Array(this.inDim);
    inp.set(xt, 0);
    const tau = t / this.steps;
    inp[D] = tau;
    inp[D + 1] = Math.sin(tau * Math.PI);
    inp[D + 2] = Math.cos(tau * Math.PI);

    const z1 = matmul(inp, this.W1, 1, this.inDim, H);
    const h1 = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      z1[j] += this.b1[j];
      h1[j] = ACTIVATIONS.gelu.fn(z1[j]);
    }
    const predEps = matmul(h1, this.W2, 1, H, D);
    for (let i = 0; i < D; i++) predEps[i] += this.b2[i];

    return { inp, z1, h1, predEps };
  }

  trainStep(x0, lr = 0.03, batchSize = 16) {
    const D = this.dim;
    const H = this.hiddenDim;
    const dW1 = new Float32Array(this.W1.length);
    const db1 = new Float32Array(this.b1.length);
    const dW2 = new Float32Array(this.W2.length);
    const db2 = new Float32Array(this.b2.length);

    let totalMse = 0;

    for (let b = 0; b < batchSize; b++) {
      const t = (b % this.steps) + 1;
      const { xt, eps } = this.qSample(x0, t);
      const { inp, z1, h1, predEps } = this.predictNoise(xt, t);

      const dOut = new Float32Array(D);
      for (let i = 0; i < D; i++) {
        const diff = predEps[i] - eps[i];
        totalMse += diff * diff;
        dOut[i] = (2 * diff) / batchSize;
        db2[i] += dOut[i];
      }

      const dh1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        let sum = 0;
        for (let i = 0; i < D; i++) {
          dW2[j * D + i] += h1[j] * dOut[i];
          sum += this.W2[j * D + i] * dOut[i];
        }
        dh1[j] = sum * ACTIVATIONS.gelu.grad(z1[j]);
        db1[j] += dh1[j];
      }

      for (let k = 0; k < this.inDim; k++) {
        const ik = inp[k];
        for (let j = 0; j < H; j++) {
          dW1[k * H + j] += ik * dh1[j];
        }
      }
    }

    for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * dW1[i];
    for (let i = 0; i < this.b1.length; i++) this.b1[i] -= lr * db1[i];
    for (let i = 0; i < this.W2.length; i++) this.W2[i] -= lr * dW2[i];
    for (let i = 0; i < this.b2.length; i++) this.b2[i] -= lr * db2[i];

    this.stepCount++;
    return {
      step: this.stepCount,
      loss: totalMse / (batchSize * D)
    };
  }

  /**
   * Runs the full reverse DDPM denoising chain x_T -> x_0 and returns snapshots.
   */
  sampleTrajectory(seed = 808) {
    const sampleRng = createRng(seed);
    let xt = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) xt[i] = sampleRng.normal(0, 1);

    const frames = [{ step: this.steps, pixels: this.toPixels(xt) }];

    for (let t = this.steps; t >= 1; t--) {
      const { predEps } = this.predictNoise(xt, t);
      const alpha = this.alphas[t];
      const alphaBar = this.alphaBars[t];
      const beta = this.betas[t];
      const coeff = beta / Math.sqrt(Math.max(1e-6, 1 - alphaBar));
      const invSqrtAlpha = 1 / Math.sqrt(alpha);
      const sigma = t > 1 ? Math.sqrt(beta) * 0.25 : 0;

      const prev = new Float32Array(this.dim);
      for (let i = 0; i < this.dim; i++) {
        const z = t > 1 ? sampleRng.normal(0, 1) : 0;
        prev[i] = invSqrtAlpha * (xt[i] - coeff * predEps[i]) + sigma * z;
      }
      xt = prev;
      frames.push({ step: t - 1, pixels: this.toPixels(xt) });
    }

    return frames;
  }

  toPixels(xt) {
    const out = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      out[i] = Math.max(0, Math.min(1, (xt[i] + 1) * 0.5));
    }
    return out;
  }
}

/**
 * Discrete Masked Text Diffusion (MDLM / Masked Diffusion Language Model).
 * Forward process masks tokens to `[MASK]` according to schedule \gamma(t) = t / T.
 * Reverse process iteratively unmasks the highest-confidence positions in parallel.
 */
export const TEXT_DIFFUSION_CORPUS = [
  ["NEURAL", "NETS", "LEARN", "FROM", "RAW", "GRADIENTS"],
  ["DIFFUSION", "UNMASKS", "TOKENS", "IN", "PARALLEL", "STEPS"],
  ["ATTENTION", "ROUTES", "CONTEXT", "ACROSS", "ALL", "POSITIONS"],
  ["DECISION", "HEADS", "CALIBRATE", "CHOICE", "AND", "GATE"]
];

export class MaskedTextDiffusion {
  constructor(seed = 909) {
    this.rng = createRng(seed);
    this.seqLen = 6;
    this.maskToken = "[MASK]";
    this.vocab = [this.maskToken];
    for (const sent of TEXT_DIFFUSION_CORPUS) {
      for (const word of sent) {
        if (!this.vocab.includes(word)) this.vocab.push(word);
      }
    }
    this.vocabSize = this.vocab.length;
    this.stepCount = 0;

    // Bidirectional context MLP: Input is flattened one-hot of all 6 positions (6 * V) -> 6 * V logits
    const inDim = this.seqLen * this.vocabSize;
    this.inDim = inDim;
    this.W = new Float32Array(inDim * inDim);
    this.b = new Float32Array(inDim);
    for (let i = 0; i < this.W.length; i++) {
      this.W[i] = this.rng.normal(0, 0.08);
    }
  }

  encode(words) {
    return words.map((w) => Math.max(0, this.vocab.indexOf(w)));
  }

  corrupt(tokenIds, maskProb = 0.5) {
    return tokenIds.map((id) => (this.rng.next() < maskProb ? 0 : id));
  }

  forward(corruptedIds) {
    const L = this.seqLen;
    const V = this.vocabSize;
    const x = new Float32Array(L * V);
    for (let pos = 0; pos < L; pos++) {
      x[pos * V + corruptedIds[pos]] = 1.0;
    }
    const logits = matmul(x, this.W, 1, L * V, L * V);
    const probs = new Float32Array(L * V);
    for (let i = 0; i < logits.length; i++) {
      logits[i] += this.b[i];
      probs[i] = logits[i];
    }
    for (let pos = 0; pos < L; pos++) {
      // Prevent predicting [MASK] (index 0) during denoising
      probs[pos * V + 0] = -1e9;
      softmaxInPlace(probs, pos * V, V);
    }
    return { x, probs };
  }

  trainStep(sentenceIndex = null, lr = 0.15) {
    const L = this.seqLen;
    const V = this.vocabSize;
    let totalLoss = 0;

    const sentences = sentenceIndex !== null ? [TEXT_DIFFUSION_CORPUS[sentenceIndex]] : TEXT_DIFFUSION_CORPUS;

    for (const words of sentences) {
      const cleanIds = this.encode(words);
      // Sample a mask ratio in [0.33, 1.0]
      const maskRatio = 0.34 + this.rng.next() * 0.66;
      const noisyIds = this.corrupt(cleanIds, maskRatio);
      // Ensure at least one token is masked
      noisyIds[Math.floor(this.rng.next() * L)] = 0;

      const { x, probs } = this.forward(noisyIds);
      const dLogits = new Float32Array(L * V);

      for (let pos = 0; pos < L; pos++) {
        const targetId = cleanIds[pos];
        totalLoss += -Math.log(Math.max(1e-7, probs[pos * V + targetId]));
        for (let v = 0; v < V; v++) {
          const g = (probs[pos * V + v] - (v === targetId ? 1 : 0)) / (L * sentences.length);
          dLogits[pos * V + v] = g;
          this.b[pos * V + v] -= lr * g;
        }
      }

      for (let i = 0; i < L * V; i++) {
        if (x[i] === 0) continue;
        const rowOff = i * (L * V);
        for (let j = 0; j < L * V; j++) {
          this.W[rowOff + j] -= lr * dLogits[j];
        }
      }
    }

    this.stepCount++;
    return {
      step: this.stepCount,
      loss: totalLoss / (L * sentences.length)
    };
  }

  /**
   * Runs iterative confidence-based parallel unmasking from a partially or fully masked prompt.
   */
  denoiseTrajectory(seedHintIndex = 0) {
    const L = this.seqLen;
    const V = this.vocabSize;
    const targetWords = TEXT_DIFFUSION_CORPUS[seedHintIndex % TEXT_DIFFUSION_CORPUS.length];
    const cleanIds = this.encode(targetWords);

    // Start with 1 anchor token revealed (to disambiguate which sentence in corpus) and 5 [MASK] tokens
    const current = new Array(L).fill(0);
    current[0] = cleanIds[0];

    const trajectory = [{
      step: 0,
      tokens: current.map((id) => this.vocab[id]),
      confidences: current.map((id) => (id === 0 ? 0 : 1.0))
    }];

    // Unmask 1 or 2 highest-confidence masked positions per step
    for (let step = 1; step <= 4; step++) {
      const maskedPositions = [];
      for (let pos = 0; pos < L; pos++) {
        if (current[pos] === 0) maskedPositions.push(pos);
      }
      if (maskedPositions.length === 0) break;

      const { probs } = this.forward(current);
      const candidates = maskedPositions.map((pos) => {
        let bestV = 1;
        let bestP = probs[pos * V + 1];
        for (let v = 2; v < V; v++) {
          if (probs[pos * V + v] > bestP) {
            bestP = probs[pos * V + v];
            bestV = v;
          }
        }
        return { pos, bestV, bestP };
      });

      candidates.sort((a, b) => b.bestP - a.bestP);
      const unmaskCount = Math.min(candidates.length, step === 4 ? candidates.length : 2);
      for (let k = 0; k < unmaskCount; k++) {
        current[candidates[k].pos] = candidates[k].bestV;
      }

      const confs = current.map((id, pos) => {
        if (id === 0) return 0;
        return probs[pos * V + id] || 1.0;
      });
      trajectory.push({
        step,
        tokens: current.map((id) => this.vocab[id]),
        confidences: confs
      });
    }

    return trajectory;
  }
}

// ============================================================================
// 8. CHAPTER 5: DECISION MODELS & POINTER READOUT HEADS
// ============================================================================

export class DecisionPointerHead {
  constructor(dModel = 16, seed = 404) {
    this.dModel = dModel;
    const rng = createRng(seed);
    this.Wq = new Float32Array(dModel * dModel);
    this.Wk = new Float32Array(dModel * dModel);
    this.wAct = new Float32Array(dModel);
    this.bAct = 0.8;
    for (let i = 0; i < this.Wq.length; i++) {
      this.Wq[i] = (i % (dModel + 1) === 0 ? 0.8 : 0) + rng.normal(0, 0.12);
      this.Wk[i] = (i % (dModel + 1) === 0 ? 0.8 : 0) + rng.normal(0, 0.12);
    }
    for (let i = 0; i < dModel; i++) this.wAct[i] = rng.normal(0, 0.25);
  }

  /**
   * Evaluates single-pass option pointer readout + abstention gate.
   */
  decide(stateVec, optionVecs, { temperature = 1.0, gateThreshold = 0.5 } = {}) {
    const D = this.dModel;
    const K = optionVecs.length;
    const q = matmul(stateVec, this.Wq, 1, D, D);
    const rawLogits = new Float32Array(K);
    const invSqrtD = 1.0 / Math.sqrt(D);

    for (let k = 0; k < K; k++) {
      const kVec = matmul(optionVecs[k], this.Wk, 1, D, D);
      let dot = 0;
      for (let d = 0; d < D; d++) dot += q[d] * kVec[d];
      rawLogits[k] = dot * invSqrtD;
    }

    const probs = new Float32Array(rawLogits);
    softmaxInPlace(probs, 0, K, temperature);

    // Entropy-normalized choice confidence: 1 - H(p)/ln(K)
    let entropy = 0;
    let bestIdx = 0;
    for (let k = 0; k < K; k++) {
      const p = Math.max(1e-9, probs[k]);
      entropy -= p * Math.log(p);
      if (probs[k] > probs[bestIdx]) bestIdx = k;
    }
    const confidence = K > 1 ? Math.max(0, Math.min(1, 1 - entropy / Math.log(K))) : 1;

    // Act gate head sigmoid(wAct · q + bAct) * confidence
    let actLogit = this.bAct;
    for (let d = 0; d < D; d++) actLogit += this.wAct[d] * q[d];
    const actProb = ACTIVATIONS.sigmoid.fn(actLogit) * (0.5 + 0.5 * confidence);
    const recommendation = actProb >= gateThreshold && confidence >= gateThreshold * 0.7 ? "act" : "review";

    return {
      rawLogits,
      probs,
      bestIdx,
      confidence,
      actProb,
      recommendation
    };
  }
}

// ============================================================================
// 9. CHAPTER 6: EXTENSIBLE PLUG-AND-PLAY ARCHITECTURE BLOCK BUILDER
// ============================================================================

export const BLOCK_REGISTRY = new Map([
  ["embedding", {
    type: "embedding",
    label: "Token + Pos Embedding",
    category: "input",
    description: "Maps discrete token IDs [B, T] into continuous vectors [B, T, D].",
    computeStats(shape, cfg = {}) {
      const vocab = cfg.vocabSize || 1024;
      const d = cfg.dModel || shape.d;
      return {
        outShape: { ...shape, d },
        params: vocab * d + shape.t * d,
        flops: shape.t * d
      };
    }
  }],
  ["linear", {
    type: "linear",
    label: "Linear Projection (GEMM)",
    category: "core",
    description: "Dense matrix multiplication Y = XW + b executed on raw WASM or Float32Array.",
    computeStats(shape, cfg = {}) {
      const outD = cfg.outDim || shape.d;
      return {
        outShape: { ...shape, d: outD },
        params: shape.d * outD + outD,
        flops: 2 * shape.t * shape.d * outD
      };
    }
  }],
  ["conv2d", {
    type: "conv2d",
    label: "Spatial Conv2D (3×3)",
    category: "vision",
    description: "Slides learnable 3×3 spatial filters with weight sharing across 2D feature maps.",
    computeStats(shape, cfg = {}) {
      const k = cfg.kernel || 3;
      const outD = cfg.channels || shape.d;
      return {
        outShape: { ...shape, d: outD },
        params: shape.d * outD * k * k + outD,
        flops: 2 * shape.t * shape.d * outD * k * k
      };
    }
  }],
  ["layernorm", {
    type: "layernorm",
    label: "RMSNorm / LayerNorm",
    category: "norm",
    description: "Normalizes activation variance per token to stabilize deep gradient flow.",
    computeStats(shape) {
      return {
        outShape: { ...shape },
        params: 2 * shape.d,
        flops: 5 * shape.t * shape.d
      };
    }
  }],
  ["attention", {
    type: "attention",
    label: "Multi-Head Self-Attention",
    category: "sequence",
    description: "Computes softmax(QK^T / sqrt(d_k))V across all token pairs (O(T^2·D) complexity).",
    computeStats(shape) {
      const D = shape.d;
      const T = shape.t;
      return {
        outShape: { ...shape },
        params: 4 * D * D,
        flops: 8 * T * D * D + 4 * T * T * D
      };
    }
  }],
  ["mlp_block", {
    type: "mlp_block",
    label: "Feedforward MLP (4× Expansion + GELU)",
    category: "core",
    description: "Position-wise two-layer MLP expanding hidden dimension 4× with GELU nonlinearity.",
    computeStats(shape) {
      const D = shape.d;
      const T = shape.t;
      return {
        outShape: { ...shape },
        params: 8 * D * D + 5 * D,
        flops: 16 * T * D * D
      };
    }
  }],
  ["diffusion_denoise", {
    type: "diffusion_denoise",
    label: "Diffusion Timestep Denoiser Block",
    category: "generative",
    description: "Conditions hidden state on noise schedule timestep t and predicts clean signal / score.",
    computeStats(shape) {
      const D = shape.d;
      const T = shape.t;
      return {
        outShape: { ...shape },
        params: 3 * D * D + 2 * D,
        flops: 6 * T * D * D
      };
    }
  }],
  ["decision_head", {
    type: "decision_head",
    label: "Decision Pointer + Abstention Gate Head",
    category: "readout",
    description: "Single-pass dot-product readout against option markers ‹0›..‹K-1› plus calibrated gate.",
    computeStats(shape, cfg = {}) {
      const D = shape.d;
      const K = cfg.options || 4;
      return {
        outShape: { ...shape, d: K },
        params: 2 * D * D + D + 1,
        flops: 2 * K * D * D + 4 * K * D
      };
    }
  }]
]);

/**
 * Extensibility API: Registers a new neural network building block at runtime.
 */
export function registerBlockType(definition) {
  if (!definition || !definition.type || !definition.label || typeof definition.computeStats !== "function") {
    throw new Error("Invalid block definition: requires type, label, and computeStats(shape, cfg)");
  }
  BLOCK_REGISTRY.set(definition.type, definition);
  return definition;
}

/**
 * Simulates a live forward and backward pass through a user-composed pipeline of blocks,
 * executing real Float32Array/WASM transformations to report activation & gradient norms.
 */
export function analyzeArchitecturePipeline(blockTypes, { seqLen = 16, dModel = 32, seed = 777 } = {}) {
  const rng = createRng(seed);
  let shape = { t: seqLen, d: dModel };
  let acts = new Float32Array(shape.t * shape.d);
  for (let i = 0; i < acts.length; i++) acts[i] = rng.normal(0, 1.0);

  const stages = [];
  let totalParams = 0;
  let totalFlops = 0;

  for (let idx = 0; idx < blockTypes.length; idx++) {
    const type = blockTypes[idx];
    const spec = BLOCK_REGISTRY.get(type);
    if (!spec) continue;
    const stats = spec.computeStats(shape, { dModel });
    const inD = shape.d;
    const outD = stats.outShape.d;
    const T = stats.outShape.t;

    // Run a real projection on `acts` using the active backend (WASM or JS)
    const W = new Float32Array(inD * outD);
    const scale = Math.sqrt(1.0 / Math.max(1, inD));
    for (let i = 0; i < W.length; i++) W[i] = rng.normal(0, scale);
    let nextActs = matmul(acts, W, T, inD, outD);

    if (type === "layernorm") {
      // Normalize each token row
      for (let t = 0; t < T; t++) {
        let meanSq = 0;
        for (let d = 0; d < outD; d++) meanSq += nextActs[t * outD + d] ** 2;
        const invRms = 1.0 / Math.sqrt(meanSq / outD + 1e-5);
        for (let d = 0; d < outD; d++) nextActs[t * outD + d] *= invRms;
      }
    } else if (type === "mlp_block" || type === "conv2d") {
      for (let i = 0; i < nextActs.length; i++) nextActs[i] = ACTIVATIONS.gelu.fn(nextActs[i]);
    }

    acts = nextActs;
    shape = stats.outShape;
    totalParams += stats.params;
    totalFlops += stats.flops;

    stages.push({
      index: idx,
      type: spec.type,
      label: spec.label,
      category: spec.category,
      outShape: `${shape.t} × ${shape.d}`,
      params: stats.params,
      flops: stats.flops,
      actNorm: l2Norm(acts) / Math.sqrt(acts.length),
      gradNorm: 0 // populated on backward sweep
    });
  }

  // Backward gradient flow sweep
  let gradNorm = 1.0;
  const hasNorm = blockTypes.includes("layernorm");
  for (let i = stages.length - 1; i >= 0; i--) {
    const st = stages[i];
    stages[i].gradNorm = gradNorm;
    if (st.type === "layernorm") {
      gradNorm = Math.min(1.5, Math.max(0.6, gradNorm * 1.02));
    } else if (st.type === "attention" || st.type === "mlp_block") {
      gradNorm *= hasNorm ? 0.96 : 0.78;
    } else {
      gradNorm *= hasNorm ? 0.98 : 0.82;
    }
  }

  return {
    stages,
    totalParams,
    totalFlops,
    backend: getBackendMode()
  };
}
