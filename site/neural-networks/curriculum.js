// @ts-check
/**
 * Deep-dive hyperparameter encyclopedia and side-by-side Raw JS / Raw WASM / WebGPU WGSL
 * implementation blueprints for the Neural Networks From First Principles workbench.
 */

export const HYPERPARAMETER_GUIDE = {
  datasets: {
    xor: {
      title: "XOR Parity (Minsky & Papert's 1969 Benchmark)",
      equation: "y = 1 iff sign(x₀) ≠ sign(x₁), with Gaussian jitter ε ~ N(0, 0.18²)",
      topology: "4 quadrants in ℝ²; positive class occupies quadrants II and IV, negative class occupies I and III.",
      theory:
        "A single linear threshold f(x) = w₀x₀ + w₁x₁ + b > 0 defines a single half-plane. Because no straight line can place opposite corners (-0.55, +0.55) and (+0.55, -0.55) on one side while excluding (-0.55, -0.55) and (+0.55, +0.55), a 1-layer [2, 1] Perceptron is mathematically capped at ~50–75% accuracy. At least 2 hidden neurons are required to fold the plane along two parallel lines and intersect them."
    },
    circle: {
      title: "Concentric Ring vs Core (Radial Manifold)",
      equation: "Inner core r ∈ [0, 0.38] → y = 1; Outer annulus r ∈ [0.58, 0.86] → y = 0",
      topology: "Closed Jordan curve boundary S¹ separating a simply connected disk from an outer annulus (Betti number β₁ = 1).",
      theory:
        "Separating a closed disk from a surrounding ring requires enclosing the origin inside a bounded polytope. In ℝ², a minimum of 3 hidden ReLU/GELU hyperplanes is required to form a closed triangle around the inner core; 4–8 hidden units approximate the smooth circular level set x₀² + x₁² = R²."
    },
    spiral: {
      title: "Two-Arm Archimedean Spiral (High-Curvature Interleaved Manifold)",
      equation: "r(t) = 0.12 + 0.78t, θ(t) = 2.6πt + π·y, t ∈ [0, 1]",
      topology: "Two 1D manifolds winding 1.3 full turns around each other; any ray from the origin alternates classes multiple times.",
      theory:
        "Because every radial line crosses alternating class boundaries, shallow networks with few hidden units struggle to allocate enough linear folds. Deeper architectures (2→8→8→1 or 2→6→6→6→1) compose piecewise-linear warps exponentially (Telgarsky's depth-separation theorem), uncoiling the spiral across successive layers."
    },
    moons: {
      title: "Interleaving Half-Moons (Non-Convex Semicircular Arcs)",
      equation: "Upper arc (cos θ, sin θ) → y = 0; Lower shifted arc (1 − cos θ, −sin θ + 0.25) → y = 1",
      topology: "Two non-convex interlocking crescents separated by a smooth S-shaped cubic decision boundary.",
      theory:
        "Tests whether the activation function can bend a linear boundary into an inflection curve (S-curve) without overfitting to local coordinate noise."
    }
  },

  architectures: {
    "2,4,4,1": {
      title: "2 → 4 → 4 → 1 (Compact 2-Hidden-Layer MLP · 37 Parameters)",
      params: "L1: 2×4+4 = 12 | L2: 4×4+4 = 20 | L3: 4×1+1 = 5 → Total = 37 float32 parameters",
      theory:
        "Layer 1 projects 2D coordinates into 4 half-space detectors; Layer 2 combines those 4 detectors into convex/non-convex regions; Layer 3 weights them into a final log-odds logit. Ideal for inspecting every single neuron and synapse in the interactive graph."
    },
    "2,8,8,1": {
      title: "2 → 8 → 8 → 1 (Standard 2-Hidden-Layer MLP · 105 Parameters)",
      params: "L1: 2×8+8 = 24 | L2: 8×8+8 = 72 | L3: 8×1+1 = 9 → Total = 105 float32 parameters",
      theory:
        "Overparameterized relative to XOR and Moons, which smooths the loss landscape and avoids bad local minima (the 'lottery ticket' effect: with 8 random initial directions in ℝ², gradient descent rapidly finds a subset of aligned hyperplanes)."
    },
    "2,12,12,1": {
      title: "2 → 12 → 12 → 1 (Wide 2-Hidden-Layer MLP · 205 Parameters)",
      params: "L1: 2×12+12 = 36 | L2: 12×12+12 = 156 | L3: 12×1+1 = 13 → Total = 205 float32 parameters",
      theory:
        "High angular resolution in Layer 1 (12 initial half-plane cuts), enabling smooth approximation of radial (Circle) and high-frequency (Spiral) boundaries in fewer epochs."
    },
    "2,6,6,6,1": {
      title: "2 → 6 → 6 → 6 → 1 (Deep 3-Hidden-Layer MLP · 109 Parameters)",
      params: "L1: 18 | L2: 42 | L3: 42 | L4: 7 → Total = 109 float32 parameters",
      theory:
        "Demonstrates depth vs gradient attenuation: each additional layer multiplies backward Jacobians δ⁽ˡ⁾ = δ⁽ˡ⁺¹⁾(W⁽ˡ⁺¹⁾)ᵀ ⊙ σ'(z⁽ˡ⁾). Try switching Activation to Sigmoid on this 3-hidden-layer stack to watch Layer 1's gradient norm vanish!"
    },
    "2,1": {
      title: "2 → 1 (Single-Layer Rosenblatt Perceptron · 3 Parameters)",
      params: "L1: 2×1+1 = 3 float32 parameters (w₀, w₁, b)",
      theory:
        "Computes ŷ = σ(w₀x₀ + w₁x₁ + b). Because the level set w₀x₀ + w₁x₁ + b = 0 is a single straight line in ℝ², it can only solve linearly separable tasks and is provably incapable of separating XOR, Circle, or Spiral."
    }
  },

  activations: {
    gelu: {
      title: "GELU — Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016)",
      equation: "GELU(z) = z · Φ(z) ≈ z · σ(1.702z),   GELU'(z) ≈ σ(1.702z) + 1.702z·σ(1.702z)(1 − σ(1.702z))",
      theory:
        "Standard activation in GPT, BERT, ViT, and modern Diffusion U-Nets/DiTs. Unlike ReLU's hard kink at z = 0, GELU weights inputs by their percentile in a standard Gaussian distribution—providing a smooth non-zero derivative for slightly negative pre-activations (z ∈ [−2, 0]) that prevents dead neurons while retaining linear non-saturating gradients for z > 0."
    },
    relu: {
      title: "ReLU — Rectified Linear Unit (Nair & Hinton, 2010)",
      equation: "ReLU(z) = max(0, z),   ReLU'(z) = 1 if z > 0 else 0",
      theory:
        "Sparked the 2012 AlexNet deep learning revolution by replacing saturating Sigmoids with an identity derivative (σ'(z) = 1) for positive inputs. Caveat: if a large learning rate pushes a neuron's bias negative so z < 0 for all training points, ReLU'(z) = 0 everywhere and the neuron permanently dies ('Dying ReLU')."
    },
    tanh: {
      title: "Hyperbolic Tangent tanh(z) (Zero-Centered Classic)",
      equation: "tanh(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ),   tanh'(z) = 1 − tanh²(z) ∈ (0, 1]",
      theory:
        "Zero-centered output in (−1, +1) with peak derivative tanh'(0) = 1.0 (four times stronger than Sigmoid). Still saturates exponentially when |z| > 2.5, which is why Elman RNNs and deep tanh MLPs require careful Xavier initialization or normalization."
    },
    sigmoid: {
      title: "Logistic Sigmoid σ(z) (Historical 1986 Default — Vanishing Gradient Culprit)",
      equation: "σ(z) = 1 / (1 + e⁻ᶻ) ∈ (0, 1),   σ'(z) = σ(z)(1 − σ(z)) ∈ (0, 0.25]",
      theory:
        "Notice two severe pathologies when used in hidden layers: (1) its maximum derivative at z = 0 is only 0.25, so every hidden layer shrinks backpropagated gradients by at least 4× (0.25³ = 1.5% after just 3 layers!); (2) its outputs are strictly positive (a > 0), forcing all weight gradients ∂L/∂Wᵢⱼ = aᵢδⱼ for a neuron to share the same sign ('zig-zagging' optimization). Reserve Sigmoid for binary output readouts and gates."
    }
  },

  optimizers: {
    adamw: {
      title: "AdamW — Adaptive Moment Estimation with Decoupled Weight Decay (Loshchilov & Hutter, 2019)",
      equation: "mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,  vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²,  θₜ = (1 − ηλ)θₜ₋₁ − η · m̂ₜ / (√v̂ₜ + ε)",
      theory:
        "Maintains per-parameter exponential moving averages of the gradient mean (mₜ, β₁=0.9) and uncentered variance (vₜ, β₂=0.999). Dividing by √v̂ₜ normalizes step sizes across layers whose gradient norms differ by orders of magnitude. Crucially, AdamW decouples weight decay (1 − ηλ)θ from the adaptive denominator √v̂ₜ so large-gradient weights still receive proportional regularization."
    },
    momentum: {
      title: "SGD with Polyak Heavy-Ball Momentum (Polyak, 1964)",
      equation: "mₜ = β·mₜ₋₁ + gₜ (β = 0.9),   θₜ = θₜ₋₁ − η(mₜ + λθₜ₋₁)",
      theory:
        "Models the parameter vector as a heavy ball rolling down the loss surface. Directions with consistent gradient sign accumulate velocity up to 1/(1−β) = 10×, while high-curvature ravine directions where gradients flip sign (+g, −g) cancel out."
    },
    sgd: {
      title: "Vanilla Stochastic Gradient Descent (Robbins & Monro, 1951)",
      equation: "θₜ = θₜ₋₁ − η(∇_θ L(θₜ₋₁) + λθₜ₋₁)",
      theory:
        "Steps directly opposite the instantaneous gradient with uniform scalar learning rate η across all layers. If Layer 1 has small gradients (‖∇W⁽¹⁾‖ = 0.01) and Layer 3 has large gradients (‖∇W⁽³⁾‖ = 0.50), Vanilla SGD either diverges in Layer 3 or stalls in Layer 1."
    }
  },

  learningRate(lr) {
    if (lr < 0.015) {
      const regime = "Conservative / Small Step (η < 0.015)";
      const detail = `At η = ${lr.toFixed(3)}, trajectory closely tracks continuous gradient flow dθ/dt = −∇L(θ), but requires many epochs to escape flat saddle points where ‖∇L‖ ≈ 0.`;
      return {
        regime,
        detail,
        title: regime,
        equation: `Δθ = −${lr.toFixed(3)} · gₜ  (Stability condition: η < 2 / λ_max(∇²L))`,
        theory: detail
      };
    }
    if (lr <= 0.09) {
      const regime = "Well-Conditioned Sweet Spot (0.015 ≤ η ≤ 0.090)";
      const detail = `At η = ${lr.toFixed(3)}, updates stay safely below the local Hessian stability limit 2/λ_max(∇²L), giving fast convergence without weight oscillation.`;
      return {
        regime,
        detail,
        title: regime,
        equation: `Δθ = −${lr.toFixed(3)} · gₜ  (0.015 ≤ η ≤ 0.090 < 2 / λ_max(∇²L))`,
        theory: detail
      };
    }
    const regime = "Aggressive / Edge of Stability (η > 0.090)";
    const detail = `At η = ${lr.toFixed(3)}, steps approach or exceed 2/λ_max(∇²L) along sharp curvature directions. Watch the decision boundary bounce or catapult between basins!`;
    return {
      regime,
      detail,
      title: regime,
      equation: `Δθ = −${lr.toFixed(3)} · gₜ  (Approaching Hessian stability threshold 2 / λ_max(∇²L))`,
      theory: detail
    };
  }
};

export const KERNEL_CODE_BLUEPRINTS = {
  gemm: {
    title: "1. Dense Matrix Multiplication (GEMM: C[M×N] = A[M×K] · B[K×N])",
    summary:
      "Over 90% of FLOPs in MLPs, CNNs (via im2col), Transformers, and Diffusion models reduce to General Matrix Multiplication (GEMM). Compare the three execution targets:",
    js: `// Raw JavaScript (Float32Array Cache-Friendly i-k-j Loop Order)
export function jsMatmul(A, B, M, K, N, out = new Float32Array(M * N)) {
  out.fill(0);
  for (let i = 0; i < M; i++) {
    const iK = i * K;
    const iN = i * N;
    // Hoisting A[i, k] in the middle loop streams row B[k, 0..N-1] sequentially in L1 cache
    for (let k = 0; k < K; k++) {
      const a = A[iK + k];
      const kN = k * N;
      for (let j = 0; j < N; j++) {
        out[iN + j] += a * B[kN + j];
      }
    }
  }
  return out;
}`,
    wasm: `;; Raw WebAssembly (wasm32 Linear Memory Bytecode — 183 Bytes Compiled In-Memory)
(module
  (memory (export "mem") 4) ;; 256 KiB linear Arena: [A | B | C]
  (func (export "gemm_f32")
    (param $aPtr i32) (param $bPtr i32) (param $cPtr i32)
    (param $M i32) (param $K i32) (param $N i32)
    (local $i i32) (local $j i32) (local $k i32) (local $sum f32)
    (block $break_i (loop $loop_i
      (br_if $break_i (i32.ge_u (local.get $i) (local.get $M)))
      (local.set $j (i32.const 0))
      (block $break_j (loop $loop_j
        (br_if $break_j (i32.ge_u (local.get $j) (local.get $N)))
        (local.set $sum (f32.const 0))
        (local.set $k (i32.const 0))
        (block $break_k (loop $loop_k
          (br_if $break_k (i32.ge_u (local.get $k) (local.get $K)))
          ;; sum += A[i*K + k] * B[k*N + j]  (opcodes: 0x2a f32.load, 0x94 f32.mul, 0x92 f32.add)
          (local.set $sum (f32.add (local.get $sum) (f32.mul
            (f32.load (i32.add (local.get $aPtr)
              (i32.shl (i32.add (i32.mul (local.get $i) (local.get $K)) (local.get $k)) (i32.const 2))))
            (f32.load (i32.add (local.get $bPtr)
              (i32.shl (i32.add (i32.mul (local.get $k) (local.get $N)) (local.get $j)) (i32.const 2)))))))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br $loop_k)))
        ;; C[i*N + j] = sum  (opcode: 0x38 f32.store)
        (f32.store (i32.add (local.get $cPtr)
          (i32.shl (i32.add (i32.mul (local.get $i) (local.get $N)) (local.get $j)) (i32.const 2)))
          (local.get $sum))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $loop_j)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loop_i)))))`,
    webgpu: `// WebGPU WGSL Compute Shader — 8×8 Shared Workgroup Tile Memory
struct Dimensions { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> dims : Dimensions;
@group(0) @binding(1) var<storage, read> A : array<f32>;
@group(0) @binding(2) var<storage, read> B : array<f32>;
@group(0) @binding(3) var<storage, read_write> C : array<f32>;

var<workgroup> tileA : array<array<f32, 8>, 8>;
var<workgroup> tileB : array<array<f32, 8>, 8>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = gid.y; let col = gid.x;
  var acc : f32 = 0.0;
  let numTiles = (dims.K + 7u) / 8u;
  for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * 8u + lid.x;
    let bRow = t * 8u + lid.y;
    tileA[lid.y][lid.x] = select(0.0, A[row * dims.K + aCol], row < dims.M && aCol < dims.K);
    tileB[lid.y][lid.x] = select(0.0, B[bRow * dims.N + col], bRow < dims.K && col < dims.N);
    workgroupBarrier();
    for (var k : u32 = 0u; k < 8u; k = k + 1u) {
      acc = acc + tileA[lid.y][k] * tileB[k][lid.x];
    }
    workgroupBarrier();
  }
  if (row < dims.M && col < dims.N) { C[row * dims.N + col] = acc; }
}`
  },

  mlp_backprop: {
    title: "2. Feedforward MLP & Analytical Reverse-Mode Backpropagation",
    summary:
      "Forward pass caches pre-activations z⁽ˡ⁾ and post-activations a⁽ˡ⁾; backward pass propagates error signals δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ from output to input via the chain rule.",
    js: `// Forward & Backward Pass for Layer l (inDim -> outDim)
// 1. Forward: Z = A_prev * W + b,  A = act.fn(Z)
const Z = matmul(prevA, layer.W, N, inDim, outDim);
for (let n = 0; n < N; n++) {
  for (let j = 0; j < outDim; j++) {
    Z[n * outDim + j] += layer.b[j];
    A[n * outDim + j] = act.fn(Z[n * outDim + j]);
  }
}

// 2. Backward: dW = (A_prev)^T * dZ,  db = sum(dZ),  dZ_prev = (dZ * W^T) ⊙ act.grad(Z_prev)
for (let n = 0; n < N; n++) {
  for (let j = 0; j < outDim; j++) {
    const dz = dZ[n * outDim + j];
    layer.db[j] += dz;
    for (let i = 0; i < inDim; i++) {
      layer.dW[i * outDim + j] += prevA[n * inDim + i] * dz;
      dPrevZ[n * inDim + i] += dz * layer.W[i * outDim + j];
    }
  }
}
for (let k = 0; k < N * inDim; k++) dPrevZ[k] *= act.grad(prevZ[k]);`,
    wasm: `;; WASM Linear + Bias + GELU Fused Row Loop Concept
;; Compute z = dot(a_prev, w_col) + bias, then store both z (for backprop) and a = σ(z)
(local.set $z (f32.add (call $gemm_dot (local.get $aPrevPtr) (local.get $wColPtr) (local.get $inDim))
                       (f32.load (local.get $bPtr))))
(f32.store (local.get $zCachePtr) (local.get $z))
;; Backprop elementwise chain rule: dPrevZ[i] = sum_j(dZ[j] * W[i,j]) * act_grad(prevZ[i])
(f32.store (local.get $dPrevZPtr)
  (f32.mul (call $gemm_transposed_dot (local.get $dZPtr) (local.get $wRowPtr) (local.get $outDim))
           (call $act_grad (f32.load (local.get $prevZPtr)))))`,
    webgpu: `// WebGPU WGSL Backward Gradient Kernel: dW[i, j] = sum_n(A_prev[n, i] * dZ[n, j])
@compute @workgroup_size(8, 8, 1)
fn backprop_dW(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y; // inDim index
  let j = gid.x; // outDim index
  if (i >= dims.InDim || j >= dims.OutDim) { return; }
  var grad : f32 = 0.0;
  for (var n : u32 = 0u; n < dims.BatchN; n = n + 1u) {
    grad = grad + A_prev[n * dims.InDim + i] * dZ[n * dims.OutDim + j];
  }
  dW[i * dims.OutDim + j] = grad;
}`
  },

  conv2d: {
    title: "3. 2D Spatial Convolution (3×3 Kernels) & MaxPool2D Switch Routing",
    summary:
      "Convolutions share a 3×3 filter K[f, u, v] across every spatial (r, c) patch; MaxPool2D records the winning argmax index during forward so gradients route solely to the maximum pixel.",
    js: `// Forward Conv2D (8x8 -> 6x6) + ReLU + 2x2 MaxPool (6x6 -> 3x3)
for (let f = 0; f < 4; f++) {
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      let sum = convB[f];
      for (let u = 0; u < 3; u++)
        for (let v = 0; v < 3; v++)
          sum += img[(r + u) * 8 + (c + v)] * convW[f * 9 + u * 3 + v];
      convA[f * 36 + r * 6 + c] = sum > 0 ? sum : 0;
    }
  }
}
// Backward through MaxPool2D & Conv2D filter gradient accumulation
for (let p = 0; p < 36; p++) {
  const winIdx = poolArgmax[p]; // only the winning pixel in each 2x2 window gets gradient!
  if (convZ[winIdx] > 0) dConvZ[winIdx] = dPool[p];
}`,
    wasm: `;; WASM 3x3 Spatial Stencil Inner Loop (Unrolled 9-Tap Multiply-Accumulate)
;; Base address: imgPtr + ((r + u) * 8 + (c + v)) * 4
(local.set $sum (f32.load (local.get $biasPtr)))
;; Tap (0,0)
(local.set $sum (f32.add (local.get $sum)
  (f32.mul (f32.load (local.get $patchPtr)) (f32.load (local.get $kernelPtr)))))
;; ... Taps (0,1)..(2,2) use constant byte offsets (+4, +8, +32, +36, +40, +64, +68, +72)
;; ReLU clamp: f32.max(sum, 0.0)
(f32.store (local.get $outPtr) (f32.max (local.get $sum) (f32.const 0.0)))`,
    webgpu: `// WebGPU WGSL 2D Convolution Dispatch: 1 Thread per (outRow, outCol, filterIdx)
@compute @workgroup_size(6, 6, 1)
fn conv2d_3x3(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y; let f = gid.z;
  if (r >= 6u || c >= 6u || f >= 4u) { return; }
  var sum : f32 = convB[f];
  for (var u : u32 = 0u; u < 3u; u = u + 1u) {
    for (var v : u32 = 0u; v < 3u; v = v + 1u) {
      sum = sum + img[(r + u) * 8u + (c + v)] * convW[f * 9u + u * 3u + v];
    }
  }
  convA[f * 36u + r * 6u + c] = max(0.0, sum);
}`
  },

  rnn_bptt: {
    title: "4. Recurrent Cells (Elman RNN vs Gated GRU) & Backpropagation Through Time (BPTT)",
    summary:
      "Vanilla RNNs multiply gradients by W_hh^T at every step (vanishing/exploding exponentially with T). GRU/LSTM adds an update gate z_t creating an additive Constant Error Carousel h_t = (1−z_t)⊙h_{t-1} + z_t⊙h̃_t.",
    js: `// Forward Recurrence (t = 0..T-1) & Unrolled BPTT (t = T-1..0)
// Forward step t:
const cand = Math.tanh(x_t * Wxh[j] + hWhh[j] + bh[j]);
const z = sigmoid(x_t * Wxz[j] + hWhz[j] + bz[j]); // update gate ∈ (0, 1)
hNext[j] = isGru ? (1 - z) * hPrev[j] + z * cand : cand;

// Backward step t (BPTT):
const dCand = isGru ? dh[j] * z : dh[j];
const da = dCand * (1 - cand * cand); // tanh'(a)
if (isGru) {
  // Direct additive identity highway across time when z is small!
  dhPrev[j] += dh[j] * (1 - z);
}
for (let i = 0; i < D; i++) {
  dWhh[i * D + j] += hPrev[i] * da;
  dhPrev[i] += da * Whh[i * D + j];
}`,
    wasm: `;; WASM Recurrent Time-Step Loop: Calls gemm_f32 for h_{t-1} * W_hh at each step t
(local.set $t (i32.const 0))
(loop $time_loop
  ;; Project h_{t-1} [1 x D] * W_hh [D x D] -> hWhh [1 x D]
  (call $gemm_f32 (local.get $hPrevPtr) (local.get $WhhPtr) (local.get $tmpPtr)
                  (i32.const 1) (local.get $D) (local.get $D))
  ;; Apply gated state update h_t = (1 - z_t) * h_{t-1} + z_t * tanh(a_t)
  (local.set $t (i32.add (local.get $t) (i32.const 1)))
  (br_if $time_loop (i32.lt_u (local.get $t) (local.get $T))))`,
    webgpu: `// WebGPU WGSL Fused GRU Gate + State Update Kernel (Dispatched per Timestep t)
@compute @workgroup_size(64)
fn gru_cell_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= dims.HiddenD) { return; }
  let z = 1.0 / (1.0 + exp(-(x_t * Wxz[j] + hWhz[j] + bz[j])));
  let cand = tanh(x_t * Wxh[j] + hWhh[j] + bh[j]);
  hNext[j] = (1.0 - z) * hPrev[j] + z * cand;
}`
  },

  deep_resnet: {
    title: "5. Very Deep Residual Highway: x^{(l)} = x^{(l-1)} + F_l(RMSNorm(x^{(l-1)}))",
    summary:
      "Pre-RMSNorm standardizes activation scale before each block, while the residual skip connection (+ x^{(l-1)}) adds an identity matrix I to every layer's backward Jacobian: ∂x_L/∂x_l = I + ∂/∂x_l ∑ F_k.",
    js: `// Forward Pre-RMSNorm + Residual Block l:
let sumSq = 0;
for (let d = 0; d < D; d++) sumSq += hIn[d] * hIn[d];
const invRms = 1.0 / Math.sqrt(sumSq / D + 1e-5);
for (let d = 0; d < D; d++) hNorm[d] = hIn[d] * invRms * gamma[d];

const z = matmul(hNorm, blk.W, 1, D, D);
for (let d = 0; d < D; d++) {
  const fOut = act.fn(z[d] + blk.b[d]);
  hNext[d] = useResidual ? hIn[d] + fOut : fOut; // Skip highway!
}

// Backward through Residual + RMSNorm:
for (let d = 0; d < D; d++) {
  const xHat = hIn[d] * invRms;
  const dhBranch = invRms * (dhNorm[d] * gamma[d] - xHat * meanDot);
  dhPrev[d] = useResidual ? dh[d] + dhBranch : dhBranch; // dh[d] passes straight through!
}`,
    wasm: `;; WASM RMSNorm Reduction + Residual Skip Addition
;; 1. Compute sumSq = sum(x[d] * x[d])
;; 2. invRms = f32.div(1.0, f32.sqrt(f32.add(f32.div(sumSq, D), 1e-5)))
(local.set $invRms (f32.div (f32.const 1.0)
  (f32.sqrt (f32.add (f32.div (local.get $sumSq) (local.get $D_f32)) (f32.const 0.00001)))))
;; 3. Residual highway: hNext[d] = f32.add(hIn[d], fOut[d])`,
    webgpu: `// WebGPU WGSL Workgroup Reduction for RMSNorm + Residual Add
@compute @workgroup_size(64)
fn rmsnorm_residual(@builtin(local_invocation_id) lid: vec3<u32>) {
  let d = lid.x;
  sharedSq[d] = hIn[d] * hIn[d];
  workgroupBarrier();
  // Parallel tree reduction in workgroup memory
  for (var stride : u32 = 32u; stride > 0u; stride = stride >> 1u) {
    if (d < stride) { sharedSq[d] = sharedSq[d] + sharedSq[d + stride]; }
    workgroupBarrier();
  }
  let invRms = inverseSqrt(sharedSq[0] / f32(dims.D) + 1e-5);
  hNorm[d] = hIn[d] * invRms * gamma[d];
}`
  },

  attention: {
    title: "6. Scaled Dot-Product Multi-Head Self-Attention & Softmax Jacobian",
    summary:
      "Projects X into Q, K, V per head, computes pairwise compatibility S = QK^T / √d_k, applies causal mask, row-wise Softmax P = softmax(S), and aggregates Context = P·V.",
    js: `// Forward Scaled Dot-Product Self-Attention (Single Head h)
const scale = 1.0 / Math.sqrt(dHead);
for (let i = 0; i < T; i++) {
  for (let j = 0; j < T; j++) {
    if (causal && j > i) { scores[i * T + j] = -1e9; continue; }
    let dot = 0;
    for (let d = 0; d < dHead; d++) dot += Q[i * dHead + d] * K[j * dHead + d];
    scores[i * T + j] = dot * scale;
  }
  softmaxInPlace(scores, i * T, T); // P[i, 0..T-1]
}
const context = matmul(scores, V, T, T, dHead);

// Backward Exact Softmax Jacobian: dS[i, j] = P[i, j] * (dP[i, j] - sum_k(P[i, k] * dP[i, k]))
let rowDot = 0;
for (let j = 0; j < T; j++) rowDot += P[i * T + j] * dP[i * T + j];
for (let j = 0; j < T; j++) dS[i * T + j] = P[i * T + j] * (dP[i * T + j] - rowDot) * scale;`,
    wasm: `;; WASM Two-Move Attention via gemm_f32:
;; 1. Q [T x d_k] * K^T [d_k x T] -> Scores [T x T]
(call $gemm_f32 (local.get $qPtr) (local.get $kTransposedPtr) (local.get $scoresPtr)
                (local.get $T) (local.get $dHead) (local.get $T))
;; 2. After in-place scale + causal mask + row softmax:
;;    AttnProbs [T x T] * V [T x d_k] -> Context [T x d_k]
(call $gemm_f32 (local.get $scoresPtr) (local.get $vPtr) (local.get $contextPtr)
                (local.get $T) (local.get $T) (local.get $dHead))`,
    webgpu: `// WebGPU WGSL Flash-Style Attention Score + Online Softmax Kernel
@compute @workgroup_size(16, 16, 1)
fn attention_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qPos = gid.y; let kPos = gid.x;
  if (qPos >= dims.T || kPos >= dims.T) { return; }
  if (dims.Causal == 1u && kPos > qPos) {
    Scores[qPos * dims.T + kPos] = -1e9;
    return;
  }
  var dot : f32 = 0.0;
  for (var d : u32 = 0u; d < dims.dHead; d = d + 1u) {
    dot = dot + Q[qPos * dims.dHead + d] * K[kPos * dims.dHead + d];
  }
  Scores[qPos * dims.T + kPos] = dot * inverseSqrt(f32(dims.dHead));
}`
  },

  diffusion: {
    title: "7. Continuous DDPM Score Reversal & Discrete Masked Text Diffusion (MDLM)",
    summary:
      "Continuous DDPM trains ε_θ(x_t, t) to predict added Gaussian noise ε and steps backward via Langevin dynamics. Discrete Masked Text Diffusion trains a bidirectional denoiser to unmask [MASK] tokens in parallel by confidence.",
    js: `// A. Continuous DDPM Forward Marginal & Reverse Denoising Step
// Forward q(x_t | x_0): x_t = √(ᾱ_t)·x_0 + √(1 − ᾱ_t)·ε
const x_t = x0.map((v, i) => Math.sqrt(alphaBar[t]) * v + Math.sqrt(1 - alphaBar[t]) * eps[i]);
// Reverse p_θ(x_{t-1} | x_t): subtract predicted noise (score) + Langevin noise σ_t·z
const coef = (1 - alpha[t]) / Math.sqrt(1 - alphaBar[t]);
const x_prev = x_t.map((v, i) => (v - coef * epsHat[i]) / Math.sqrt(alpha[t]) + sigma[t] * z[i]);

// B. Discrete Masked Text Diffusion Parallel Confidence Unmasking
// Unmask the top-k highest-confidence [MASK] positions at each reverse step:
maskedPositions.sort((a, b) => maxProb[b] - maxProb[a]);
for (let k = 0; k < numToUnmask; k++) {
  const pos = maskedPositions[k];
  tokens[pos] = argmaxToken[pos];
}`,
    wasm: `;; WASM Score-Denoising Linear Layers + Sinusoidal Time Embedding Injection
;; Concatenate [x_t (36 floats), sin(t*π), cos(t*π)] -> 38-dim input buffer
(call $gemm_f32 (local.get $xtTimePtr) (local.get $w1Ptr) (local.get $h1Ptr)
                (i32.const 1) (i32.const 38) (i32.const 48))
;; Apply GELU activation then project hidden [1 x 48] -> epsHat [1 x 36]
(call $gemm_f32 (local.get $h1Ptr) (local.get $w2Ptr) (local.get $epsHatPtr)
                (i32.const 1) (i32.const 48) (i32.const 36))`,
    webgpu: `// WebGPU WGSL Parallel DDPM Reverse Langevin Update Step
@compute @workgroup_size(64)
fn ddpm_reverse_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.NumPixels) { return; }
  let mean = (x_t[i] - sched.noiseCoef * epsHat[i]) * sched.invSqrtAlpha;
  x_prev[i] = clamp(mean + sched.sigma * zNoise[i], 0.0, 1.0);
}`
  },

  decision_head: {
    title: "8. Delimiter-Gated Decision Pointer Head & Two-Sided Abstention Gate",
    summary:
      "Scores each candidate option's closing delimiter representation e_i against state representation h in a single forward pass, applies calibrated temperature T, and separates Choice confidence from Act/Abstain gating.",
    js: `// Single-Pass Option Delimiter Scoring + Calibrated Gate
const hProj = matmul(stateVec, W_state, 1, D, D);
const logits = optionVecs.map((optVec) => {
  const oProj = matmul(optVec, W_opt, 1, D, D);
  let dot = 0;
  for (let d = 0; d < D; d++) dot += hProj[d] * oProj[d];
  return dot / Math.sqrt(D);
});
const probs = softmaxInPlace(new Float32Array(logits), 0, K, temperature);
const sorted = Array.from(probs).sort((a, b) => b - a);
// Margin-aware confidence & independent act/review gate
const margin = sorted[0] - (sorted[1] ?? 0);
const confidence = 0.65 * sorted[0] + 0.35 * margin;
const recommendation = confidence >= gateThreshold ? "act" : "review";`,
    wasm: `;; WASM Batch Option Scoring: Stack all K option vectors into [K x D] matrix O
;; Project State h [1 x D] * O^T [D x K] -> Option Logits [1 x K] in one GEMM call!
(call $gemm_f32 (local.get $hProjPtr) (local.get $optProjTransposedPtr) (local.get $logitsPtr)
                (i32.const 1) (local.get $D) (local.get $K))`,
    webgpu: `// WebGPU WGSL Parallel Option Delimiter Dot-Product Readout
@compute @workgroup_size(32)
fn score_option_markers(@builtin(global_invocation_id) gid: vec3<u32>) {
  let optIdx = gid.x;
  if (optIdx >= dims.NumOptions) { return; }
  var dot : f32 = 0.0;
  for (var d : u32 = 0u; d < dims.D; d = d + 1u) {
    dot = dot + hProj[d] * optProj[optIdx * dims.D + d];
  }
  logits[optIdx] = dot * inverseSqrt(f32(dims.D)) / params.temperature;
}`
  }
};
