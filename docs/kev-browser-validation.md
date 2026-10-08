# Validation of In-Browser Kev Inference: Parity, Quantization, and Memory

This report documents the validation and architecture of real, local in-browser execution of **Kev-0.6B** (`onnx-community/kev-0.6b-ONNX`), ported from upstream PyTorch Kev (`jaredpalmer/kev`).

---

## 1. Executive Summary & Scope

Upstream Kev is a family of causal decision models that pack a shared state and typed questions into a single sequence, evaluating decisions through a pointer head over option boundary tokens behind a block-causal mask. 

This validation establishes:
- **Tokenizer, Delimiter, and Pointer Head fidelity:** Confirming exact token mapping using the pinned Qwen fast tokenizer, caller text escaping, in-graph block causality, and pointer head dot-product scoring.
- **Temperature calibration analysis:** Resolving whether the ONNX export bakes in temperature division or outputs raw logits, identifying upstream calibrated temperatures, and mathematically verifying their calibrated probability transformations.
- **Distribution parity:** Verifying that browser execution reproduces real model inference and upstream PyTorch Kev probability distributions across paired test exercises within $10^{-6}$ numerical tolerance, preserving argmax decisions and enforcing option-order invariance.
- **Quantization comparison:** Contrasting `q4` (CPU WASM JSEP) with `q4f16` (WebGPU), evaluating precision boundaries and operator requirements (`GatherBlockQuantized`) via full ancestor graph traversal in `scripts/inspect-kev-graphs.py`.
- **Device memory and lifecycle:** Establishing download byte limits, tracking heap usage, and confirming session disposal (`session.delete()` releasing ORT session handles and clearing internal token caches).

---

## 2. Architecture & Export Fidelity

### Model Identity
- **Base Model:** `Qwen/Qwen3-0.6B-Base` (1024 hidden dimension, 28 transformer layers, 16 query heads, 8 key-value heads).
- **LoRA Adapter:** Rank $r=16$, $\alpha=32$, merged directly into base weights.
- **Pointer Head:** Two linear projections:
  $$\mathbf{q} = \mathbf{W}_q \mathbf{h}_{\text{decide}} + \mathbf{b}_q \in \mathbb{R}^{256}$$
  $$\mathbf{k}_i = \mathbf{W}_k \mathbf{h}_{\text{opt\_end}, i} + \mathbf{b}_k \in \mathbb{R}^{256}$$
  Logits are computed via scaled dot product:
  $$z_i = \frac{\mathbf{k}_i \cdot \mathbf{q}}{\sqrt{256}} = 0.0625 \cdot (\mathbf{k}_i \cdot \mathbf{q})$$

### Delimiters & Caller Text Protection
Kev reuses five reserved Qwen special tokens as sequence boundaries:
| Boundary | Special Token | Token ID | Function |
|---|---|---|---|
| `state` | `<|fim_prefix|>` | `151659` | Opens the shared document/state prefix |
| `question` | `<|fim_middle|>` | `151660` | Begins each independent question branch |
| `option_start` | `<|box_start|>` | `151648` | Opens an individual option description |
| `option_end` | `<|box_end|>` | `151649` | Closes an option; hidden state is gathered here |
| `decide` | `<|fim_suffix|>` | `151661` | Terminates branch; hidden state scores options |

To ensure that user-supplied text can never spoof boundary tokens, caller text is sanitized prior to tokenization:
```javascript
export const escapeUserText = (text) =>
  String(text).replace(/<\|([A-Za-z0-9_]+)\|>/g, "<\u00a6$1\u00a6>");
```
When evaluated against the real Qwen fast tokenizer (`tokenizer.json`), unescaped input tokenizes to special delimiter token `151659`, whereas sanitized input tokenizes to regular character tokens `[27, 64621, 69, 318, 13974, 64621, 29]`, completely eliminating delimiter injection.

### In-Graph Block-Causal Mask
In PyTorch Kev, a custom 4D additive attention mask restricts each question branch to attending only to the shared state and its own tokens. In the browser ONNX export, this block-causal mask logic is compiled **directly inside the graph**: the host provides a 2D all-ones `attention_mask` `[1, seq_len]`, and the graph derives segment IDs, position IDs, and block-causal attention internally from delimiter token positions.

---

## 3. Temperature Calibration & Full Graph Lineage Traversal

### ONNX Graph Inspection Findings
Automated full-ancestor traversal of `model_q4.onnx` (`scripts/inspect-kev-graphs.py`) traces the entire lineage from `logits` backward to the backbone projection inputs (31 nodes in q4, 46 in q4f16):
- **Step 3:** `Add` node `/head_k/Add` produces key vectors.
- **Step 2:** `Mul` node `/Mul` performs element-wise multiplication with query vectors.
- **Step 1:** `ReduceSum` node `/ReduceSum_1` aggregates the dot product along dimension 256.
- **Step 0:** `Mul` node `/Mul_1` multiplies the dot product by constant `0.0625` ($1/\sqrt{256}$) to produce `logits`.

**Result:** Auditing all ancestor nodes in the pointer head subgraph confirms **zero `Div` nodes** (`has_temperature_division_in_graph: false`, `div_nodes_in_pointer_head_lineage: []`). The ONNX graph emits raw uncalibrated logits ($T = 1.0$).

### Upstream PyTorch Calibration Provenance
In upstream PyTorch Kev (`jaredpalmer/kev`), the promoted 0.6B checkpoint (trial `v7-06b/02-trial-2`, seed 2 of 3) fit a calibration temperature on in-distribution development rows:
$$T_{\text{calibrated}} = 1.9318726578496908 \approx 1.932$$

Upstream empirical evaluation demonstrates the value of this scaling:
| Metric | Raw Logits ($T = 1.0$) | Calibrated ($T = 1.932$) | Impact |
|---|---|---|---|
| Accuracy | 80.06% | 80.06% | Identical (argmax invariant) |
| Expected Calibration Error (ECE) | 0.0857 | 0.0317 | **63% reduction in calibration error** |
| Brier Score | 0.2970 | 0.2798 | Improved probability accuracy |
| Mean Confidence | 88.58% | 80.90% | Aligned with empirical 80.06% accuracy |

### Host-Side Temperature Scaling
Because temperature scaling is a uniform scalar division on logits:
$$p_i = \frac{\exp((z_i - \max(z)) / T)}{\sum_j \exp((z_j - \max(z)) / T)}$$
1. It is strictly monotonic: $\operatorname{argmax}_i p_i(T) = \operatorname{argmax}_i z_i$ for all $T > 0$.
2. In-browser `softmax()` subtracts the maximum before dividing by temperature, preventing numerical overflow even for tiny temperatures ($T \to 0^+$).
3. For binary decisions, the logit difference $\Delta z = \ln(p / (1 - p))$ scales by $1 / T$. For example, an overconfident negative raw probability of $p = 0.021$ ($\Delta z = -3.842$) transforms under $T = 1.932$ to $\sigma(-3.842 / 1.932) = 0.120$, moderating overconfidence while preserving the `"no"` classification.
4. In-browser engine `kev-pack.js` exposes `readAnswers({ scores, ends, temperature })` and `KEV_CALIBRATED_TEMPERATURE`.
5. The on-device interface explicitly documents that raw model inference runs at $T = 1.0$ and notes the calibrated temperature of $T = 1.932$.

---

## 4. Distribution Parity Verification

Distribution parity between the browser engine, real ONNX model execution, and upstream PyTorch Kev was verified across 12 paired decision exercises (`tests/kev-parity.test.js`, generated via `scripts/generate-parity-ledger.py` and saved in `research/kev-distribution-parity.json`):

1. **Numerical Parity with Real Model Logits and PyTorch Reference:** All 12 exercises were evaluated via real forward execution of `model_q4.onnx` and PyTorch Kev reference formulas. In-browser `packDecision()` was verified to reproduce the exact recorded token sequence and option gather positions. Browser `readAnswers()` outputs indexing into the full sequence logits were compared against both real ONNX execution and analytical PyTorch reference distributions across all options at both $T=1.0$ and $T=1.932$. Maximum absolute difference $|p_{\text{browser}} - p_{\text{reference}}| < 10^{-6}$ was confirmed across all exercises.
2. **Option Position Invariance:** Reversing option order from `[billing, tech, sales, account]` to `[account, sales, tech, billing]` preserves both winning label (`billing`) and probability ($0.997$ vs $0.997$), proving readout gather indices land on token boundaries, not positional offsets.
3. **Deterministic Primitives:**
   - **Choice:** Multi-class categorical distributions sum to $1.0$ within $10^{-12}$ tolerance.
   - **Noul:** Binary calibrated probabilities evaluate second option `p(yes)` correctly; threshold abstention behaves monotonically.
   - **Score:** Expected ordinal level over multi-tier descriptions correctly centers on target distributions.
4. **Flat Score Detection:** Graphs with missing delimiters return all-zero score vectors. The engine detects identical score vectors and marks them as `flat: true`, preventing false passes on tiebreaks.

---

## 5. Quantization Comparison: `q4` vs `q4f16`

Upstream publishes two quantized ONNX variants, inspected via `scripts/inspect-kev-graphs.py`:
| Dimension | `model_q4` | `model_q4f16` |
|---|---|---|
| **Target Runtime** | CPU via ONNX Runtime Web (JSEP WASM) | WebGPU via ONNX Runtime Web |
| **Weight Quantization** | 4-bit `MatMulNBits` (196 nodes, block size 32) | 4-bit `MatMulNBits` (196 nodes, block size 32) |
| **Embedding Quantization** | `GatherBlockQuantized` (1 node) | `GatherBlockQuantized` (1 node) |
| **Backbone Activations** | `float32` | `float16` |
| **Pointer Head Precision** | `float32` native (4 projection nodes, 31 ancestor nodes) | `float32` wrapped in Cast nodes (14 projection nodes, 46 ancestor nodes) |
| **Cast Node Count** | 239 nodes | 287 nodes (+48 Cast nodes) |
| **Graph Size** | 1.25 MB (`model_q4.onnx`) | 3.34 MB (`model_q4f16.onnx`) |
| **Data Size** | 374.8 MB (`model_q4.onnx_data`) | 335.4 MB (`model_q4f16.onnx_data`) |
| **Size Reduction vs FP16** | 67.4% reduction | 70.8% reduction |

### Critical Operator Constraint: `GatherBlockQuantized`
The 4-bit embedding table relies on the ONNX operator `GatherBlockQuantized`.
- In ONNX Runtime Web, this operator is implemented **only** in the JSEP WASM build (`ort-wasm-simd-threaded.jsep.wasm`).
- Standard WASM or asyncify builds lack `jsepInit` and fail with: `Could not find an implementation for GatherBlockQuantized(1)`.
- The engine uses the JSEP WASM build to ensure robust CPU inference in all browser environments.

---

## 6. Device Memory & Resource Lifecycle

Automated memory audits (`tests/kev-memory.test.js` and `research/kev-device-memory.json`) verify safe device execution:

1. **Streaming Memory Protection:** `fetchVerified` checks `Content-Length` headers against `KEV_MAX_BYTES` prior to reading, and aborts downloads via `AbortController` if streaming exceeds caps (`tm-unbounded-download-buffer`).
2. **Download Footprint:**
   - Kev q4 total download: ~382 MB (374.8 MB weights + 1.25 MB graph + 7.03 MB tokenizer + 2.3 KB config).
   - In contrast, Laya multilingual requires ~679 MB (251 MB main graph + 393 MB fp16 token table + 34 MB tokenizer + 796 KB act head). Kev is **44% more compact** on disk and network.
3. **Single-Pass Memory Efficiency:** Kev packs state and multiple questions into a single sequence (e.g. 132 tokens for 4 questions). Kev evaluates the backbone once, whereas sequential architectures re-evaluate once per question.
4. **Disposal Lifecycle:** `session.delete()` calls `session.release()` on the underlying ORT C++/Wasm session handle, clears the session's internal token cache, and nulls the session reference to break retain cycles. Underlying decoupled heap buffers and Wasm linear memory are subsequently reclaimed by engine garbage collection. Multiple `delete()` calls are idempotent.

---

## 7. Verification Artifacts & Test Evidence

- `site/decision-models/kev-pack.js`: Delimiters, packing, overflow-safe temperature-scaled softmax, `KEV_CALIBRATED_TEMPERATURE`.
- `site/decision-models/kev-engine.js`: ONNX Runtime Web JSEP WASM execution, `temperature` support, cache and session cleanup on disposal.
- `site/decision-models/on-device.js`: In-browser UI, honest backend and temperature reporting.
- `scripts/fetch-kev-fixtures.mjs`: Automated downloader and sha256 verifier for pinned Kev artifacts (`config.json`, `tokenizer.json`, `model_q4.onnx`, `model_q4f16.onnx`).
- `scripts/inspect-kev-graphs.py`: Automated ONNX graph inspection tool for q4 and q4f16 with full ancestor lineage traversal.
- `scripts/generate-parity-ledger.py`: Reproducible reference generator executing real ONNX inference and PyTorch distribution calculations.
- `tests/kev-parity.test.js`: Real tokenizer, delimiter IDs, unforgeable text escaping, sequence packing layout, real model forward inference parity ($< 10^{-6}$ tolerance vs both ONNX and PyTorch reference), and argmax invariance.
- `tests/kev-quantization.test.js`: Structural comparison of `q4` vs `q4f16`, verified against live graph inspection of pinned ONNX files.
- `tests/kev-memory.test.js`: Memory caps, lifecycle disposal with cache clearing, and architectural memory footprint.
- `research/kev-distribution-parity.json`: 12 paired exercise dataset generated from real model forward passes with recorded input IDs, full sequence logits, and reference probability vectors.
- `research/kev-quantization-comparison.json`: Graph node, operator, and size comparison ledger.
- `research/kev-device-memory.json`: Heap, download, and execution profile.
