# Validation of In-Browser Kev Inference: Parity, Quantization, and Memory

This report documents the validation and architecture of real, local in-browser execution of **Kev-0.6B** (`onnx-community/kev-0.6b-ONNX`), ported from upstream PyTorch Kev (`jaredpalmer/kev`).

---

## 1. Executive Summary & Scope

Upstream Kev is a family of causal decision models that pack a shared state and typed questions into a single sequence, evaluating decisions through a pointer head over option boundary tokens behind a block-causal mask. 

Previous reviews identified unmet verification requirements:
- **Tokenizer, Delimiter, and Pointer Head fidelity:** Confirming exact token mapping, caller text escaping, in-graph block causality, and pointer head dot-product scoring.
- **Temperature calibration analysis:** Resolving whether the ONNX export bakes in temperature division or outputs raw logits, identifying upstream calibrated temperatures, and documenting their effect.
- **Distribution parity:** Verifying that browser execution matches upstream PyTorch Kev probability distributions, preserves argmax decisions, and enforces option-order invariance.
- **Quantization comparison:** Contrasting `q4` (CPU WASM JSEP) with `q4f16` (WebGPU), evaluating precision boundaries and operator requirements (`GatherBlockQuantized`).
- **Device memory and lifecycle:** Establishing download byte limits, tracking heap usage, and confirming clean session disposal.

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
Because the fast tokenizer tokenizes `<\u00a6...` as regular character sequences rather than special tokens, delimiter tokens cannot be injected.

### In-Graph Block-Causal Mask
In PyTorch Kev, a custom 4D additive attention mask restricts each question branch to attending only to the shared state and its own tokens. In the browser ONNX export, this block-causal mask logic is compiled **directly inside the graph**: the host provides a 2D all-ones `attention_mask` `[1, seq_len]`, and the graph derives segment IDs, position IDs, and block-causal attention internally from delimiter token positions.

---

## 3. Temperature Calibration & Graph Inspection

### ONNX Graph Inspection Findings
Inspection of the exported ONNX model (`model_q4.onnx`) reveals the exact final node sequence producing `logits`:
- **Step 3:** `Add` node `/head_k/Add` produces key vectors.
- **Step 2:** `Mul` node `/Mul` performs element-wise multiplication with query vectors.
- **Step 1:** `ReduceSum` node `/ReduceSum_1` aggregates the dot product along dimension 256.
- **Step 0:** `Mul` node `/Mul_1` multiplies the dot product by constant `0.0625` ($1/\sqrt{256}$) to produce `logits`.

**Result:** There is **no division by temperature** in the exported ONNX graph. The ONNX graph emits raw uncalibrated logits ($T = 1.0$).

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
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
1. It is strictly monotonic: $\operatorname{argmax}_i p_i(T) = \operatorname{argmax}_i z_i$ for all $T > 0$.
2. In-browser engine `kev-pack.js` exposes `readAnswers({ scores, ends, temperature })` and `KEV_CALIBRATED_TEMPERATURE`.
3. The on-device interface explicitly documents that raw model inference runs at $T = 1.0$ and notes the calibrated temperature of $T = 1.932$.

---

## 4. Distribution Parity Verification

Distribution parity between the browser engine and upstream PyTorch Kev was verified across 12 standard decision exercises (`tests/kev-parity.test.js` and `research/kev-distribution-parity.json`):

1. **Option Position Invariance:** Reversing option order from `[billing, tech, sales, account]` to `[account, sales, tech, billing]` preserves both winning label (`billing`) and probability ($0.995$ vs $0.993$), proving readout gather indices land on token boundaries, not positional offsets.
2. **Deterministic Primitives:**
   - **Choice:** Multi-class categorical distributions sum to $1.0$ within $10^{-12}$ tolerance.
   - **Noul:** Binary calibrated probabilities evaluate second option `p(yes)` correctly; threshold abstention behaves monotonically.
   - **Score:** Expected ordinal level over multi-tier descriptions correctly centers on target distributions.
3. **Out-of-Domain Ranking:** On phishing credential detection, the model scores a credential grab at $0.142$ versus a harmless message at $0.055$. The ranking is correct, but both fall below the $0.5$ midpoint—demonstrating that ordering holds out-of-domain even when raw thresholds do not.
4. **Flat Score Detection:** Graphs with missing delimiters return all-zero score vectors. The engine detects identical score vectors and marks them as `flat: true`, preventing false passes on tiebreaks.

---

## 5. Quantization Comparison: `q4` vs `q4f16`

Upstream publishes two quantized ONNX variants:
| Dimension | `model_q4` | `model_q4f16` |
|---|---|---|
| **Target Runtime** | CPU via ONNX Runtime Web (JSEP WASM) | WebGPU via ONNX Runtime Web |
| **Weight Quantization** | 4-bit `MatMulNBits` (block size 32) | 4-bit `MatMulNBits` (block size 32) |
| **Embedding Quantization** | `GatherBlockQuantized` | `GatherBlockQuantized` |
| **Backbone Activations** | `float32` | `float16` |
| **Pointer Head Precision** | `float32` native (4 nodes) | `float32` wrapped in Cast nodes (14 nodes) |
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
4. **Disposal Lifecycle:** `session.delete()` releases internal ORT session handles and Wasm linear memory buffers, ensuring no orphaned allocations persist. Multiple `delete()` calls are idempotent.

---

## 7. Verification Artifacts & Test Evidence

- `site/decision-models/kev-pack.js`:Delimiters, packing, temperature-scaled softmax, `KEV_CALIBRATED_TEMPERATURE`.
- `site/decision-models/kev-engine.js`: ONNX Runtime Web JSEP WASM execution, `temperature` support.
- `site/decision-models/on-device.js`: In-browser UI, honest backend and temperature reporting.
- `tests/kev-parity.test.js`: Tokenizer, delimiter IDs, packing arithmetic, and distribution parity tests.
- `tests/kev-quantization.test.js`: Structural comparison of `q4` vs `q4f16`.
- `tests/kev-memory.test.js`: Memory caps, lifecycle disposal, and architectural memory footprint.
- `research/kev-distribution-parity.json`: 12-exercise parity dataset.
- `research/kev-quantization-comparison.json`: Graph node, operator, and size comparison ledger.
- `research/kev-device-memory.json`: Heap, download, and execution profile.
