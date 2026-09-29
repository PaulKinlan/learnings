# Laya multilingual host contract

This package targets `convaiinnovations/laya` multilingual, checkpoint revision
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`, with the laya 0.3.4 host behavior.
[laya_host.py](laya_host.py) is the executable NumPy/CompiledModel reference;
its vendored builder and decoder functions retain their upstream source text.
The [Android sample](android/README.md) implements the same contract in Kotlin.
Only the multilingual embedding-input graphs are supplied.

## A. Tokenizer

Load the complete supplied `tokenizer.json` and `tokenizer_config.json` locally.
The vocabulary contains 256,000 IDs: PAD=0 `<pad>`, SEP=1 `<eos>`, CLS=2 `<bos>`,
UNK=3 `<unk>`, MASK=4 `<mask>`. The builder uses the tokenizer's CLS=2.

This is BPE with byte fallback and ranked merges. Match added/special tokens before
normalization; the normalizer replaces ASCII space with U+2581 `▁`. Metaspace uses
replacement `▁`, prepend_scheme=`always`, split=`true`. Unknown UTF-8 bytes fall
back to `<0xNN>` entries. Do not introduce another Unicode normalization pass.
`<mask>` is special, lstrip=true, rstrip=false, normalized=false, single_word=false.
Builder markers are inserted as integer 4 rather than encoded text. All text
fragments use `add_special_tokens=False`; manually inserted CLS/SEP/MASK must not
be duplicated by a tokenizer postprocessor.

## B. Schema normalization and prompt builder

One graph invocation contains one question row. Preserve question order and option
insertion order. Type order is choice=0, score=1, noul=2. The public schema uses
`type`, `instructions`, `criteria`; the internal builder uses `t`, `ins`, `crit`.

1. Convert a choice criteria list to an insertion-ordered `{label: None}` map;
   duplicate labels collapse as Python dict keys do. A criteria map retains order.
   Non-string instructions use `json.dumps(instructions)` with its default ASCII
   escaping; string instructions pass through.
2. Choice options are the label alone for null/empty descriptions, otherwise
   `label + ": " + description`. Numeric zero and false are real descriptions.
   Score options are `level i: criterion` with zero-based i. Noul options are always
   false then true, defaulting to `false: no, the statement does not hold` and
   `true: yes, the statement holds`. Structured criteria use
   `json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)`.
3. Replace literal `<mask>` in instructions with one ASCII space, then encode
   `<type> question: <instructions>` without automatic special tokens. The angle
   brackets here denote substitution; for example the literal prefix is
   `choice question: `.
4. For each option, replace literal `<mask>` with a space, prepend one ASCII space,
   encode, keep the first 48 text tokens, and prepend integer MASK=4. Each unsqueezed
   option therefore occupies at most 49 positions.
5. Set `opt_budget = 256 - sum(len(option_ids))`. If it is <16, compute
   `per = max(4, (256 - 16) // max(1,K))`, slice each marked option to `[:per]`,
   and recompute opt_budget. Slice instruction IDs to `[:max(8,opt_budget)]`.
6. Start `[CLS] + instruction_ids + [SEP]`. Before each marked option, record its
   zero-based start position, then append the option. Append another SEP.
7. A string state is unchanged; dict/list state uses `json.dumps(state,
   ensure_ascii=False)` with insertion order, JSON booleans/null, comma-space and
   colon-space separators. A conversation list is JSON text, not a chat template.
   Replace literal `<mask>` with spaces and tokenize without special tokens.
8. `room = max(0, N - len(ids) - 1)` after the option-ending SEP. Append
   `state_ids[:room] + [SEP]`, slice the whole sequence to `[:N]`, then keep only
   markers `<N`. Do not repair a final SEP lost by this final slice.
9. Reject before inference if the retained marker count differs from the number
   of rendered options. A head-length estimate alone is not the rejection rule.

Use `max_len=N` and `head_max_len=256` for both N=256 and N=512. Padding follows
construction and is not part of that budget. The supported scope has 2–20 options;
more than 20 is out of scope. Empty state is allowed. S256 retained 68 right-truncated
rows in the 201-row fixture set; S512 retained none. Build for the selected window.

## C. Embedding-input graphs and table lookup

Every graph exposes one `serving_default` signature. All tensors below are float32,
including when weights are stored in FP16. Map buffers by signature name and the
runtime's signature order, never by flatbuffer TensorMap order. N is fixed at 256
or 512; the Android UI uses 256.

| Graph | Direction | Name | Shape |
|---|---|---|---|
| main | input | inputs_embeds | [1,N,768] |
| main | input | attention_mask | [1,N] |
| main | input | qtype_onehot | [1,3] |
| main | output | token_logits | [1,N] |
| main | output | pooled_cls | [1,768] |
| act | input | pooled_cls | [1,768] |
| act | input | feats | [1,4] |
| act | output | act_logits | [1,2] |

Memory-map `token_embeddings_fp16.bin` read-only as row-major little-endian float16
`[256000,768]`, exactly 393,216,000 bytes. Metadata, hash and checkpoint revision
are in `token_embeddings.json`. Right-pad the built IDs with PAD id 0 to N, gather
every table row including padding, and convert the gathered values to float32.
**A padded position uses the actual PAD row, not an all-zero vector.** The complete
table's fp32→fp16→fp32 maximum absolute error was 0 for this checkpoint.

Set attention_mask=1 for real sequence positions, including special tokens, and 0
for padding. qtype_onehot is `[1,0,0]`, `[0,1,0]`, or `[0,0,1]`. Marker positions
and K remain on the host. The main graph scores every position; gather only the
retained markers. pooled_cls is h[:,0] after the typed head layers. The same FP32
action graph serves every supplied window and storage format.

The selected main graph stores fully connected weights in FP16. GPU execution
requests FP32 arithmetic explicitly. Both main and act must compile fully on the
GPU; this sample exposes CPU as a user-selected alternative. Timing includes buffer
writes, run enqueue and output readback, since run can be asynchronous.

## D. Decoder and calibration

The NumPy functions in `laya_host.py` are the arithmetic reference. Keep intermediate
probabilities unrounded. Final scalars use `round(float(value),4)`: ties-to-even on
the represented binary double. Rounded probabilities need not sum to exactly one.

1. Gather K raw marker logits from token_logits in original option order.
2. Compute stable raw softmax. With `k=max(K,2)`, form action features
   `[top1, top1-top2, -sum(p*log(max(p,1e-9)))/log(k), k/255]`, using the two largest
   raw probabilities. Send float32 `[1,4]` features and pooled_cls to the action graph.
   These features never use calibrated or rounded probabilities.
3. `act_probability=softmax(act_logits)[0]`. Do not apply an option temperature to
   act logits or replace this value with option confidence.
4. The option-count bucket suffix is `2` for K≤2, `3-5` for K≤5, `6-10` for K≤10,
   else `11+`, prefixed by the type. Look up `temperature_by_options[bucket]` first,
   then `temperature[qtype]`. Divide logits by `max(1e-3,T)` and apply stable softmax.
5. Choice returns the first argmax label and all ordered option probabilities.
   Score returns `sum(i*p[i])` with zero-based indices and the original legend;
   the reference NumPy integer-index multiplication promotes that reduction.
   Noul returns the continuous true probability p[1], without a boolean threshold.
6. Choice/score confidence is `clip(1-H(p)/log(K),0,1)`, using the log floor 1e-12
   (or 1 for K<2). Noul confidence is `max(p[1],1-p[1])`.

```text
choice: {type, choice, probabilities:{label:p,...}, confidence, action:{act_probability}}
score:  {type, score, legend:{"0":criterion,...}, probabilities:{"0":p,...}, confidence, action:{act_probability}}
noul:   {type, noul:p_true, confidence, action:{act_probability}}
```

`predict` returns `{model:"laya-rl-agent", answers:{id:answer,...},
usage:{input_tokens:sum(real_sequence_lengths), output_tokens:0}}`. Shared state
tokens are counted again for each question; padding is not counted.

The supplied calibration is fitted for S256. `choice:6-10` intentionally keeps T=1
under the fixed language-ECE rule; other supplied buckets use balanced fits.
Conversion parity uses T=1, distinct from calibrated app output. S512 numerical
parity is validated at T=1; calibration quality was not fitted/evaluated for S512.

## E. Executable example and captured gate

The card's Python example runs the staged host locally with the included tokenizer,
table and WFP16 main graph. Its invented Japanese duplicate-payment request returns
`refund` with calibrated probability 0.9954, confidence 0.958 and act_probability 1.0;
its single prompt has 37 tokens. See [Minimal usage](README.md#minimal-usage).

`fixtures/gate_rows_s256.json` contains 201 invented question rows, including state,
question schema, captured IDs/markers, raw marker/act logits, and official T=1
dictionaries. Use the Android debug runner described in `android/README.md` to
re-tokenize and infer every row. Mac CPU validation also exercised actual
`LayaHost.predict` single-row calls for all four supplied graphs; the vendored
builder matched 402/402 S256/S512 IDs and marker positions.

## F. Source observations and limits

Source act_probability saturated at 1.0 on all 201 multilingual fixture rows at
each window (402 window-row observations). This does not establish correctness
or calibrated escalation. Preserve the act graph and formula.

These five Japanese source observations occur at both windows and are not demo
material. Values are uncalibrated source outputs; conversion parity does not fix them.

| Row | Intended reading | Source output | Probability |
|---|---|---|---:|
| ML_A05/churn_risk | true | 0.0021 | 0.0021 |
| ML_C07/intent | technical_help | cancellation | 0.9571 |
| ML_E06/harassment | true | 0.017 | 0.017 |
| ML_E07/threat | true | 0.0 | 0.0 |
| ML_F04/domain | factual_lookup | math_or_logic | 0.8893 |

The domain row is also a recorded ambiguity. No fixture was edited to hide these
observations. Numerical agreement is not a task-quality measurement; validate the
chosen schema, text domain and decision policy independently.
