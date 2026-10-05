# Changelog

## 0.1.0b2

- Several instructions in one request no longer cost two backbone passes each on recurrent backbones: the state is prefilled once and every label row carries its instruction, so any request is one prefill and one branch pass. 10 instructions of 4 labels: 433 to 154 ms on an RTX 5090.
- On CUDA, the backbone's short convolution runs on fla's Triton kernel when the `causal-conv1d` package is not installed (it needs a CUDA toolkit matching torch's to build). transformers no longer falls back to its PyTorch version or warns about it; requests are 3 to 11% faster, most on long contexts.
- README: speed table measured on an RTX 5090 (few and many labels, tournament and all at once, a long context, many instructions).

## 0.1.0b1.post1

Documentation only; the code is 0.1.0b1's.

- README: Hugging Face and blog links, and a speed table measured on an RTX 5090 (long context, a 219-label tournament, 10 instructions).
- README corrections: tournaments encode labels once per request, `serve --dtype` defaults to `auto`, the validation rules, and which way context truncation goes.

## 0.1.0b1

First beta: the GPU path (CUDA, bfloat16, fused linear-attention kernels) is the supported fast path.

- Tournaments encode every label once per request; later rounds only rerun the judge. A 219-label, 3-round tournament makes 2 backbone passes instead of 6.
- Recurrent backbones read each instruction once, in the prefix: a label row carries only the label, so the backbone computes less than half the tokens it did (5,097 to 2,263 on the 219-label request).
- Fixed: branched recurrent caches outlived their request until Python's cyclic gc ran, so GPU memory grew across requests (3.7 to 14 GB over 12 requests of 219 labels) until one ran out of memory and was retried at half the chunk.
- `usage` for a tournament now counts the single encoding pass, not one pass per round.
- Truncation warnings name a label by its position in the instruction's full label list.

## 0.1.0a2

- Default model is now `numidlabs/wazn-2b-v0.1` (was `numid/wazn-2b-v0.1`, which does not exist).

## 0.1.0a1

First experimental release.

- `Request` / `Instruction` / `Label` builders, validated on construction. Few-shot examples belong to their label: `Label(name, definition, examples=[...])`, and `{"definition", "examples"}` on the wire; examples anywhere else are refused.
- `Client` for a running server; `Wazn` for in-process use. Both return the same `Response`.
- `wazn-experimental serve`: a local HTTP API (`POST /predict`, `GET /info`, `GET /health`).
- Tournament answering for large label sets, set per instruction: `Instruction(..., tournament=Tournament(group_size, top_k, seed))`, `"tournament": {...}` on the wire. Instructions with and without one share a request.
- NONE gate surfaced as `Answer.none_probability` / `Answer.is_none`.
- `dtype="auto"` runs the checkpoint's own dtype (bfloat16) on every device; non-finite activations (e.g. under `float16`) raise instead of returning garbage.
- `[cuda]` extra (Linux): fused linear-attention kernels via flash-linear-attention; a warning at load when CUDA runs without them. `/info` reports `lora` and `fast_kernels`.
- Truncation of context, labels or examples is reported in `Response.warnings`.
- `wazn-experimental export`: training run directory → hub-ready checkpoint (`head.safetensors`, pinned backbone revision, model card stub).
