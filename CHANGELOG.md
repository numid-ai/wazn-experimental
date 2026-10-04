# Changelog

## 0.1.0a2

- Default model is now `numidlabs/wazn-2b-v0.1` (was `numid/wazn-2b-v0.1`, which does not exist).

## 0.1.0a1

First experimental release.

- `Request` / `Instruction` / `Label` builders, validated on construction. Few-shot examples belong to their label: `Label(name, definition, examples=[...])`, and `{"definition", "examples"}` on the wire; examples anywhere else are refused.
- `Client` for a running server; `Wazn` for in-process use. Both return the same `Response`.
- `wazn-experimental serve`: a local HTTP API (`POST /predict`, `GET /info`, `GET /health`).
- Tournament answering for large label sets, set per instruction: `Instruction(..., tournament=Tournament(group_size, top_k, seed))`, `"tournament": {...}` on the wire. Instructions with and without one share a request. The backbone encodes every label once per request; rounds only rerun the judge.
- NONE gate surfaced as `Answer.none_probability` / `Answer.is_none`.
- `dtype="auto"` runs the checkpoint's own dtype (bfloat16) on every device; non-finite activations (e.g. under `float16`) raise instead of returning garbage.
- `[cuda]` extra (Linux): fused linear-attention kernels via flash-linear-attention; a warning at load when CUDA runs without them. `/info` reports `lora` and `fast_kernels`.
- Truncation of context, labels or examples is reported in `Response.warnings`.
- `wazn-experimental export`: training run directory → hub-ready checkpoint (`head.safetensors`, pinned backbone revision, model card stub).
