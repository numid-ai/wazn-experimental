# wazn-experimental

🤗 [Model on Hugging Face](https://huggingface.co/numidlabs/wazn-2b-v0.1) · 📝 [Technical blog](https://blog.djellalmohamedaniss.workers.dev/posts/wazn-2b-release/)

Try out **Wazn** on your own machine.

Wazn is a decision model. You give it a context, optional rules, and one or more instructions. Each instruction comes with labels (and optionally a definition and a few examples for each), and Wazn returns a probability distribution over those labels. It doesn't generate text: it reads every label against the context and compares them.

The label set is yours, per request, with no retraining: two labels or a thousand. Large sets (hundreds to 1,000+ classes) are handled by [tournaments](#many-labels-tournaments).

> **Beta.** This package is for testing the model, not for production. Requests run one at a time, and the API may change between `0.x` releases.

## Speed

| Config | Instructions × labels | Input tokens (context / instructions / labels) | Median | p90 | Inference health |
|---|---|---|---|---|---|
| Long context (NDA, ContractNLI) | 1 × 10 | 6,485 (6,317 / 14 / 154) | **171 ms** | 173 ms | ✅ OK |
| Normal context, many labels | 1 × 219 (tournament, groups of 10, top 2) | 1,052 (60 / 13 / 979) | **94 ms** | 94 ms | ✅ OK |
| Normal context, many instructions (incident report) | 10 × 4 | 1,621 (648 / 189 / 784) | 433 ms | 434 ms | ⚠️ Needs a fix: each instruction runs its own backbone passes, one after another (~21 passes), so latency grows with the number of instructions rather than with tokens |

`numidlabs/wazn-2b-v0.1` on an NVIDIA RTX 5090, bfloat16, `[server,cuda]`. Server-side time over 10 requests sent one after another, after a warm-up request; no input was truncated. The first request after a start is slower, because it compiles the GPU kernels.

## Install

Python 3.11+. Use a virtual environment, and pick the install that matches what you want to do:

| You want | Install |
|---|---|
| Run the model locally (server + in-process use) | `pip install --pre "wazn-experimental[server]"` |
| Same, on Linux with an NVIDIA GPU, with fused kernels | `pip install --pre "wazn-experimental[server,cuda]"` |
| Only talk to a server running elsewhere (no torch) | `pip install --pre wazn-experimental` |

```bash
python -m venv .venv && source .venv/bin/activate
pip install --pre "wazn-experimental[server]"
wazn-experimental --help        # check it installed
```

With [uv](https://docs.astral.sh/uv/): `uv pip install --pre "wazn-experimental[server]"`, or run the server without installing anything into your project: `uvx --prerelease allow --from "wazn-experimental[server]" wazn-experimental serve`.

Notes:

- **`--pre` is needed** while every release is a pre-release (`0.1.0aN`, `0.1.0bN`); without it pip reports "no matching distribution". You can also pin a version, e.g. `wazn-experimental==0.1.0b1`.
- The `[server]` extra pulls in PyTorch, which is a large download (several GB on Linux with CUDA). The client-only install is a few small packages.
- On macOS the `[cuda]` extra installs nothing; it is for Linux with an NVIDIA GPU. It installs `flash-linear-attention`, the kernels that matter. transformers may still warn that `causal_conv1d` is not installed: that only covers a short convolution, and is safe to ignore.
- To upgrade: `pip install --pre -U "wazn-experimental[server]"`.

Runs on CUDA, Apple Silicon (MPS) or CPU. The first load downloads the backbone from the Hugging Face Hub.

### Hardware and speed

| Setup | What runs | Expect |
|---|---|---|
| Linux + NVIDIA GPU, `[server,cuda]` | bfloat16, fused Triton kernels for the linear-attention layers | the fast path; see [Speed](#speed) |
| Linux + NVIDIA GPU, `[server]` only | bfloat16, reference PyTorch loop for those layers | correct but far slower; a warning says so at load |
| Apple Silicon | bfloat16 on MPS, reference PyTorch loop | around a second per short request |
| CPU | bfloat16 | slow; fine for a quick check |

The device is picked automatically and the model runs in bfloat16, the dtype it was trained in (`--device` and `--dtype` override them). `--dtype float16` is available, but if it ever gives a "non-finite activations" error, go back to bfloat16. Startup prints what it picked:

```
loaded wazn-2b-v0.1 on cuda:0 in bfloat16; LoRA merged; linear-attention kernels: fused
```

## Quickstart

Start a server with the current model, **wazn-2b-v0.1** (downloaded on first run):

```bash
wazn-experimental serve
# serving on http://127.0.0.1:8000
```

Then, from Python:

```python
from wazn_experimental import Client, Request, Instruction, Label

request = Request(
    rules="Returns handles exchanges and refunds. Billing handles charges.",
    context="My running shoes arrived in the wrong size. Can I swap them for a 10?",
    instructions=[
        Instruction(
            "Which team should handle this?",
            labels=[
                Label("returns", "Exchanges, refunds, wrong or damaged items"),
                Label("shipping", "Delivery status, delays, lost packages"),
                Label("billing", "Charges, invoices, payment problems",
                      examples=["I was charged twice for the same order"]),
            ],
            name="department",
        ),
    ],
)

with Client() as client:                 # http://127.0.0.1:8000
    response = client.predict(request)

answer = response["department"]
print(answer.choice, answer.confidence)  # e.g. returns 0.93
print(answer.probabilities)              # e.g. {'returns': 0.93, 'shipping': 0.04, 'billing': 0.03}
print(answer.none_probability)           # P(none of the labels applies)
```

Or skip the server and load the model in process (e.g. in a notebook). `predict` takes the same arguments and returns the same `Response`:

```python
from wazn_experimental import Wazn

model = Wazn.load()                      # numidlabs/wazn-2b-v0.1; or a hub id / local directory
response = model.predict(request)
```

## Writing requests

| Piece | What it is |
|---|---|
| `context` | The text to answer about. A string, or a list of parts (a document and a message about it), joined by a blank line. |
| `rules` | Optional. The policy or procedure that should settle the answer. |
| `Instruction(text, labels, name=None, true_label=None, tournament=None)` | One question about the context. A request can carry several; the context is read once and shared by all of them. `tournament` answers this instruction by [tournament](#many-labels-tournaments). |
| `Label(name, definition="", examples=[])` | One possible answer. **The model reads the definition**, not the name (it falls back to the name when there is no definition), so write definitions that say when the label applies. `examples` is a list of strings: inputs that should get this label. Only this label reads them, so they help the model tell labels apart without biasing the others. Labels can also be given as plain names or a `{"name": "definition"}` mapping. |
| `true_label` | Optional. If you know the answer, the response tells you whether the model got it right (`answer.correct`). It is never shown to the model. |

Requests are validated when you build them (at least 2 labels per instruction, unique label names, no two labels the model would read identically, a `true_label` that is one of the labels, and so on), so mistakes raise a `RequestError` immediately.

A request is plain JSON on the wire: `request.to_dict()` and `Request.from_file(path)` convert both ways. A label is its definition string, or an object when it has examples:

```json
"criteria": {
  "returns": "Exchanges, refunds, wrong or damaged items",
  "billing": {"definition": "Charges, invoices, payment problems",
              "examples": ["I was charged twice for the same order"]}
}
```

The files in [`examples/requests/`](examples/requests/) are ready to send:

```bash
wazn-experimental predict examples/requests/aerospace_rules_k3.json              # to a running server
wazn-experimental predict examples/requests/aerospace_rules_k3.json --model numidlabs/wazn-2b-v0.1  # in process
```

## Reading responses

`response[name]` (or `response.answer` when there is only one instruction) is an `Answer`:

- `choice`, `confidence`, `probabilities`: the label distribution. It always sums to 1, so the model ranks the labels even when none of them fits.
- `none_probability` and `is_none`: a separate estimate that **no** label applies (for checkpoints with a NONE gate).
- `top(n)`: the `n` most likely labels.
- `correct` and `true_label_probability`: set when you passed a `true_label`.

`response.warnings` lists anything the model didn't read in full. A context that is too long is cut from the left to fit the token budget. The rules come before the context, so **the rules are the first thing lost**. Labels or examples that exceed the per-label budget are cut too. Check it when you send long inputs. `response.usage` reports token counts, and `response.prediction_seconds` the inference time.

## Many labels: tournaments

Wazn can classify among a very large number of classes, even 1,000 or more, in a single request.

By default the model compares all of an instruction's labels at once, which suits up to a few dozen. For a larger label set, give that instruction a tournament:

```python
from wazn_experimental import Tournament

request = Request(
    context="How often should I get my oil changed?",
    instructions=[
        Instruction("What is the user asking for?", labels=intents,   # 1,000 labels
                    tournament=Tournament(group_size=10, top_k=2, seed=0), name="intent"),
        Instruction("What is the tone?", labels=["negative", "neutral", "positive"],
                    name="tone"),                                      # answered as usual
    ],
)
```

The tournament is set per instruction: each one in a request picks its own mode and settings, so a 1,000-label intent question and a 3-label tone question go in the same request. In JSON it is a field of the question: `"tournament": {"group_size": 10, "top_k": 2, "seed": 0}`.

The labels are split into groups of `group_size`. The `top_k` of each group advance, and the survivors are regrouped until a single group is left. The model then never compares more than `group_size` labels at a time, however many there are. With 1,000 labels, groups of 10 and `top_k=2`, that is four rounds (1,000 → 200 → 40 → 8, then the final group). The backbone reads the context and encodes every label once per request; each round only reruns the judge on the groups it needs, so extra rounds are cheap. `seed` shuffles the labels before the first round. `answer.probabilities` covers the final group, and `answer.rounds` records every group along the way (it is `None` for an instruction without a tournament). The server's `--max-labels` limit applies to what is compared at once, so it counts `group_size` for a tournament. See [`examples/tournament.py`](examples/tournament.py).

## Server

```bash
wazn-experimental serve [--model numidlabs/wazn-2b-v0.1] [--revision REV] [--host 127.0.0.1] [--port 8000] \
    [--device auto] [--dtype auto] [--candidate-chunk N] [--max-labels 512] [--max-instructions 64]
```

| Endpoint | |
|---|---|
| `POST /predict` | `{"request": {...}, "options": {"usage_detail": false}}` |
| `GET /info` | model, backbone, token budgets, limits |
| `GET /health` | status, counters, GPU memory |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d "{\"request\": $(cat examples/requests/aerospace_rules_k3.json)}"
```

`--dtype auto` is the checkpoint's own dtype (bfloat16). `--candidate-chunk` caps how many labels go through the backbone per pass; leave it unset, and on running out of GPU memory the server halves it and retries by itself (`oom_retries` in `/health`).

The server binds to localhost by default and has no authentication. Don't expose it.

## Checkpoints

A checkpoint is a small directory: `config.json` plus `head.safetensors` (the judge, NONE gate, tag embeddings and LoRA adapters). The backbone's base weights are not part of it. They are fetched from the Hugging Face Hub at the commit `config.json` pins. `--model` takes a hub repo id or a local directory, and defaults to `numidlabs/wazn-2b-v0.1`.

| Model | Backbone | Hub |
|---|---|---|
| wazn-2b-v0.1 | Qwen3.5-2B-Base | `numidlabs/wazn-2b-v0.1` |

To publish a training run (maintainers):

```bash
wazn-experimental export runs/<run> dist/wazn-2b-v0.1 --name wazn-2b-v0.1 --repo numidlabs/wazn-2b-v0.1
hf upload numidlabs/wazn-2b-v0.1 dist/wazn-2b-v0.1
```

## Development

```bash
uv sync --extra server
uv run pytest                     # runtime tests use a tiny random backbone; no real weights
uv run pytest tests/test_*.py     # client-only tests, no torch needed
```

## License

Apache-2.0. The backbone model keeps its own license; see [NOTICE](NOTICE).
