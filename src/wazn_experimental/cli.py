"""`wazn-experimental` command line.

    wazn-experimental serve   [--model <hub id | dir>] [--port 8000]   (default model: wazn-2b-v0.1)
    wazn-experimental predict request.json            (against a running server)
    wazn-experimental predict request.json --model <hub id | dir>   (in process)
    wazn-experimental export  <run dir> <out dir>     (training run -> checkpoint)

`predict` takes the JSON in `examples/`; `--jsonl` reads one request per
line. Responses go to stdout, logs to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._defaults import DEFAULT_MODEL
from ._version import __version__
from .client import DEFAULT_URL
from .errors import WaznError
from .request import PredictOptions, Request


def _add_model_args(p: argparse.ArgumentParser, default: str | None) -> None:
    p.add_argument("--model", default=default,
                   help=f"hub repo id or local checkpoint directory (default: {default or 'none'})")
    p.add_argument("--revision", help="hub revision of the checkpoint")
    p.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), or a torch device")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                   help="auto: the dtype the checkpoint was trained in")
    p.add_argument("--candidate-chunk", type=int,
                   help="hybrid backbones: labels encoded per forward pass")


def _add_predict_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--group-size", type=int, help="answer by tournament over groups of this size")
    p.add_argument("--top-k", type=int, default=1, help="labels advancing from each group")
    p.add_argument("--seed", type=int, help="shuffle labels before the first round")
    p.add_argument("--usage-detail", action="store_true", help="break token usage down")


def _load(a):
    from .runtime.api import Wazn

    print(f"loading {a.model} ...", file=sys.stderr)
    wazn = Wazn.load(a.model, revision=a.revision, device=a.device, dtype=a.dtype,
                     candidate_chunk=a.candidate_chunk)
    info = wazn.info()
    kernels = {None: "n/a", True: "fused", False: "reference PyTorch (slow)"}[info["fast_kernels"]]
    print(f"loaded {info['model']} on {info['device']} in {info['dtype']}; LoRA {info['lora']}; "
          f"linear-attention kernels: {kernels}", file=sys.stderr)
    return wazn


def cmd_serve(a) -> None:
    wazn = _load(a)
    wazn.serve(host=a.host, port=a.port, max_labels=a.max_labels,
               max_instructions=a.max_instructions)


def cmd_predict(a) -> None:
    text = Path(a.request).read_text() if a.request != "-" else sys.stdin.read()
    if a.jsonl:
        requests = [Request.from_json(line) for line in text.splitlines() if line.strip()]
    else:
        requests = [Request.from_json(text)]
    options = PredictOptions(a.group_size, a.top_k, a.seed, a.usage_detail)

    if a.model:
        wazn = _load(a)
        responses = [wazn.predict_with(r, options) for r in requests]
    else:
        from .client import Client

        with Client(a.url) as client:
            responses = [client.predict(r, **vars(options)) for r in requests]

    for r in responses:
        for w in r.warnings:
            print(f"warning: {w}", file=sys.stderr)
    if a.jsonl:
        print("\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in responses))
    else:
        print(responses[0].to_json())


def cmd_export(a) -> None:
    from .runtime.export import export

    out = export(a.run_dir, a.out_dir, name=a.name, head=a.head,
                 revision=a.backbone_revision, repo=a.repo)
    print(f"wrote {out}", file=sys.stderr)
    for f in sorted(out.iterdir()):
        print(f"  {f.name}  {f.stat().st_size:,} bytes", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="wazn-experimental", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"wazn-experimental {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="load a model and serve it over HTTP")
    _add_model_args(s, default=DEFAULT_MODEL)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-labels", type=int, default=512,
                   help="labels per request, outside a tournament")
    s.add_argument("--max-instructions", type=int, default=64)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("predict", help="answer a request file")
    s.add_argument("request", help="a request JSON file, or - for stdin")
    s.add_argument("--jsonl", action="store_true", help="one request per line")
    s.add_argument("--url", default=DEFAULT_URL, help="server to ask (ignored with --model)")
    _add_model_args(s, default=None)  # no --model: ask the server at --url
    _add_predict_options(s)
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("export", help="turn a training run into a publishable checkpoint")
    s.add_argument("run_dir")
    s.add_argument("out_dir")
    s.add_argument("--name", help="model name reported in responses (default: out dir name)")
    s.add_argument("--head", default="head.pt", help="head file inside the run dir")
    s.add_argument("--backbone-revision", help="pin the backbone to this commit "
                   "(default: the hub's current one)")
    s.add_argument("--repo", help="hub repo id, for the model card's usage snippet")
    s.set_defaults(fn=cmd_export)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except WaznError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
