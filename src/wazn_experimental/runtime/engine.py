"""Answering requests: one state, many instructions, one prefill.

A request with several instructions shares its context, so the backbone
reads it once and the cache is nested:

    [S]                          one prefill, batch 1
      |- Q_1 ... Q_J             one batched pass, cache broadcast J ways
          |- C_11 ... C_JK       one batched pass, cache broadcast N ways

The cost is `L_S + sum_j L_Qj + sum_jk L_Cjk` tokens against the
`sum_jk (L_S + L_Qj + L_Cjk)` of one prompt per label, and the result is
numerically the computation each `(state, instruction, label)` would get on
its own: padding inside a segment is masked and positions are computed from
real lengths. `share_state=False` runs that per-row computation instead, and
the tests hold the two paths to each other.

Hybrid backbones (with recurrent layers) cannot continue past padding, so
they prefill the state once and continue every row with its own instruction
and label, padded only at the end.
"""

from __future__ import annotations

import copy
import gc
import random
from typing import Sequence

import numpy as np
import torch

from ..request import Instruction, PredictOptions, Request
from ..response import Answer, Response, Usage
from .formatting import render_state
from .model import WaznModel, expand_cache
from .tokenization import PromptEncoder, Segments


def _positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(dim=-1) - 1).clamp_min(0)


def _pad_to(rows: Sequence[Sequence[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token rows. -> (ids [n, L], mask [n, L])."""
    length = max(len(r) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, : len(r)] = 1
    return ids.to(device), mask.to(device)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class WaznEngine:
    """Answers `Request`s with a loaded `WaznModel`. Holds no state between
    calls besides counters, so one engine serves any number of requests."""

    def __init__(
        self,
        model: WaznModel,
        *,
        model_name: str = "wazn",
        share_state: bool = True,
        candidate_chunk: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model = model.eval()
        self.model_name = model_name
        self.share_state = share_state
        # Recurrent backbones: branch at most this many labels per forward
        # (None = all at once). Halved automatically on out-of-memory.
        self.candidate_chunk = candidate_chunk
        # Flat path: rows per forward.
        self.batch_size = batch_size
        self.encoder = PromptEncoder(model.tokenizer, model.config.prompt)
        self.device = next(model.parameters()).device
        self.oom_retries = 0

    # ------------------------------------------------------------ public API

    def predict(self, request: Request, options: PredictOptions | None = None) -> Response:
        options = options or PredictOptions()
        if options.group_size is not None:
            response = self._tournament(request, options.group_size, options.top_k, options.seed)
        else:
            response = self._plain(request)
        response.usage_detail = options.usage_detail
        return response

    # ---------------------------------------------------------------- modes

    def _plain(self, request: Request) -> Response:
        ins = request.instructions
        probs, gates, usage, warnings = self._score(request, ins, [i.label_names for i in ins])
        answers = {}
        for i, p, q in zip(ins, probs, gates):
            dist = {name: float(p[k]) for k, name in enumerate(i.label_names)}
            answers[i.name] = self._answer(i, dist, q)
        return Response(self.model_name, answers, usage, warnings=warnings)

    def _tournament(
        self, request: Request, group_size: int, top_k: int, seed: int | None
    ) -> Response:
        """Each instruction's labels are cut into groups of `group_size`,
        every group is scored on its own, its `top_k` advance, and the
        survivors are regrouped until at most `group_size` remain; that final
        group is scored once more. A trailing group of at most `top_k`
        labels advances without a pass (a bye). Every group of every
        instruction in a round shares one prefill of the context."""
        ins = request.instructions
        alive = [list(i.label_names) for i in ins]
        if seed is not None:
            rng = random.Random(seed)
            for names in alive:
                rng.shuffle(names)

        rounds: list[list[dict]] = [[] for _ in ins]
        final: list[tuple[dict[str, float], float | None] | None] = [None] * len(ins)
        usage: Usage | None = None
        warnings: list[str] = []

        while any(f is None for f in final):
            plan: list[tuple[int, list[str], bool]] = []
            byes: list[list[str]] = [[] for _ in ins]
            for j, names in enumerate(alive):
                if final[j] is not None:
                    continue
                if len(names) <= group_size:
                    plan.append((j, names, True))
                    continue
                for g in range(0, len(names), group_size):
                    group = names[g : g + group_size]
                    if len(group) <= top_k:
                        byes[j].extend(group)
                    else:
                        plan.append((j, group, False))

            probs, gates, u, w = self._score(
                request, [ins[j] for j, _, _ in plan], [group for _, group, _ in plan]
            )
            usage = u if usage is None else usage + u
            warnings += [x for x in w if x not in warnings]

            groups: list[list[dict[str, float]]] = [[] for _ in ins]
            advanced: list[list[str]] = [[] for _ in ins]
            for (j, group, is_final), p, q in zip(plan, probs, gates):
                dist = {c: float(p[k]) for k, c in enumerate(group)}
                groups[j].append(dist)
                if is_final:
                    final[j] = (dist, q)
                else:
                    # keep the seeding order among survivors, so the next
                    # round's groups are not stacked by rank
                    top = sorted(np.argsort(-p.numpy(), kind="stable")[:top_k])
                    advanced[j].extend(group[k] for k in top)

            for j in range(len(ins)):
                if not groups[j] and not byes[j]:
                    continue
                entry: dict = {"groups": groups[j]}
                if byes[j]:
                    entry["byes"] = list(byes[j])
                rounds[j].append(entry)
                if final[j] is None:
                    alive[j] = advanced[j] + byes[j]

        answers = {}
        for i, (dist, q), trace in zip(ins, final, rounds):
            answers[i.name] = self._answer(i, dist, q, rounds=trace)
        return Response(self.model_name, answers, usage, warnings=warnings)

    def _answer(
        self, ins: Instruction, dist: dict[str, float], gate: float | None, rounds=None
    ) -> Answer:
        best = max(dist, key=dist.get)
        none_probability = is_none = None
        if gate is not None:
            none_probability = 1.0 - gate
            is_none = gate < self.model.config.none_threshold
        return Answer(
            choice=best,
            confidence=dist[best],
            probabilities=dist,
            none_probability=none_probability,
            is_none=is_none,
            true_label=ins.true_label,
            rounds=rounds,
        )

    # -------------------------------------------------------------- scoring

    def _score(
        self,
        request: Request,
        instructions: Sequence[Instruction],
        label_names: Sequence[Sequence[str]],
    ):
        """Score `label_names[j]` (all of instruction j's labels, or a
        tournament group of them); each label brings its own examples.
        -> (probabilities per set, gate per set, usage, warnings)."""
        state = render_state(rules=request.rules, context=request.context)
        labels = [[i.label(n) for n in names] for i, names in zip(instructions, label_names)]
        texts = [[lab.text for lab in own] for own in labels]
        shots = [[lab.examples for lab in own] for own in labels]
        seg = self.encoder.encode(
            state, [i.text for i in instructions], texts, shots,
            names=[i.name for i in instructions],
        )
        if not self.share_state or not seg.state:
            c_flat = self._encode_flat(seg)
        elif self.model.is_recurrent:
            c_flat = self._encode_shared_recurrent(seg)
        else:
            c_flat = self._encode_shared(seg)
        if not bool(torch.isfinite(c_flat).all()):
            raise FloatingPointError(
                f"the backbone produced non-finite activations in "
                f"{self.model.config.backbone.dtype}; reload with dtype='bfloat16' "
                "(or --dtype bfloat16)"
            )
        idx = torch.tensor(seg.question_idx, dtype=torch.long, device=self.device)
        slot = torch.tensor(seg.slot, dtype=torch.long, device=self.device)
        probs, gate = self.model.judge_sets(c_flat, idx, slot, len(instructions))
        gates = gate.tolist() if gate is not None else [None] * len(instructions)
        return probs, gates, self._usage(seg), seg.warnings

    # ---------------------------------------------------- shared-prefix path

    @torch.no_grad()
    def _prefill(self, seg: Segments):
        model, device = self.model, self.device
        prefix = torch.tensor([seg.state], dtype=torch.long, device=device)
        mask = torch.ones_like(prefix)
        out = model.backbone(
            inputs_embeds=model.embed_ids(prefix),
            attention_mask=mask,
            position_ids=torch.arange(prefix.size(1), device=device).unsqueeze(0),
            use_cache=True,
        )
        return out.past_key_values, mask

    @torch.no_grad()
    def _encode_shared(self, seg: Segments) -> torch.Tensor:
        model, device = self.model, self.device
        j, n, lp = len(seg.questions), len(seg.candidates), len(seg.state)

        # 1. [S], one prefill
        cache, prefix_mask = self._prefill(seg)

        # 2. Q_1..Q_J branch off it, one batched pass
        q_ids, q_mask = _pad_to(seg.questions, self.encoder.pad_id, device)
        lq = q_ids.size(1)
        q_lengths = q_mask.sum(dim=-1)
        cache = expand_cache(cache, torch.tensor([j], device=device))
        q_out = model.backbone(
            inputs_embeds=model.embed_ids(q_ids),
            attention_mask=torch.cat([prefix_mask.expand(j, lp), q_mask], dim=1),
            position_ids=lp + _positions_from_mask(q_mask),
            past_key_values=cache,
            cache_position=torch.arange(lp, lp + lq, device=device),
            use_cache=True,
        )

        # 3. every label of every instruction, one batched pass; the cache is
        #    broadcast in question order, which is the order rows arrive in
        idx = torch.tensor(seg.question_idx, dtype=torch.long, device=device)
        counts = torch.bincount(idx, minlength=j)
        cand_cache = expand_cache(q_out.past_key_values, counts)
        c_ids, c_mask = _pad_to(seg.candidates, self.encoder.pad_id, device)
        ls = c_ids.size(1)
        out = model.backbone(
            inputs_embeds=model.embed_ids(c_ids),
            attention_mask=torch.cat([prefix_mask.expand(n, lp), q_mask[idx], c_mask], dim=1),
            position_ids=(lp + q_lengths[idx]).unsqueeze(-1) + _positions_from_mask(c_mask),
            past_key_values=cand_cache,
            cache_position=torch.arange(lp + lq, lp + lq + ls, device=device),
            use_cache=False,
        )
        rep_pos = torch.tensor([len(r) - 1 for r in seg.candidates], device=device)
        return out.last_hidden_state[torch.arange(n, device=device), rep_pos]

    @torch.no_grad()
    def _encode_shared_recurrent(self, seg: Segments) -> torch.Tensor:
        """The state is prefilled once; each row continues with its own
        instruction and label, `chunk` rows per pass. On out-of-memory the
        chunk is halved and the request retried from the prefill (a branched
        hybrid cache is consumed in place, so it cannot be reused)."""
        rows = [seg.questions[q] + c for q, c in zip(seg.question_idx, seg.candidates)]
        chunk = self.candidate_chunk or len(rows)
        while True:
            try:
                return self._branch_rows(seg, rows, chunk)
            except torch.OutOfMemoryError:
                free_memory()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                self.oom_retries += 1

    def _branch_rows(self, seg: Segments, rows: list[list[int]], chunk: int) -> torch.Tensor:
        model, device = self.model, self.device
        lp, n = len(seg.state), len(rows)
        prefix_cache, prefix_mask = self._prefill(seg)
        reps = []
        for lo in range(0, n, chunk):
            part = rows[lo : lo + chunk]
            m = len(part)
            cache = prefix_cache if lo + chunk >= n else copy.deepcopy(prefix_cache)
            cache = expand_cache(cache, torch.tensor([m], device=device))
            ids, mask = _pad_to(part, self.encoder.pad_id, device)
            out = model.backbone(
                inputs_embeds=model.embed_ids(ids),
                attention_mask=torch.cat([prefix_mask.expand(m, lp), mask], dim=1),
                position_ids=lp + _positions_from_mask(mask),
                past_key_values=cache,
                cache_position=torch.arange(lp, lp + ids.size(1), device=device),
                use_cache=False,
            )
            rep_pos = torch.tensor([len(r) - 1 for r in part], device=device)
            reps.append(out.last_hidden_state[torch.arange(m, device=device), rep_pos])
            del out, cache
        return torch.cat(reps)

    # ------------------------------------------------------- reference path

    @torch.no_grad()
    def _encode_flat(self, seg: Segments) -> torch.Tensor:
        """Every `[state, instruction, label]` as its own sequence, no cache.
        The reference the shared paths are checked against."""
        model, device = self.model, self.device
        rows = [seg.state + seg.questions[q] + c for q, c in zip(seg.question_idx, seg.candidates)]
        reps = []
        for lo in range(0, len(rows), self.batch_size):
            part = rows[lo : lo + self.batch_size]
            ids, mask = _pad_to(part, self.encoder.pad_id, device)
            out = model.backbone(
                inputs_embeds=model.embed_ids(ids),
                attention_mask=mask,
                position_ids=_positions_from_mask(mask),
                use_cache=False,
            )
            rep_pos = torch.tensor([len(r) - 1 for r in part], device=device)
            reps.append(out.last_hidden_state[torch.arange(len(part), device=device), rep_pos])
        return torch.cat(reps)

    # ------------------------------------------------------------ accounting

    @staticmethod
    def _usage(seg: Segments) -> Usage:
        lp = len(seg.state)
        q_lens = [len(q) for q in seg.questions]
        c_lens = [len(c) for c in seg.candidates]
        questions, cands = sum(q_lens), sum(c_lens)
        return Usage(
            input_tokens=lp + questions + cands,
            prefix_tokens=lp,
            question_tokens=questions,
            candidate_tokens=cands,
            unshared_equivalent_tokens=sum(
                lp + q_lens[j] + c for j, c in zip(seg.question_idx, c_lens)
            ),
        )
