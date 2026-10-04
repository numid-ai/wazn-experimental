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

import torch

from ..request import Instruction, Request
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

    def predict(self, request: Request, usage_detail: bool = False) -> Response:
        """Answer every instruction of a request, each in its own mode.

        An instruction without a tournament is one set: all of its labels,
        scored in the first round. An instruction with one has its labels
        cut into groups of `group_size`; every group is scored on its own,
        its `top_k` advance, and the survivors are regrouped until at most
        `group_size` remain, which are scored once more as the final group.
        A trailing group of at most `top_k` labels advances without a pass (a
        bye). A label's representation does not depend on the set it is
        judged in, so the backbone encodes every label once, up front, and
        each round only runs the judge over the sets it needs. Results stay
        on the device until the end.
        """
        ins = request.instructions
        reps, usage, warnings = self._encode(request)
        alive = [list(i.label_names) for i in ins]
        for i, names in zip(ins, alive):
            if i.tournament is not None and i.tournament.seed is not None:
                random.Random(i.tournament.seed).shuffle(names)

        # every probability vector and gate the response reports, on the
        # device; the rounds and finals below refer to them by index
        held: list[torch.Tensor] = []

        def hold(t: torch.Tensor) -> int:
            held.append(t.reshape(-1))
            return len(held) - 1

        rounds: list[list[dict]] = [[] for _ in ins]
        final: list[tuple[list[str], int, int | None] | None] = [None] * len(ins)

        while any(f is None for f in final):
            plan: list[tuple[int, list[str], bool]] = []
            byes: list[list[str]] = [[] for _ in ins]
            for j, (i, names) in enumerate(zip(ins, alive)):
                if final[j] is not None:
                    continue
                t = i.tournament
                if t is None or len(names) <= t.group_size:
                    plan.append((j, names, True))
                    continue
                for g in range(0, len(names), t.group_size):
                    group = names[g : g + t.group_size]
                    if len(group) <= t.top_k:
                        byes[j].extend(group)
                    else:
                        plan.append((j, group, False))

            probs, gates = self._judge(
                [[reps[ins[j].name, c] for c in group] for j, group, _ in plan]
            )

            groups: list[list[tuple[list[str], int]]] = [[] for _ in ins]
            advanced: list[list[str]] = [[] for _ in ins]
            for (j, group, is_final), p, q in zip(plan, probs, gates):
                groups[j].append((group, hold(p)))
                if is_final:
                    final[j] = (group, groups[j][-1][1], None if q is None else hold(q))
                else:
                    # keep the seeding order among survivors, so the next
                    # round's groups are not stacked by rank
                    top = sorted(torch.argsort(-p, stable=True)[: ins[j].tournament.top_k].tolist())
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

        host = [t.tolist() for t in torch.cat(held).cpu().split([t.numel() for t in held])]

        def dist(names: list[str], h: int) -> dict[str, float]:
            return dict(zip(names, host[h]))

        answers = {}
        for i, (names, h, q), trace in zip(ins, final, rounds):
            # only a tournament has rounds to report
            if i.tournament is None:
                trace = None
            else:
                for entry in trace:
                    entry["groups"] = [dist(*g) for g in entry["groups"]]
            gate = None if q is None else host[q][0]
            answers[i.name] = self._answer(i, dist(names, h), gate, rounds=trace)
        return Response(self.model_name, answers, usage, warnings=warnings,
                        usage_detail=usage_detail)

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

    def _encode(self, request: Request):
        """Run the backbone over every label of every instruction; each label
        brings its own examples.
        -> ({(instruction name, label name): representation on the device},
            usage, warnings)."""
        instructions = request.instructions
        state = render_state(rules=request.rules, context=request.context)
        labels = [[i.label(n) for n in i.label_names] for i in instructions]
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
        reps = {
            (instructions[j].name, instructions[j].label_names[k]): c
            for j, k, c in zip(seg.question_idx, seg.slot, c_flat)
        }
        return reps, self._usage(seg), seg.warnings

    def _judge(self, sets: Sequence[Sequence[torch.Tensor]]):
        """Judge each set of label representations on its own.
        -> (probabilities per set, gate per set or None), on the device."""
        c_flat = torch.stack([c for own in sets for c in own])
        idx = torch.tensor([s for s, own in enumerate(sets) for _ in own],
                           dtype=torch.long, device=self.device)
        slot = torch.tensor([k for own in sets for k in range(len(own))],
                            dtype=torch.long, device=self.device)
        probs, gate = self.model.judge_sets(c_flat, idx, slot, len(sets))
        gates = list(gate) if gate is not None else [None] * len(sets)
        return probs, gates

    # ---------------------------------------------------- shared-prefix path

    @torch.no_grad()
    def _prefill(self, ids: list[int], cache=None, start: int = 0):
        """Run `ids` as one sequence at positions `start..`, continuing
        `cache` if given (a normal, unbranched update). -> cache"""
        model, device = self.model, self.device
        prefix = torch.tensor([ids], dtype=torch.long, device=device)
        positions = torch.arange(start, start + len(ids), device=device)
        out = model.backbone(
            inputs_embeds=model.embed_ids(prefix),
            attention_mask=torch.ones((1, start + len(ids)), dtype=torch.long, device=device),
            position_ids=positions.unsqueeze(0),
            past_key_values=cache,
            cache_position=positions if cache is not None else None,
            use_cache=True,
        )
        return out.past_key_values

    @torch.no_grad()
    def _encode_shared(self, seg: Segments) -> torch.Tensor:
        model, device = self.model, self.device
        j, n, lp = len(seg.questions), len(seg.candidates), len(seg.state)

        # 1. [S], one prefill
        cache = self._prefill(seg.state)
        prefix_mask = torch.ones((1, lp), dtype=torch.long, device=device)

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
        """Each instruction is folded into the prefix, so a row is only its
        label: one instruction prefills `[S, Q]` in one pass; several prefill
        `[S]` once and extend a copy of it with each `Q_j`. Every row then
        continues from its instruction's cache, `chunk` rows per pass. On
        out-of-memory the chunk is halved and the request retried from the
        prefill (a branched hybrid cache is consumed in place, so it cannot
        be reused)."""
        chunk = self.candidate_chunk or len(seg.candidates)
        while True:
            try:
                return self._branch_rows(seg, chunk)
            except torch.OutOfMemoryError:
                free_memory()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                self.oom_retries += 1

    def _branch_rows(self, seg: Segments, chunk: int) -> torch.Tensor:
        lp, n_q = len(seg.state), len(seg.questions)
        rows_of = [[r for r, q in enumerate(seg.question_idx) if q == j] for j in range(n_q)]
        reps = [None] * len(seg.candidates)
        state_cache = self._prefill(seg.state) if n_q > 1 else None
        for j, question in enumerate(seg.questions):
            if not rows_of[j]:
                continue
            if state_cache is None:
                cache = self._prefill(seg.state + question)
            else:
                # the last instruction may consume the state cache itself
                base = state_cache if j == n_q - 1 else copy.deepcopy(state_cache)
                cache = self._prefill(question, cache=base, start=lp)
            for row, rep in zip(rows_of[j], self._branch_labels(seg, cache, lp + len(question),
                                                                rows_of[j], chunk)):
                reps[row] = rep
        return torch.stack(reps)

    def _branch_labels(
        self, seg: Segments, prefix_cache, lp: int, rows: list[int], chunk: int
    ) -> list[torch.Tensor]:
        """Continue `prefix_cache` (length `lp`) with each of `rows`' labels."""
        model, device = self.model, self.device
        n, reps = len(rows), []
        for lo in range(0, n, chunk):
            part = [seg.candidates[r] for r in rows[lo : lo + chunk]]
            m = len(part)
            cache = prefix_cache if lo + chunk >= n else copy.deepcopy(prefix_cache)
            cache = expand_cache(cache, torch.tensor([m], device=device))
            ids, mask = _pad_to(part, self.encoder.pad_id, device)
            out = model.backbone(
                inputs_embeds=model.embed_ids(ids),
                attention_mask=torch.cat([mask.new_ones((m, lp)), mask], dim=1),
                position_ids=lp + _positions_from_mask(mask),
                past_key_values=cache,
                cache_position=torch.arange(lp, lp + ids.size(1), device=device),
                use_cache=False,
            )
            rep_pos = torch.tensor([len(r) - 1 for r in part], device=device)
            reps += out.last_hidden_state[torch.arange(m, device=device), rep_pos].unbind()
            del out, cache
        return reps

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
