"""The relational judge: the only place candidates meet.

Each candidate's representation c_i is read into P(a_i) by a per-candidate
encoder. A candidate's score is then built only from comparisons with the
others,

    d_ij  = [P(a_i) - P(a_j), P(a_i) * P(a_j), cos(P(a_i), P(a_j))]
    phi   = psi(d_ij) - psi(d_ji)                 (antisymmetric)
    s_i   = mean_{j != i} phi(a_i, a_j)

so there is no positional encoding anywhere (shuffling the labels shuffles
the scores) and sum_i s_i = 0: the selector says which candidate wins, never
whether any is good. That second question is the NONE gate's: a separate
head that reads the same set and returns the logit of P(a valid candidate
exists). It never feeds the scores.

Parameter names match the training checkpoints and must not change.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .config import JudgeConfig

GATE_INPUTS = ("pair", "selected", "mean", "contrast")

_PAIR_FEATURE_DIMS = {"diff": lambda d: d, "prod": lambda d: d, "cos": lambda d: 1}


class RelationalParts(NamedTuple):
    scores: torch.Tensor  # [B, K], -inf at padded slots
    z: torch.Tensor  # [B, K, d_j] P(a_i)
    gate_logit: torch.Tensor | None  # [B] logit of P(a valid candidate exists)


def _feedforward_trunk(dim: int, ffn_mult: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers += [
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        ]
    return nn.Sequential(*layers)


def pair_feature_dim(features: tuple[str, ...], dim: int) -> int:
    return sum(_PAIR_FEATURE_DIMS[f](dim) for f in features)


def pair_features(p: torch.Tensor, features: tuple[str, ...]) -> torch.Tensor:
    """Relational features of every ordered pair. [B, K, d] -> [B, K, K, F]

    Built so that swapping a pair negates `diff` exactly and leaves the
    symmetric features bitwise equal, which makes phi exactly antisymmetric.
    """
    b, k, d = p.shape
    left = p.unsqueeze(2).expand(b, k, k, d)
    right = p.unsqueeze(1).expand(b, k, k, d)
    parts: list[torch.Tensor] = []
    for name in features:
        if name == "diff":
            parts.append(left - right)
        elif name == "prod":
            parts.append(left * right)
        elif name == "cos":
            q = torch.nn.functional.normalize(p, dim=-1, eps=1e-6)
            # elementwise rather than a matmul, so [i, j] and [j, i] reduce
            # in the same order
            parts.append((q.unsqueeze(2) * q.unsqueeze(1)).sum(-1, keepdim=True))
    return torch.cat(parts, dim=-1)


class RelationalJudge(nn.Module):
    def __init__(self, input_dim: int, config: JudgeConfig) -> None:
        super().__init__()
        self.config = config
        self.features = tuple(config.relational_features)

        self.project = nn.Linear(input_dim, config.dim)
        self.encoder = _feedforward_trunk(config.dim, config.ffn_mult, config.local_layers)
        self.encoder_norm = nn.LayerNorm(config.dim)

        hidden = config.dim * config.ffn_mult
        in_dim = pair_feature_dim(self.features, config.dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(config.dropout)]
        for _ in range(max(0, config.n_layers - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(config.dropout)]
        layers.append(nn.Linear(hidden, 1))
        self.psi = nn.Sequential(*layers)

        self.gate_pair: nn.Sequential | None = None
        self.gate_head: nn.Sequential | None = None
        self.gate_inputs = tuple(i for i in GATE_INPUTS if i in config.gate_inputs)
        if config.none_gate:
            if "pair" in self.gate_inputs:
                self.gate_pair = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(config.dropout)
                )
            widths = {"pair": hidden, "selected": config.dim, "mean": config.dim,
                      "contrast": 4 * config.dim}
            self.gate_head = nn.Sequential(
                nn.Linear(sum(widths[i] for i in self.gate_inputs), config.dim),
                nn.GELU(),
                nn.Linear(config.dim, 1),
            )

    @property
    def has_gate(self) -> bool:
        return self.gate_head is not None

    def encode(self, c: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        p = self.project(c)
        p = p.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.encoder_norm(p + self.encoder(p))

    def _gate(
        self, p: torch.Tensor, feats: torch.Tensor, scores: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if "pair" in self.gate_inputs:
            h = self.gate_pair(feats)
            u = h + h.transpose(1, 2)
            k = candidate_mask.size(1)
            eye = torch.eye(k, dtype=torch.bool, device=p.device)
            pairs = candidate_mask.unsqueeze(1) & candidate_mask.unsqueeze(2) & ~eye
            n_pairs = pairs.sum(dim=(1, 2)).clamp_min(1).unsqueeze(-1).to(u.dtype)
            parts.append((u * pairs.unsqueeze(-1)).sum(dim=(1, 2)) / n_pairs)

        selection = torch.softmax(scores, dim=-1).unsqueeze(-1)
        p_sel = (selection * p).sum(dim=1)
        if "selected" in self.gate_inputs:
            parts.append(p_sel)

        real = candidate_mask.unsqueeze(-1).to(p.dtype)
        if "mean" in self.gate_inputs:
            parts.append((p * real).sum(dim=1) / real.sum(dim=1).clamp_min(1))

        if "contrast" in self.gate_inputs:
            n_rivals = (real.sum(dim=1, keepdim=True) - 1).clamp_min(1)
            others = (1.0 - selection) * real / n_rivals
            p_oth = (others * p).sum(dim=1)
            parts += [p_sel, p_oth, p_sel - p_oth, p_sel * p_oth]

        return self.gate_head(torch.cat(parts, dim=-1)).squeeze(-1)

    def forward(self, c: torch.Tensor, candidate_mask: torch.Tensor) -> RelationalParts:
        """c: [B, K, d_model]; candidate_mask: [B, K] bool, True where real."""
        padding_mask = ~candidate_mask
        p = self.encode(c, padding_mask)

        feats = pair_features(p, self.features)
        psi = self.psi(feats).squeeze(-1)  # [B, K, K]
        phi = psi - psi.transpose(1, 2)
        pair_mask = candidate_mask.unsqueeze(1) & candidate_mask.unsqueeze(2)
        phi = phi.masked_fill(~pair_mask, 0.0)

        r = phi.sum(dim=-1)
        if self.config.relational_aggregate == "mean":
            r = r / (candidate_mask.sum(dim=-1, keepdim=True) - 1).clamp_min(1)
        scores = r.masked_fill(padding_mask, float("-inf"))

        gate_logit = self._gate(p, feats, scores, candidate_mask) if self.has_gate else None
        return RelationalParts(scores=scores, z=p, gate_logit=gate_logit)
