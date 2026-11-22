"""TRACE-MER structured predictor and losses.

Counterfactual losses operate on paired factual/intervention outputs. Audio
interventions and their selectivity acceptance masks are deliberately supplied
by the caller; this module never assumes that a transformation is selective.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import (_equal_song_huber, _masked_song_means,
                    centered_correlation_loss, masked_huber,
                    multiscale_change_loss, center_valid)


# Disjoint columns in the current 75-dimensional interpretable cache.
DEFAULT_GROUPS = {
    "rhythm": (5,),                           # onset strength
    "harmony": tuple(range(6, 75)),           # chroma + tonal descriptors
    "spectrum": (1, 2, 3, 4),                # centroid/bandwidth/rolloff/flux
    "energy": (0,),                           # RMS
}


class MaskedTemporalBlock(nn.Module):
    def __init__(self, features, hidden, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(features, hidden, 5, padding=2*dilation,
                              dilation=dilation)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, valid):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return F.gelu(self.norm(y)).masked_fill(~valid.unsqueeze(-1), 0)


class TRACEMER(nn.Module):
    """Structured cue decomposition with local and long temporal context."""
    def __init__(self, embedding_dim=768, acoustic_dim=75, hidden=96,
                 heads=4, groups=None, cue_decomposition=True):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.acoustic_dim = acoustic_dim
        self.embedding_dim = embedding_dim
        self.cue_decomposition = cue_decomposition
        groups = DEFAULT_GROUPS if groups is None else groups
        self.group_names = tuple(groups)
        self.group_indices = {k: tuple(v) for k, v in groups.items()}
        if any(not ix or min(ix) < 0 or max(ix) >= acoustic_dim
               for ix in self.group_indices.values()):
            raise ValueError("cue group index outside acoustic input")

        self.embedding_projection = nn.Linear(embedding_dim, hidden, bias=False)
        self.group_projection = nn.ModuleDict({
            k: nn.Linear(len(ix), hidden, bias=False)
            for k, ix in self.group_indices.items()})
        self.shared_bias = nn.Parameter(torch.zeros(hidden))
        self.shared_norm = nn.LayerNorm(hidden)
        self.local = MaskedTemporalBlock(hidden, hidden, dilation=1)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden*2, .1,
                                            batch_first=True, activation="gelu")
        self.context = nn.TransformerEncoder(layer, num_layers=2,
                                             enable_nested_tensor=False)
        self.fusion = nn.Sequential(nn.Linear(hidden*2, hidden),
                                    nn.LayerNorm(hidden), nn.GELU())
        self.baseline_head = nn.Linear(hidden, 2)
        self.residual_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                           nn.Linear(hidden, 2))
        self.cue_encoders = nn.ModuleDict({
            k: nn.ModuleList((MaskedTemporalBlock(len(ix), hidden),
                             MaskedTemporalBlock(hidden, hidden, dilation=4)))
            for k, ix in self.group_indices.items()}) if cue_decomposition else nn.ModuleDict()
        self.cue_heads = nn.ModuleDict({k: nn.Linear(hidden, 2)
                                       for k in self.group_names}) if cue_decomposition else nn.ModuleDict()
        self.disagreement_head = nn.Linear(hidden, 2)

    def forward(self, x, times, valid):
        if x.ndim != 3 or x.shape[-1] != self.acoustic_dim+self.embedding_dim:
            raise ValueError("Expected normalized acoustic descriptors followed by MERT")
        if times.shape != x.shape[:2] or valid.shape != x.shape[:2]:
            raise ValueError("Invalid time or padding mask")
        acoustic, embedding = x.split((self.acoustic_dim, self.embedding_dim), -1)
        z = self.embedding_projection(embedding)
        for name, ix in self.group_indices.items():
            z = z + self.group_projection[name](acoustic[..., list(ix)])
        z = self.shared_norm(z+self.shared_bias).masked_fill(~valid.unsqueeze(-1), 0)
        local = self.local(z, valid)
        context = self.context(z, src_key_padding_mask=~valid)
        context = context.masked_fill(~valid.unsqueeze(-1), 0)
        hidden = self.fusion(torch.cat((local, context), -1))
        hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0)
        weight = valid.unsqueeze(-1).to(x.dtype)
        pooled = (hidden*weight).sum(1)/weight.sum(1).clamp_min(1)
        baseline = self.baseline_head(pooled)
        residual = center_valid(self.residual_head(hidden), valid)
        cues = {}
        for name, ix in self.group_indices.items() if self.cue_decomposition else ():
            q = acoustic[..., list(ix)].masked_fill(~valid.unsqueeze(-1), 0)
            first, second = self.cue_encoders[name]
            u = second(first(q, valid), valid)
            cues[name] = center_valid(self.cue_heads[name](u), valid)
        cue_sum = (torch.stack(tuple(cues.values())).sum(0) if cues
                   else torch.zeros_like(residual))
        logits = (baseline[:, None] + cue_sum + residual)*weight
        cue_logits = (baseline[:, None] + cue_sum)*weight
        return {"mean": torch.tanh(logits)*weight,
                "cue_mean": torch.tanh(cue_logits)*weight,
                "logits": logits, "baseline": baseline,
                "song_mean": torch.tanh(baseline), "cues": cues,
                "residual": residual, "hidden": hidden,
                "tau": (F.softplus(self.disagreement_head(hidden))+1e-4)*weight}


TRACE_WEIGHTS = {"point": 1., "shape": .02, "multi": .3, "song": .3,
                 "cue": .3, "residual": .01, "orthogonal": .01,
                 "disagreement": .2}


def _orthogonality_loss(residual, cues, valid, eps=1e-6):
    if not cues:
        return residual.sum()*0
    terms = []
    mask = valid.unsqueeze(-1).expand_as(residual)
    for cue in cues.values():
        r = center_valid(residual, valid)
        c = center_valid(cue, valid)
        numerator = (r*c*mask).sum(1)
        denom = ((r.square()*mask).sum(1).clamp_min(eps).sqrt() *
                 (c.square()*mask).sum(1).clamp_min(eps).sqrt())
        terms.append((numerator/denom).square().mean())
    return torch.stack(terms).mean()


def trace_supervised_loss(out, batch, weights=None, eps=1e-3):
    weights = TRACE_WEIGHTS if weights is None else weights
    if set(weights) != set(TRACE_WEIGHTS):
        raise ValueError("TRACE factual loss weights are incomplete")
    target, mask = batch["mean"], batch["mask"].bool()
    components = {
        "point": _equal_song_huber(out["mean"], target, mask),
        "shape": centered_correlation_loss(out["mean"], target, mask),
        "multi": multiscale_change_loss(out["mean"], target, mask,
                                         batch["times"], batch["step"]),
        "song": masked_huber(out["song_mean"], _masked_song_means(target, mask),
                              mask.any(1)),
        "cue": _equal_song_huber(out["cue_mean"], target, mask),
        "residual": (out["residual"].square()*batch["valid"].unsqueeze(-1)).sum() /
                    batch["valid"].sum().clamp_min(1),
        "orthogonal": _orthogonality_loss(out["residual"], out["cues"],
                                           batch["valid"]),
        "disagreement": _equal_song_huber(
            torch.log(out["tau"]+eps),
            torch.log(batch["std"].clamp_min(eps)+eps),
            batch["std_mask"].bool() & mask),
    }
    total = sum(float(weights[k])*v for k, v in components.items())
    return total, {k: float(v.detach()) for k, v in components.items()}


def trace_variant_weights(variant):
    """Prespecified matched T1-T3 factual objectives."""
    weights = dict(TRACE_WEIGHTS)
    if variant == "T1_no_cue_sufficiency":
        weights["cue"] = 0.
    elif variant == "T2_no_cues":
        weights["cue"] = weights["residual"] = weights["orthogonal"] = 0.
    elif variant != "T3_cue_sufficiency":
        raise ValueError(variant)
    return weights


def trace_counterfactual_loss(factual, altered, target_group, region_mask,
                              accepted, leak_weight=1.):
    """Equation 28 plus branchwise leakage, only for accepted interventions."""
    if target_group not in factual["cues"]:
        raise KeyError(target_group)
    use = region_mask.bool() & accepted.bool()[:, None]
    if not use.any():
        zero = factual["logits"].sum()*0
        return zero, {"counterfactual": 0., "leakage": 0., "accepted": 0}
    delta_logits = altered["logits"]-factual["logits"]
    delta_cue = altered["cues"][target_group]-factual["cues"][target_group]
    mask = use.unsqueeze(-1).expand_as(delta_logits)
    cf = F.huber_loss(delta_logits[mask], delta_cue[mask], delta=.1)
    baseline = (altered["baseline"]-factual["baseline"]).square().sum(-1)
    leak = baseline[accepted.bool()].mean()
    leak = leak + ((altered["residual"]-factual["residual"])[mask]).square().mean()
    for name in factual["cues"]:
        if name != target_group:
            leak = leak + ((altered["cues"][name]-factual["cues"][name])[mask]).square().mean()
    total = cf + leak_weight*leak
    return total, {"counterfactual": float(cf.detach()),
                   "leakage": float(leak.detach()),
                   "accepted": int(accepted.sum())}
