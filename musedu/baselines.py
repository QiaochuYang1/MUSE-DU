"""Transparent comparison models for dynamic valence/arousal regression.

These are independently trained *adaptations*, not claimed reconstructions of
published systems. All models predict bounded [-1, 1] trajectories; neither
inter-rater disagreement nor calibration is part of their point-prediction head.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

ACOUSTIC_DIM = 18
NEURAL_METHODS = {
    'mert_linear': 'mert',
    'mert_tcn': 'mert',
    'mert_transformer': 'mert',
    'cnn_bilstm': 'fusion',
    'cnn_bigru': 'fusion',
    'cnn_bigru_hierarchical': 'fusion',
    'tcn': 'fusion',
    'transformer': 'fusion',
    'conformer': 'fusion',
}
CLASSICAL_METHODS = {'ridge': 'acoustic', 'xgboost': 'acoustic', 'random_forest': 'acoustic'}
METHODS = {**CLASSICAL_METHODS, **NEURAL_METHODS}


def choose_view(z, view):
    import numpy as np
    a = np.asarray(z['acoustic'], dtype='float32')
    e = np.asarray(z['embedding'], dtype='float32')
    if a.ndim != 2 or a.shape[1] != ACOUSTIC_DIM or e.ndim != 2 or len(a) != len(e):
        raise ValueError('Expected aligned [time,18] acoustic descriptors and [time,D] embeddings')
    if view in ('mert', 'fusion') and e.shape[1] == 0:
        raise ValueError('MERT-dependent baseline requires --backend mert prepared embeddings (not acoustic-only)')
    if view == 'acoustic': return a
    if view == 'mert': return e
    if view == 'fusion': return np.concatenate((a, e), axis=1)
    raise ValueError(f'Unknown input view: {view}')


class TemporalConv(nn.Module):
    def __init__(self, hidden, layers=3, kernel=3):
        super().__init__()
        blocks = []
        for layer in range(layers):
            dilation = 2**layer
            blocks += [nn.Conv1d(hidden, hidden, kernel, padding=dilation*(kernel-1)//2,
                                 dilation=dilation), nn.GELU()]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Compact noncausal Conformer-style block: macaron FFN, MHSA, GLU depthwise conv.

    This is a *controlled architectural baseline*, not an exact published
    Conformer checkpoint or particular paper's original implementation.
    """
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4*hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(4*hidden, hidden))
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(hidden)
        self.pointwise_in = nn.Conv1d(hidden, 2*hidden, 1)
        self.depthwise = nn.Conv1d(hidden, hidden, 15, groups=hidden, padding=7)
        self.pointwise_out = nn.Conv1d(hidden, hidden, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4*hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(4*hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, valid):
        x = x + .5*self.ff1(x)
        v = self.attn_norm(x)
        a, _ = self.attn(v, v, v, key_padding_mask=~valid, need_weights=False)
        x = x + a
        c = self.conv_norm(x).masked_fill(~valid.unsqueeze(-1), 0).transpose(1, 2)
        c = F.glu(self.pointwise_in(c), dim=1)
        c = self.pointwise_out(F.silu(self.depthwise(c))).transpose(1, 2)
        x = x + c
        x = self.norm(x + .5*self.ff2(x))
        return x.masked_fill(~valid.unsqueeze(-1), 0)


class BaselineNetwork(nn.Module):
    def __init__(self, method, in_features, hidden=96, dropout=.1):
        super().__init__()
        if method not in NEURAL_METHODS: raise ValueError(f'Unknown neural baseline: {method}')
        if hidden <= 0 or hidden % 4: raise ValueError('hidden must be positive and divisible by 4')
        self.method = method
        self.input_dim = in_features
        self.hidden_dim = hidden
        self.proj = (nn.Identity() if method == 'mert_linear' else
                     nn.Sequential(nn.Linear(in_features, hidden), nn.LayerNorm(hidden), nn.GELU()))
        if method == 'mert_linear':
            self.encoder = nn.Identity()
        elif method in ('mert_tcn', 'tcn'):
            self.encoder = TemporalConv(hidden)
        elif method in ('cnn_bilstm', 'cnn_bigru', 'cnn_bigru_hierarchical'):
            self.cnn = TemporalConv(hidden, layers=1, kernel=5)
            cell = nn.LSTM if method == 'cnn_bilstm' else nn.GRU
            self.encoder = cell(hidden, hidden//2, num_layers=1, bidirectional=True, batch_first=True)
        elif method in ('transformer', 'mert_transformer'):
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden*2, dropout,
                                                batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        elif method == 'conformer':
            self.encoder = nn.ModuleList([ConformerBlock(hidden, 4, dropout) for _ in range(2)])
        if method=='cnn_bigru_hierarchical':
            self.global_head=nn.Linear(hidden,2)
            self.local_head=nn.Linear(hidden,2)
        else:self.head = nn.Linear(in_features if method == 'mert_linear' else hidden, 2)

    def forward(self, x, times, valid):
        if x.ndim != 3 or times.shape != x.shape[:2] or valid.shape != x.shape[:2]:
            raise ValueError('Expected x[B,T,F], times[B,T], valid[B,T]')
        h = self.proj(x)
        if self.method in ('transformer', 'mert_transformer', 'conformer'):
            # Absolute annotation time is retained but the encoding is periodic and bounded.
            half = self.hidden_dim // 2
            denom = torch.exp(torch.linspace(0, 8, half, device=x.device, dtype=x.dtype))
            phase = times.unsqueeze(-1) / denom
            h = h + torch.cat((phase.sin(), phase.cos()), -1)
        h = h.masked_fill(~valid.unsqueeze(-1), 0)
        if self.method in ('cnn_bilstm', 'cnn_bigru', 'cnn_bigru_hierarchical'):
            local = self.cnn(h)
            # Padding is only at the right, and valid=True for all actual audio bins.
            lengths = valid.sum(1).to('cpu')
            packed = nn.utils.rnn.pack_padded_sequence(local, lengths, batch_first=True,
                                                        enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True,
                                                     total_length=x.shape[1])
        elif self.method in ('transformer', 'mert_transformer'):
            h = self.encoder(h, src_key_padding_mask=~valid)
        elif self.method == 'conformer':
            for block in self.encoder: h = block(h, valid)
        else: h = self.encoder(h)
        if self.method=='cnn_bigru_hierarchical':
            weight=valid.unsqueeze(-1).to(h.dtype)
            pooled=(h*weight).sum(1)/weight.sum(1).clamp_min(1)
            residual=self.local_head(h)
            residual=(residual-(residual*weight).sum(1,keepdim=True)/
                      weight.sum(1,keepdim=True).clamp_min(1))*weight
            return torch.tanh(self.global_head(pooled)[:,None,:]+residual)*weight
        return torch.tanh(self.head(h)).masked_fill(~valid.unsqueeze(-1), 0)


def frame_regression_loss(pred, target, mask, delta=.1):
    if not mask.any(): return pred.sum()*0
    return F.huber_loss(pred[mask], target[mask], delta=delta)
