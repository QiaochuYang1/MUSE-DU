"""Minimal event-conditioned affective state model and fixed-gate control.

Both variants share the same encoder, persistence, candidate, and innovation
networks. The control replaces the time-varying event gate with one learned
constant gate per emotion axis. No disagreement or uncertainty head is included.
"""
from __future__ import annotations

import torch
from torch import nn


class ECHOMER(nn.Module):
    def __init__(self, in_features: int, hidden: int = 96,
                 event_conditioned: bool = True, lags=(1, 4, 10, 20),long_valence=False):
        super().__init__()
        self.event_conditioned = event_conditioned
        self.lags = tuple(lags)
        self.long_valence=long_valence
        self.projection = nn.Sequential(nn.Linear(in_features, hidden),
                                        nn.LayerNorm(hidden), nn.GELU())
        self.context_first = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.context_second = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.activation = nn.GELU()
        self.persistence = nn.Linear(hidden, 2)
        self.candidate = nn.Linear(hidden, 2)
        self.innovation = nn.Linear(hidden, 2)
        self.event_gate = nn.Sequential(nn.Linear(hidden * (len(lags)+1), hidden),
                                        nn.GELU(), nn.Linear(hidden, 2))
        self.constant_gate_logit = nn.Parameter(torch.zeros(2))
        self.initial_state = nn.Linear(hidden, 2)
        if long_valence:
            self.valence_context_first=nn.Conv1d(hidden,hidden,5,padding=8,dilation=4)
            self.valence_context_second=nn.Conv1d(hidden,hidden,3,padding=10,dilation=10)
            with torch.no_grad():self.persistence.bias[0]=2.2

    def forward(self, x, times, valid):
        if x.ndim != 3 or times.shape != x.shape[:2] or valid.shape != x.shape[:2]:
            raise ValueError('Expected x[B,T,F], times[B,T], valid[B,T]')
        z = self.projection(x).masked_fill(~valid.unsqueeze(-1), 0)
        h = self.activation(self.context_first(z.transpose(1, 2))).transpose(1, 2)
        h = h.masked_fill(~valid.unsqueeze(-1), 0)
        hv=h
        if self.long_valence:
            hv=self.activation(self.valence_context_first(h.transpose(1,2))).transpose(1,2)
            hv=hv.masked_fill(~valid.unsqueeze(-1),0)
            hv=self.activation(self.valence_context_second(hv.transpose(1,2))).transpose(1,2)
            hv=hv.masked_fill(~valid.unsqueeze(-1),0)
        h = self.activation(self.context_second(h.transpose(1, 2))).transpose(1, 2)
        h = h.masked_fill(~valid.unsqueeze(-1), 0)
        # Exact, uninterrupted lags avoid treating padding or timestamp gaps as events.
        events = []
        dt = times[:, 1:] - times[:, :-1]
        expected = torch.where((dt > 0) & valid[:, 1:] & valid[:, :-1], dt,
                               torch.full_like(dt, float('inf'))).amin(1, keepdim=True)
        expected = torch.where(torch.isfinite(expected), expected, torch.ones_like(expected))
        adjacent = torch.isclose(dt, expected, rtol=.01, atol=1e-4)
        for lag in self.lags:
            change = torch.zeros_like(h)
            if lag < x.shape[1]:
                good = valid[:, lag:] & valid[:, :-lag]
                for offset in range(lag):
                    good = good & adjacent[:, offset:offset+x.shape[1]-lag]
                change[:, lag:] = (h[:, lag:] - h[:, :-lag]) * good.unsqueeze(-1)
            events.append(change)
        def axis_context(module):
            value=module(h)
            if self.long_valence:value=torch.stack((module(hv)[:,:,0],value[:,:,1]),dim=-1)
            return value
        rho = torch.sigmoid(axis_context(self.persistence))
        candidate = axis_context(self.candidate)
        delta = torch.tanh(axis_context(self.innovation))
        if self.event_conditioned:
            gate = torch.sigmoid(self.event_gate(torch.cat((h, *events), dim=-1)))
            if self.long_valence:
                val_events=[]
                for lag in self.lags:
                    change=torch.zeros_like(hv)
                    if lag<x.shape[1]:
                        good=valid[:,lag:]&valid[:,:-lag]
                        for offset in range(lag):good=good&adjacent[:,offset:offset+x.shape[1]-lag]
                        change[:,lag:]=(hv[:,lag:]-hv[:,:-lag])*good.unsqueeze(-1)
                    val_events.append(change)
                vg=torch.sigmoid(self.event_gate(torch.cat((hv,*val_events),dim=-1)))[:,:,0]
                gate=torch.stack((vg,gate[:,:,1]),dim=-1)
        else:
            gate = torch.sigmoid(self.constant_gate_logit).view(1, 1, 2).expand_as(rho)
        state = self.initial_state(h[:, 0])
        if self.long_valence:
            state=torch.stack((self.initial_state(hv[:,0])[:,0],state[:,1]),dim=-1)
        trajectory = []
        for t in range(x.shape[1]):
            updated = rho[:, t] * state + (1-rho[:, t]) * candidate[:, t] + gate[:, t] * delta[:, t]
            state = torch.where(valid[:, t, None], updated, state)
            trajectory.append(torch.tanh(state) * valid[:, t, None])
        mean = torch.stack(trajectory, dim=1)
        return {'mean': mean, 'gate': gate.masked_fill(~valid.unsqueeze(-1), 0),
                'persistence': rho.masked_fill(~valid.unsqueeze(-1), 0)}
