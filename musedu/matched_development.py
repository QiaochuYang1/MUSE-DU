"""Matched exploratory DEAM development for temporal shape and architecture controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baselines import BaselineNetwork, choose_view
from .echo_mer import ECHOMER
from .experiment import load_manifest, nominal_step, on_device, pack, read_npz, seed_everything
from .metrics import song_metrics
from .model import (MUSEDU, MUSEDUV2, V2_LOSS_WEIGHTS, _equal_song_huber,
                    centered_correlation_loss, legacy_centered_correlation_loss,
                    supervised_loss, v2_supervised_loss)
from .v2_experiment import LengthBuckets, _reference_split, _within_song_corr


VARIANTS = {
    'B1': ('v1', 'fusion', 'original'),
    'B2': ('v1', 'fusion', 'point_shape'),
    'B3': ('cnn_bigru', 'fusion', 'point'),
    'B4': ('cnn_bigru', 'fusion', 'point_shape'),
    'B5': ('cnn_bigru_hierarchical', 'fusion', 'point_shape'),
    'B6': ('mert_tcn', 'mert', 'point_shape'),
    'B7': ('v2', 'fusion', 'point'),
    'B8': ('v2', 'fusion', 'point_shape'),
    'B9': ('v2', 'fusion', 'full'),
}
ECHO_VARIANTS = {
    'E5': ('echo_constant', 'fusion', 'point_shape'),
    'E6': ('echo_event', 'fusion', 'point_shape'),
    'E4': ('tcn', 'fusion', 'point_shape'),
}
SHAPE_WEIGHT = .02


class Songs:
    def __init__(self, df, ids, norm, view):
        lookup = dict(zip(df.song_id, df.file))
        mu, sd = norm
        self.rows = []
        self.lengths = []
        for sid in ids:
            z = read_npz(lookup[sid])
            x = (choose_view(z, view) - mu) / sd
            self.rows.append({'x': x, 'times': z['times'], 'mean': z['mean'],
                              'std': z['std'], 'mask': z['mask'],
                              'std_mask': z['std_mask'], 'step_seconds': nominal_step(z)})
            self.lengths.append(len(x))

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def train_normalizer(df, ids, view):
    lookup = dict(zip(df.song_id, df.file))
    x = np.concatenate([choose_view(read_npz(lookup[sid]), view) for sid in ids])
    return x.mean(0).astype('float32'), np.maximum(x.std(0), 1e-4).astype('float32')


def make_model(architecture, features, hidden):
    if architecture == 'v1': return MUSEDU(features, hidden=hidden)
    if architecture == 'v2': return MUSEDUV2(features, hidden=hidden)
    if architecture in ('echo_constant', 'echo_event'):
        return ECHOMER(features, hidden=hidden,
                       event_conditioned=architecture == 'echo_event')
    return BaselineNetwork(architecture, features, hidden=hidden)


def output_mean(output):
    return output['mean'] if isinstance(output, dict) else output


def loss_for(output, batch, objective, shape_version='stable'):
    pred = output_mean(output)
    if objective == 'original': return supervised_loss(output, batch)[0]
    if objective == 'full': return v2_supervised_loss(output, batch, V2_LOSS_WEIGHTS)[0]
    point = _equal_song_huber(pred, batch['mean'], batch['mask'])
    if objective == 'point': return point
    if objective == 'point_shape':
        shape=legacy_centered_correlation_loss if shape_version=='legacy' else centered_correlation_loss
        return point + SHAPE_WEIGHT * shape(pred, batch['mean'], batch['mask'])
    raise ValueError(objective)


def selector(output, batch, shape_version='stable'):
    """One fixed checkpoint rule for all variants, regardless of train objective."""
    pred = output_mean(output)
    shape=legacy_centered_correlation_loss if shape_version=='legacy' else centered_correlation_loss
    return (_equal_song_huber(pred, batch['mean'], batch['mask']) +
            SHAPE_WEIGHT * shape(pred, batch['mean'], batch['mask']))


def run_one(manifest, reference_run, out, variant, seed, *, epochs=15, hidden=96,
            batch_size=8, device='cuda', shape_version='stable'):
    if shape_version not in ('stable','legacy'):raise ValueError(shape_version)
    architecture, view, objective = (VARIANTS | ECHO_VARIANTS)[variant]
    df = load_manifest(manifest)
    split = _reference_split(df, reference_run)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(out)
    seed_everything(seed)
    norm = train_normalizer(df, split['train'], view)
    np.savez(out/'normalizer.npz', mean=norm[0], std=norm[1])
    protocol = {'variant': variant, 'architecture': architecture, 'view': view,
                'training_objective': objective, 'shape_weight': SHAPE_WEIGHT,
                'shape_version': shape_version,
                'checkpoint_selection': 'validation equal-song Huber + 0.02 centered-correlation loss',
                'seed': seed, 'epochs': epochs, 'patience': 5, 'hidden': hidden,
                'batch_size': batch_size, 'train_song_ids': split['train'],
                'validation_song_ids': split['val'],
                'excluded_song_counts': {k: len(split[k]) for k in ('scale', 'cal', 'test')},
                'exploratory': True, 'test_set_used': False}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    train = Songs(df, split['train'], norm, view)
    val = Songs(df, split['val'], norm, view)
    tr = DataLoader(train, batch_sampler=LengthBuckets(train.lengths, batch_size, True, seed), collate_fn=pack)
    va = DataLoader(val, batch_sampler=LengthBuckets(val.lengths, batch_size, False, seed), collate_fn=pack)
    model = make_model(architecture, len(norm[0]), hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    best = float('inf'); bad = 0; history = []
    for epoch in range(epochs):
        model.train(); train_losses = []
        for data in tr:
            batch = on_device(data, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model(batch['x'], batch['times'], batch['valid']), batch, objective, shape_version)
            if not torch.isfinite(loss): raise FloatingPointError(f'{variant} nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval(); total = 0.; n = 0
        with torch.inference_mode():
            for data in va:
                batch = on_device(data, device)
                score = selector(model(batch['x'], batch['times'], batch['valid']), batch, shape_version)
                count = len(data['x'])
                total += float(score) * count; n += count
        score = total/n
        history.append({'epoch': epoch+1, 'train_objective': float(np.mean(train_losses)),
                        'validation_selector': score})
        print(f'{variant} seed {seed} epoch {epoch+1}: selector={score:.5f}', flush=True)
        if score < best - 1e-7:
            best = score; bad = 0
            torch.save({'state_dict': model.state_dict(), 'architecture': architecture,
                        'view': view, 'features': len(norm[0]), 'hidden': hidden,
                        'shape_version': shape_version,
                        'variant': variant, 'seed': seed}, out/'model.pt')
        else:
            bad += 1
            if bad >= 5: break
    pd.DataFrame(history).to_csv(out/'training_history.csv', index=False)
    model.load_state_dict(torch.load(out/'model.pt', map_location=device, weights_only=True)['state_dict'])
    model.eval(); lookup = dict(zip(df.song_id, df.file)); rows = []
    with torch.inference_mode():
        for sid in split['val']:
            z = read_npz(lookup[sid]); x = (choose_view(z, view)-norm[0])/norm[1]
            xx = torch.as_tensor(x[None], dtype=torch.float32, device=device)
            tt = torch.as_tensor(z['times'][None], dtype=torch.float32, device=device)
            valid = torch.ones((1, len(x)), dtype=torch.bool, device=device)
            pred = output_mean(model(xx, tt, valid))[0].cpu().numpy()
            corr = _within_song_corr(pred, z['mean'], z['mask'])
            rows.append({'song_id': sid, **song_metrics(pred, z['mean'], z['mask']),
                         'valence_within_pearson': corr[0], 'arousal_within_pearson': corr[1]})
    table = pd.DataFrame(rows)
    table.to_csv(out/'validation_by_song.csv', index=False)
    summary = {'variant': variant, 'seed': seed, 'best_selector': best,
               'selected_epoch': int(np.argmin([x['validation_selector'] for x in history])+1),
               'validation_metrics': {k: float(table[k].mean()) for k in table if k != 'song_id'},
               'exploratory': True, 'test_set_used': False}
    (out/'validation_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS | ECHO_VARIANTS,
                        default=['B2','B4','B8','B9'])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(2026,2031)))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shape-version',choices=('stable','legacy'),default='stable')
    args = parser.parse_args()
    root = Path(args.out)
    for variant in args.variants:
        for seed in args.seeds:
            path = root/variant/f'seed{seed}'
            if (path/'validation_summary.json').is_file():
                print(f'skip complete {path}', flush=True)
                continue
            run_one(args.manifest, args.reference_run, path, variant, seed,
                    epochs=args.epochs, device=args.device,shape_version=args.shape_version)


if __name__ == '__main__': main()
