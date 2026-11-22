"""Temporal validation metrics for the matched Stage 1 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import choose_view
from .experiment import load_manifest, nominal_step, read_npz
from .matched_development import VARIANTS, ECHO_VARIANTS, make_model, output_mean
from .metrics import event_f1_from_counts, song_metrics, transition_metrics
from .v2_experiment import _reference_split, _within_song_corr


def thresholds(df, ids):
    lookup = dict(zip(df.song_id, df.file)); values = [[], []]
    for sid in ids:
        z = read_npz(lookup[sid]); t = z['times']; m = z['mask']; y = z['mean']
        adjacent = np.isclose(np.diff(t), nominal_step(z), rtol=.01, atol=1e-4)
        for axis in range(2):
            keep = adjacent & m[1:,axis] & m[:-1,axis]
            values[axis].append(np.abs(np.diff(y[:,axis])[keep]))
    return [float(np.quantile(np.concatenate(axis_values), .9)) for axis_values in values]


def run(manifest, reference_run, root, device='cuda', variants=None):
    df = load_manifest(manifest)
    split = _reference_split(df, reference_run)
    threshold = thresholds(df, split['train'])
    root = Path(root)
    lookup = dict(zip(df.song_id, df.file)); summary_rows = []
    (root/'temporal_thresholds.json').write_text(json.dumps({
        'train_absolute_first_difference_90th_percentile': threshold,
        'prediction_thresholds': threshold, 'evaluation': 'validation only'}, indent=2))
    for variant in (VARIANTS if variants is None else variants):
        for seed in range(2026,2031):
            path = root/variant/f'seed{seed}'
            state = torch.load(path/'model.pt', map_location=device, weights_only=True)
            model = make_model(state['architecture'],state['features'],state['hidden']).to(device)
            model.load_state_dict(state['state_dict']); model.eval()
            with np.load(path/'normalizer.npz') as z: norm = z['mean'], z['std']
            rows = []; pooled_pred = [[],[]]; pooled_truth = [[],[]]
            with torch.inference_mode():
                for sid in split['val']:
                    z = read_npz(lookup[sid])
                    x = (choose_view(z, state['view'])-norm[0])/norm[1]
                    xx = torch.as_tensor(x[None], dtype=torch.float32, device=device)
                    tt = torch.as_tensor(z['times'][None], dtype=torch.float32, device=device)
                    valid = torch.ones((1,len(x)), dtype=torch.bool, device=device)
                    pred = output_mean(model(xx,tt,valid))[0].cpu().numpy()
                    metric = song_metrics(pred,z['mean'],z['mask'])
                    corr = _within_song_corr(pred,z['mean'],z['mask'])
                    trans = transition_metrics(pred,z['mean'],z['mask'],z['times'],
                                               nominal_step(z),threshold,threshold)
                    rows.append({'song_id':sid,**metric,
                                 'valence_within_pearson':corr[0],
                                 'arousal_within_pearson':corr[1],**trans})
                    for axis in range(2):
                        m = z['mask'][:,axis]
                        pooled_pred[axis].append(pred[m,axis]); pooled_truth[axis].append(z['mean'][m,axis])
            table = pd.DataFrame(rows)
            table.to_csv(path/'validation_temporal_by_song.csv',index=False)
            summary = {'variant':variant,'seed':seed}
            for axis,name in enumerate(('valence','arousal')):
                p = np.concatenate(pooled_pred[axis]); y = np.concatenate(pooled_truth[axis])
                mu_p,mu_y = p.mean(),y.mean()
                cov = np.mean((p-mu_p)*(y-mu_y))
                summary[f'{name}_pooled_ccc'] = float(2*cov/(p.var()+y.var()+(mu_p-mu_y)**2))
                for key in ('ccc','within_pearson','mae','rmse','first_difference_mae'):
                    summary[f'{name}_{key}'] = float(table[f'{name}_{key}'].mean())
                tp,fp,fn = (int(table[f'{name}_event_{item}'].sum()) for item in ('tp','fp','fn'))
                summary[f'{name}_transition_f1'] = event_f1_from_counts(tp,fp,fn)
            summary_rows.append(summary)
            print(f'temporal {variant} seed {seed}',flush=True)
    pd.DataFrame(summary_rows).to_csv(root/'temporal_summary_by_seed.csv',index=False)
    return pd.DataFrame(summary_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--reference-run',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS | ECHO_VARIANTS)
    args = parser.parse_args()
    run(args.manifest,args.reference_run,args.root,args.device,args.variants)
