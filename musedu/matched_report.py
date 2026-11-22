"""Summarize paired validation songs across matched development seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .matched_development import VARIANTS

METRICS = ('valence_within_pearson', 'arousal_within_pearson',
           'valence_ccc', 'arousal_ccc', 'valence_mae', 'arousal_mae',
           'valence_rmse', 'arousal_rmse')
COMPARISONS = (('B2','B1'), ('B4','B3'), ('B8','B7'),
               ('B8','B2'), ('B8','B4'), ('B8','B5'), ('B8','B6'),
               ('B9','B8'), ('B9','B2'))


def report(root, seeds=range(2026, 2031), bootstraps=10000):
    root = Path(root)
    tables = {}
    summary = []
    for variant in VARIANTS:
        frames = []
        for seed in seeds:
            f = root/variant/f'seed{seed}'/'validation_by_song.csv'
            if not f.is_file(): raise FileNotFoundError(f)
            df = pd.read_csv(f, dtype={'song_id': str}).set_index('song_id')
            frames.append(df.loc[:, METRICS].sort_index())
        if any(not x.index.equals(frames[0].index) for x in frames[1:]):
            raise ValueError(f'{variant}: mismatched validation songs')
        tables[variant] = np.stack([x.to_numpy() for x in frames])
        for index, name in enumerate(METRICS):
            seed_means = np.nanmean(tables[variant][:,:,index], axis=1)
            summary.append({'variant': variant, 'metric': name,
                            'mean': float(seed_means.mean()),
                            'seed_sd': float(seed_means.std(ddof=1)),
                            'n_seeds': len(seed_means), 'n_songs': len(frames[0])})
    pd.DataFrame(summary).to_csv(root/'matched_summary.csv', index=False)
    rng = np.random.default_rng(20261026)
    pairs = []
    n_songs = next(iter(tables.values())).shape[1]
    draws = rng.integers(0, n_songs, size=(bootstraps, n_songs))
    for left, right in COMPARISONS:
        diff = tables[left]-tables[right]
        for index, name in enumerate(METRICS):
            by_song = np.nanmean(diff[:,:,index], axis=0)
            finite = np.isfinite(by_song)
            if not finite.any(): continue
            vals = by_song[finite]
            local_draws = draws if finite.all() else rng.integers(0, len(vals), size=(bootstraps,len(vals)))
            boot = vals[local_draws].mean(1)
            pairs.append({'comparison': f'{left}-{right}', 'metric': name,
                          'mean_difference': float(vals.mean()),
                          'bootstrap_low': float(np.quantile(boot, .025)),
                          'bootstrap_high': float(np.quantile(boot, .975)),
                          'seed_difference_sd': float(np.nanmean(diff[:,:,index],axis=1).std(ddof=1)),
                          'n_songs': len(vals)})
    pd.DataFrame(pairs).to_csv(root/'paired_song_bootstrap.csv', index=False)
    (root/'report_protocol.json').write_text(json.dumps({
        'seeds': list(seeds), 'bootstrap_replicates': bootstraps,
        'bootstrap_unit': 'song, after pairing the same song and seed',
        'random_seed': 20261026, 'validation_only': True,
        'caution': 'Exploratory development; repeated selection on this validation split limits inferential interpretation.'}, indent=2))
    return pd.DataFrame(summary), pd.DataFrame(pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    report(args.root)
