"""Paired development-song comparison for initial ECHO-MER mechanism test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ('valence_within_pearson','arousal_within_pearson',
           'valence_ccc','arousal_ccc','valence_mae','arousal_mae',
           'valence_rmse','arousal_rmse')


def report(echo_root, baseline_root, seeds=range(2026,2031), bootstraps=10000):
    echo_root = Path(echo_root); baseline_root = Path(baseline_root)
    tables = {}
    for variant in ('E4','E5','E6','B2','B4','B8'):
        root = echo_root if variant.startswith('E') else baseline_root
        frames = [pd.read_csv(root/variant/f'seed{s}'/'validation_by_song.csv',
                              dtype={'song_id':str}).set_index('song_id').loc[:,METRICS].sort_index()
                  for s in seeds]
        if any(not frame.index.equals(frames[0].index) for frame in frames[1:]):
            raise ValueError(f'{variant}: mismatched song IDs')
        tables[variant] = np.stack([frame.to_numpy() for frame in frames])
    means=[]
    for variant,array in tables.items():
        for index,metric in enumerate(METRICS):
            seed_means = np.nanmean(array[:,:,index],axis=1)
            means.append({'variant':variant,'metric':metric,'mean':float(seed_means.mean()),
                          'seed_sd':float(seed_means.std(ddof=1))})
    pd.DataFrame(means).to_csv(echo_root/'comparison_summary.csv',index=False)
    pairs=[];rng=np.random.default_rng(20261027)
    for left,right in (('E6','E5'),('E6','E4'),('E6','B2'),('E6','B4'),('E6','B8')):
        for index,metric in enumerate(METRICS):
            diff = np.nanmean(tables[left][:,:,index]-tables[right][:,:,index],axis=0)
            diff = diff[np.isfinite(diff)]
            draws=rng.integers(0,len(diff),size=(bootstraps,len(diff)))
            boot=diff[draws].mean(1)
            pairs.append({'comparison':f'{left}-{right}','metric':metric,
                          'mean_difference':float(diff.mean()),
                          'bootstrap_low':float(np.quantile(boot,.025)),
                          'bootstrap_high':float(np.quantile(boot,.975)),
                          'n_songs':len(diff)})
    pd.DataFrame(pairs).to_csv(echo_root/'paired_song_bootstrap.csv',index=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--echo-root',required=True)
    parser.add_argument('--baseline-root',required=True)
    args=parser.parse_args()
    report(args.echo_root,args.baseline_root)
