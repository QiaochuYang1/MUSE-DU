"""Paired song bootstrap on saved DEAM five-fold outer predictions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METRICS=('valence_mae','arousal_mae','valence_rmse','arousal_rmse',
         'valence_ccc','arousal_ccc','valence_within_pearson',
         'arousal_within_pearson','med')


def read(path):
    with open(path,newline='',encoding='utf-8') as handle:
        return {r['song_id']:r for r in csv.DictReader(handle)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-root',required=True)
    parser.add_argument('--adapted-root',required=True)
    parser.add_argument('--draws',type=int,default=10000)
    args=parser.parse_args()
    root=Path(args.cv_root)
    tables={method:read(root/method/'out_of_fold_by_song.csv')
            for method in ('B2','B2_point','opensmile_lstm')}
    tables['opensmile_librosa_ridge_adapted']=read(Path(args.adapted_root)/'out_of_fold_by_song.csv')
    ids=sorted(tables['B2'])
    if len(ids)!=1802 or any(set(t)!=set(ids) for t in tables.values()):
        raise ValueError('Paired methods do not cover the identical 1,802 songs')
    rng=np.random.default_rng(2026)
    samples=rng.integers(0,len(ids),size=(args.draws,len(ids)),dtype=np.int32)
    output={'unit':'song','draws':args.draws,'seed':2026,'songs':len(ids),
            'difference':'B2 minus comparator','comparisons':{}}
    for competitor in ('B2_point','opensmile_librosa_ridge_adapted','opensmile_lstm'):
        results={}
        for metric in METRICS:
            a=np.array([float(tables['B2'][sid][metric]) for sid in ids])
            b=np.array([float(tables[competitor][sid][metric]) for sid in ids])
            difference=a-b
            draws=np.nanmean(difference[samples],axis=1)
            results[metric]={'B2':float(np.nanmean(a)),
                             'comparator':float(np.nanmean(b)),
                             'difference':float(np.nanmean(difference)),
                             'ci95':[float(x) for x in np.quantile(draws,[.025,.975])],
                             'bootstrap_sign_probability':float(np.mean(draws>0))}
        output['comparisons'][competitor]=results
    (root/'paired_song_bootstrap.json').write_text(json.dumps(output,indent=2))
    for competitor,item in output['comparisons'].items():
        print(competitor,'MED',item['med'],'Pearson-V',item['valence_within_pearson'],
              'Pearson-A',item['arousal_within_pearson'],flush=True)


if __name__=='__main__':main()
