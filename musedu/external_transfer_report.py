"""Summarize frozen external results with paired song bootstrap intervals."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

METRICS=('valence_mae','arousal_mae','valence_rmse','arousal_rmse',
         'valence_ccc','arousal_ccc','valence_within_pearson',
         'arousal_within_pearson','med')


def read(path,dataset):
    with open(path,newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    prefix=dataset+'_'
    return {r['song_id'][len(prefix):] if r['song_id'].startswith(prefix) else r['song_id']:r for r in rows}


def number(value):
    return float(value) if value not in ('',None) else float('nan')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=('pmemo','merp'))
    p.add_argument('--b2-root',required=True)
    p.add_argument('--baseline-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--draws',type=int,default=10000)
    a=p.parse_args()
    files={
        'B2_temporal_shape':Path(a.b2_root)/'B2_temporal_shape'/'validation_by_song.csv',
        'B2_point_only':Path(a.b2_root)/'B2_point_only'/'validation_by_song.csv',
        'opensmile_lstm':Path(a.baseline_root)/a.dataset/'opensmile_lstm'/'validation_by_song.csv',
        'opensmile_librosa_ridge_adapted':Path(a.baseline_root)/a.dataset/'opensmile_librosa_ridge_adapted'/'validation_by_song.csv'}
    tables={name:read(path,a.dataset) for name,path in files.items()}
    ids=sorted(tables['B2_temporal_shape'])
    if any(set(table)!=set(ids) for table in tables.values()):
        raise ValueError('External comparison does not contain identical songs')
    rng=np.random.default_rng(2026)
    samples=rng.integers(0,len(ids),size=(a.draws,len(ids)),dtype=np.int32)
    results={'dataset':a.dataset,'songs':len(ids),'bootstrap_unit':'song',
             'bootstrap_draws':a.draws,'bootstrap_seed':2026,
             'difference':'B2 temporal-shape minus comparator',
             'methods':{},'paired_differences':{}}
    for name,table in tables.items():
        results['methods'][name]={metric:float(np.nanmean([number(table[sid][metric]) for sid in ids]))
                                  for metric in METRICS}
    for name in tables:
        if name=='B2_temporal_shape':continue
        paired={}
        for metric in METRICS:
            x=np.array([number(tables['B2_temporal_shape'][sid][metric]) for sid in ids])
            y=np.array([number(tables[name][sid][metric]) for sid in ids])
            difference=x-y
            draws=np.nanmean(difference[samples],axis=1)
            paired[metric]={'difference':float(np.nanmean(difference)),
                            'ci95':[float(v) for v in np.quantile(draws,[.025,.975])]}
        results['paired_differences'][name]=paired
    Path(a.out).write_text(json.dumps(results,indent=2))
    print(f'{a.dataset}: {len(ids)} paired songs',flush=True)
    for name,metrics in results['methods'].items():
        print(f'{name} MED={metrics["med"]:.6f} V-Pearson={metrics["valence_within_pearson"]:.6f} A-Pearson={metrics["arousal_within_pearson"]:.6f}',flush=True)


if __name__=='__main__':main()
