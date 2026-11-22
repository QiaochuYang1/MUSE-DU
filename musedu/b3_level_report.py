"""Recompute B3-C pilot metrics and song-paired bootstrap intervals from predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .published_baselines.io import load_index,evaluate

METRICS=('med','valence_within_pearson','arousal_within_pearson')

def read_predictions(folder,ids):
    files=list(folder.glob('*.npz'))
    if {f.stem for f in files}!=set(ids):raise AssertionError(f'Incomplete predictions: {folder}')
    pp={}
    for f in files:
        with np.load(f,allow_pickle=False) as z:pp[f.stem]=z['pred']
    return pp

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--matrix-root',required=True);p.add_argument('--mert-root',required=True)
    p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root);matrix=Path(a.matrix_root);mert=Path(a.mert_root)
    index,_=load_index(a.manifest,a.reference_run)
    splits=json.loads((matrix/'frozen_splits.json').read_text())['folds']
    rows=[];song_rows=[];bootstrap=[];rng=np.random.default_rng(2026)
    for repeat in range(3):
        pp={name:{} for name in ('B3_C','B2_shape_shape','MERT_330M_adapted')}
        for split in splits:
            fold=split['fold'];ids=split['outer_ids']
            folders={
                'B3_C':root/f'repeat_{repeat}'/f'fold_{fold}'/'predictions',
                'B2_shape_shape':matrix/f'repeat_{repeat}'/'train_shape'/f'fold_{fold}'/'select_shape'/'predictions',
                'MERT_330M_adapted':mert/f'fold_{fold}'/'predictions'}
            for name,folder in folders.items():
                fold_pp=read_predictions(folder,ids)
                if set(pp[name])&set(fold_pp):raise AssertionError('Duplicate song')
                pp[name].update(fold_pp)
        tables={}
        for name,pred in pp.items():
            if set(pred)!=set(index):raise AssertionError('Missing full OOF set')
            by_song,metric=evaluate(pred,index,sorted(index))
            by_song=by_song.set_index('song_id');tables[name]=by_song
            rows.append({'repeat':repeat,'model':name,'songs':1802,**metric})
            for sid,row in by_song.iterrows():
                song_rows.append({'repeat':repeat,'model':name,'song_id':sid,**row.to_dict()})
        for other in ('B2_shape_shape','MERT_330M_adapted'):
            for metric in METRICS:
                ids=sorted(index)
                delta=(tables['B3_C'].loc[ids,metric]-tables[other].loc[ids,metric]).to_numpy(dtype=float)
                draws=rng.integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
                sample=np.nanmean(delta[draws],axis=1)
                bootstrap.append({'repeat':repeat,'contrast':'B3_C minus '+other,'metric':metric,
                                  'difference':float(np.nanmean(delta)),
                                  'ci95_lower':float(np.quantile(sample,.025)),
                                  'ci95_upper':float(np.quantile(sample,.975)),
                                  'bootstrap_draws':10000,'bootstrap_unit':'song'})
        for axis in ('valence','arousal'):
            a_col=tables['B3_C'][axis+'_within_pearson'].to_numpy()
            b_col=tables['B2_shape_shape'][axis+'_within_pearson'].to_numpy()
            if not np.allclose(a_col,b_col,atol=2e-5,equal_nan=True):
                raise AssertionError('Positive scale failed to preserve within-song Pearson')
    pd.DataFrame(rows).to_csv(root/'seed_summary.csv',index=False)
    pd.DataFrame(song_rows).to_csv(root/'per_song_metrics.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(root/'paired_bootstraps.csv',index=False)
    summary=pd.DataFrame(rows)
    print(summary[['repeat','model','med','valence_within_pearson','arousal_within_pearson']].to_string(index=False))
    print(summary.groupby('model')[list(METRICS)].agg(['mean','std']).to_string())

if __name__=='__main__':main()
