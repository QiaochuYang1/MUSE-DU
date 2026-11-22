"""Verify nested B3-C outer predictions and calculate paired bootstrap intervals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .published_baselines.io import load_index,evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root)
    index,_=load_index(a.manifest,a.reference_run)
    names=('B2','MERT','B3_C');preds={name:{} for name in names}
    params=[]
    for fold in range(1,6):
        folder=root/f'fold_{fold}'
        split=json.loads((folder/'split.json').read_text())
        parts=[set(split[k]) for k in ('fit_ids','checkpoint_ids','calibration_ids','outer_ids')]
        if sum(map(len,parts))!=1802 or len(set.union(*parts))!=1802:
            raise AssertionError('Nested split incomplete or overlapping')
        cal=json.loads((folder/'calibration.json').read_text())
        if cal['outer_labels_used_for_fit_selection_or_calibration']:
            raise AssertionError('Outer-fold leakage flag')
        params.append(cal)
        for name in names:
            pred_dir=folder/name/'evaluation'/'predictions'
            if {f.stem for f in pred_dir.glob('*.npz')}!=parts[-1]:
                raise AssertionError('Fold prediction IDs differ from frozen outer IDs')
            for file in pred_dir.glob('*.npz'):
                if file.stem in preds[name]:raise AssertionError('Duplicate OOF song')
                with np.load(file,allow_pickle=False) as z:preds[name][file.stem]=z['pred']
    rows=[];tables={}
    for name in names:
        if set(preds[name])!=set(index):raise AssertionError('Missing complete OOF set')
        by_song,metric=evaluate(preds[name],index,sorted(index))
        tables[name]=by_song.set_index('song_id')
        rows.append({'model':name,'songs':1802,**metric})
    for axis in ('valence','arousal'):
        aa=tables['B3_C'][axis+'_within_pearson'].to_numpy()
        bb=tables['B2'][axis+'_within_pearson'].to_numpy()
        if not np.allclose(aa,bb,atol=2e-5,equal_nan=True):
            raise AssertionError('B3 Pearson not preserved')
    summary=pd.DataFrame(rows);summary.to_csv(root/'summary.csv',index=False)
    pd.DataFrame(params).to_csv(root/'calibration_by_fold.csv',index=False)
    rng=np.random.default_rng(2026);boot=[];ids=sorted(index)
    for other in ('B2','MERT'):
        for metric in ('med','valence_within_pearson','arousal_within_pearson'):
            delta=(tables['B3_C'].loc[ids,metric]-tables[other].loc[ids,metric]).to_numpy(dtype=float)
            draws=rng.integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
            sample=np.nanmean(delta[draws],axis=1)
            boot.append({'contrast':'B3_C minus '+other,'metric':metric,
                         'difference':float(np.nanmean(delta)),
                         'ci95_lower':float(np.quantile(sample,.025)),
                         'ci95_upper':float(np.quantile(sample,.975)),
                         'bootstrap_draws':10000,'bootstrap_unit':'song'})
    pd.DataFrame(boot).to_csv(root/'paired_bootstraps.csv',index=False)
    print(summary[['model','med','valence_within_pearson','arousal_within_pearson']].to_string(index=False))

if __name__=='__main__':main()
