"""Aggregate nested shape-0.05 fusion and compare with its paired components."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .published_baselines.io import load_index,evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--nested-root',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root);nested=Path(a.nested_root)
    index,_=load_index(a.manifest,a.reference_run)
    pp={name:{} for name in ('B2_05','B3_C_05','MERT','B2_02','B3_C_02')}
    for fold in range(1,6):
        sources={'B2_05':root/f'fold_{fold}'/'B2_05'/'evaluation'/'predictions',
                 'B3_C_05':root/f'fold_{fold}'/'B3_C_05'/'evaluation'/'predictions',
                 'MERT':nested/f'fold_{fold}'/'MERT'/'evaluation'/'predictions',
                 'B2_02':nested/f'fold_{fold}'/'B2'/'evaluation'/'predictions',
                 'B3_C_02':nested/f'fold_{fold}'/'B3_C'/'evaluation'/'predictions'}
        for name,folder in sources.items():
            for file in folder.glob('*.npz'):
                if file.stem in pp[name]:raise AssertionError('Duplicate OOF song')
                with np.load(file,allow_pickle=False) as z:pp[name][file.stem]=z['pred']
    rows=[];tables={}
    for name,pred in pp.items():
        if set(pred)!=set(index):raise AssertionError('Incomplete OOF predictions')
        table,metrics=evaluate(pred,index,sorted(index))
        rows.append({'model':name,'songs':1802,**metrics});tables[name]=table.set_index('song_id')
    pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    rng=np.random.default_rng(2026);boot=[];ids=sorted(index)
    for other in ('B2_05','MERT','B3_C_02'):
        for metric in ('med','valence_ccc','arousal_ccc','valence_within_pearson','arousal_within_pearson'):
            delta=(tables['B3_C_05'].loc[ids,metric]-tables[other].loc[ids,metric]).to_numpy(dtype=float)
            draws=rng.integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
            sample=np.nanmean(delta[draws],axis=1)
            boot.append({'contrast':'B3_C_05 minus '+other,'metric':metric,
                         'difference':float(np.nanmean(delta)),
                         'ci95_lower':float(np.quantile(sample,.025)),
                         'ci95_upper':float(np.quantile(sample,.975)),
                         'bootstrap_draws':10000})
    pd.DataFrame(boot).to_csv(root/'paired_bootstraps.csv',index=False)
    print(pd.DataFrame(rows)[['model','med','valence_within_pearson',
                              'arousal_within_pearson','valence_ccc','arousal_ccc']].to_string(index=False))

if __name__=='__main__':main()
