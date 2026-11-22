"""Nested confirmation of level-only B3 fusion with B2 trajectory scale fixed to one."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .b3_level_fusion import fit_calibrator,fuse
from .published_baselines.io import load_index,evaluate,result_root,save_report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--nested-root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();nested=Path(a.nested_root);out=Path(a.out)
    index,_=load_index(a.manifest,a.reference_run)
    results={name:{} for name in ('B2','MERT','B3_level_only')}
    out.mkdir(parents=True,exist_ok=True)
    for fold in range(1,6):
        source=nested/f'fold_{fold}'
        split=json.loads((source/'split.json').read_text())
        cal=split['calibration_ids'];outer=split['outer_ids']
        b2_cal={};mert_cal={}
        for sid in cal:
            with np.load(source/'calibration_predictions'/f'{sid}.npz',allow_pickle=False) as z:
                b2_cal[sid]=z['b2'];mert_cal[sid]=z['mert']
        params,detail=fit_calibrator(cal,b2_cal,mert_cal,index,scale_fixed=1.)
        if not np.array_equal(params[2:],[1.,1.]):raise AssertionError('Scale drift')
        outer_predictions={}
        for name in ('B2','MERT'):
            outer_predictions[name]={}
            for sid in outer:
                with np.load(source/name/'evaluation'/'predictions'/f'{sid}.npz',allow_pickle=False) as z:
                    outer_predictions[name][sid]=z['pred']
        fused={sid:fuse(outer_predictions['B2'][sid],outer_predictions['MERT'][sid],params).astype(np.float32)
               for sid in outer}
        destination=out/f'fold_{fold}'
        if not (destination/'validation_metrics.json').exists():
            result_root(destination)
            save_report(destination,fused,index,outer,
                        {'fold':fold,'alpha_valence':float(params[0]),
                         'alpha_arousal':float(params[1]),'scale_valence':1.,'scale_arousal':1.,
                         'calibration':detail,'calibration_songs':len(cal),
                         'checkpoint_and_calibration_disjoint':True,
                         'outer_labels_used_for_calibration':False})
        for name,pp in (('B2',outer_predictions['B2']),('MERT',outer_predictions['MERT']),
                        ('B3_level_only',fused)):
            results[name].update(pp)
    summary=[];by_song={}
    for name,pp in results.items():
        if set(pp)!=set(index):raise AssertionError('Incomplete OOF predictions')
        table,metric=evaluate(pp,index,sorted(index));by_song[name]=table.set_index('song_id')
        summary.append({'model':name,'songs':1802,**metric})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    for axis in ('valence','arousal'):
        aa=by_song['B3_level_only'][axis+'_within_pearson'].to_numpy()
        bb=by_song['B2'][axis+'_within_pearson'].to_numpy()
        if not np.allclose(aa,bb,atol=2e-5,equal_nan=True):
            raise AssertionError('Level-only Pearson changed')
    rng=np.random.default_rng(2026);boot=[];ids=sorted(index)
    for other in ('B2','MERT'):
        for metric in ('med','valence_ccc','arousal_ccc','valence_within_pearson','arousal_within_pearson'):
            delta=(by_song['B3_level_only'].loc[ids,metric]-by_song[other].loc[ids,metric]).to_numpy(dtype=float)
            draws=rng.integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
            sample=np.nanmean(delta[draws],axis=1)
            boot.append({'contrast':'B3_level_only minus '+other,'metric':metric,
                         'difference':float(np.nanmean(delta)),
                         'ci95_lower':float(np.quantile(sample,.025)),
                         'ci95_upper':float(np.quantile(sample,.975)),
                         'bootstrap_draws':10000})
    pd.DataFrame(boot).to_csv(out/'paired_bootstraps.csv',index=False)
    print(pd.DataFrame(summary)[['model','med','valence_within_pearson',
                                 'arousal_within_pearson','valence_ccc','arousal_ccc']].to_string(index=False))

if __name__=='__main__':main()
