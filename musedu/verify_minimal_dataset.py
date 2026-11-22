"""Independently recompute principal metrics from the shareable CSV exports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

_bundle_results=Path(__file__).resolve().parents[2]/'results'
CV=(_bundle_results/'deam_cv5_stage1' if _bundle_results.is_dir()
    else Path(r'E:\deam_cv5_stage1'))
PMEMO=(_bundle_results/'deam_pmemo_frozen_transfer' if _bundle_results.is_dir()
       else Path(r'E:\deam_pmemo_frozen_transfer'))
MERP=(_bundle_results/'deam_merp_frozen_transfer' if _bundle_results.is_dir()
      else Path(r'E:\deam_merp_frozen_transfer'))

def pearson(x,y):
    if len(x)<3:return np.nan
    x=x-x.mean();y=y-y.mean()
    d=np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/d) if d>1e-9 else np.nan

def summarize(table,prefix,keys):
    rows=[]
    for _,song in table.groupby(keys,sort=False):
        p=song[[prefix+'_valence',prefix+'_arousal']].to_numpy(np.float64)
        y=song[['true_valence','true_arousal']].to_numpy(np.float64)
        m=song[['valid_valence','valid_arousal']].to_numpy(bool)
        if np.any(~np.isfinite(y[m])) or np.any(~np.isfinite(p)):
            raise AssertionError('Invalid values in CSV')
        both=m.all(axis=1)
        result={'med':float(np.linalg.norm(p[both]-y[both],axis=1).mean()) if both.any() else np.nan}
        for j,axis in enumerate(('valence','arousal')):
            x=y[m[:,j],j];z=p[m[:,j],j]
            result[axis+'_within_pearson']=pearson(x,z)
        rows.append(result)
    return {name:float(np.nanmean([r[name] for r in rows]))
            for name in ('med','valence_within_pearson','arousal_within_pearson')},len(rows)

def check(actual,expected,label):
    for key,value in actual.items():
        if not np.isclose(value,expected[key],rtol=0,atol=5e-7):
            raise AssertionError(f'{label} {key}: {value} != {expected[key]}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True)
    a=parser.parse_args();folder=a.folder.resolve()
    deam=pd.read_csv(folder/'DEAM_predictions.csv',dtype={'song_id':str})
    if len(deam)!=130002 or deam.song_id.nunique()!=1802:
        raise AssertionError('DEAM export count mismatch')
    for prefix,method in (('b2','B2'),('point','B2_point')):
        metric,songs=summarize(deam,prefix,['song_id'])
        expected=json.loads((CV/method/'crossval_summary.json').read_text())['metrics']
        if songs!=1802:raise AssertionError('DEAM song count')
        check(metric,expected,method)
        print('DEAM',method,metric)
    external=pd.read_csv(folder/'External_predictions.csv',dtype={'song_id':str})
    if len(external)!=52043:raise AssertionError('External frame count')
    for dataset,root,count in (('PMEmo',PMEMO,767),('MERP',MERP,50)):
        subset=external[external.dataset==dataset]
        if subset.song_id.nunique()!=count:raise AssertionError(dataset+' song count')
        for prefix,method in (('b2','B2_temporal_shape'),('point','B2_point_only')):
            metric,songs=summarize(subset,prefix,['song_id'])
            expected=json.loads((root/method/'validation_metrics.json').read_text())['metrics']
            if songs!=count:raise AssertionError(dataset+' song count')
            check(metric,expected,dataset+' '+method)
            print(dataset,method,metric)
    baseline=pd.read_csv(folder/'Baseline_results.csv')
    ablation=pd.read_csv(folder/'Ablation_results.csv')
    bootstrap=pd.read_csv(folder/'Bootstrap_results.csv')
    if (len(baseline),len(ablation),len(bootstrap))!=(42,54,81):
        raise AssertionError('Results table row count mismatch')
    zip_path=folder.parent/(folder.name+'.zip')
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:raise AssertionError('Corrupt ZIP member')
        names=sorted(p.name for p in folder.iterdir() if p.is_file())
        if sorted(z.namelist())!=names:raise AssertionError('ZIP member mismatch')
        for name in names:
            if z.read(name)!=(folder/name).read_bytes():
                raise AssertionError('ZIP bytes differ: '+name)
    print('PASS: metrics, row counts, and ZIP members verified')

if __name__=='__main__':main()
