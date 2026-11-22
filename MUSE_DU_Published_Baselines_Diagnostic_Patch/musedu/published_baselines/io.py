"""Strict loaders, train-only normalization, shared metrics, anti-leakage checks."""
from __future__ import annotations
import json, os, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch

AXES = ('valence', 'arousal')


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_index(manifest, reference_run):
    """Only the preexisting five-way split is accepted; *never* read test labels."""
    from musedu.experiment import load_manifest, provenance
    df = load_manifest(manifest)
    if df.dataset.nunique() != 1 or df.song_id.duplicated().any():
        raise ValueError('Exactly one dataset and unique song IDs required')
    root = Path(reference_run)
    splits = json.loads((root/'splits.json').read_text())
    if set(splits) != {'train','val','scale','cal','test'}:
        raise ValueError('Reference split must have train, val, scale, cal, test')
    ids = [str(s) for v in splits.values() for s in v]
    if len(set(ids)) != len(ids) or set(ids) != set(df.song_id.astype(str)):
        raise ValueError('Reference split IDs duplicate or do not match manifest')
    original = json.loads((root/'source_manifest.json').read_text())
    lhs = {(str(r['dataset']), str(r['song_id'])):r['audio_sha256'] for r in original}
    rhs = {(str(r['dataset']), str(r['song_id'])):r['audio_sha256'] for r in provenance(df)}
    if lhs != rhs:
        raise ValueError('Audio provenance differs from reference MUSE-DU run')
    return {str(row.song_id):str(row.file) for row in df.itertuples()}, splits


def load_song(index, sid):
    from musedu.experiment import read_npz
    z = read_npz(index[str(sid)])
    times=np.asarray(z['times'],np.float64)
    if np.any(np.diff(times)<=0):raise ValueError(f'{sid}: unsorted timestamps')
    return z


def feature_archive_path(feature_root, kind, sid):
    sid=str(sid)
    if '/' in sid or '\\' in sid or sid in ('.','..'):raise ValueError('Unsafe song_id')
    return Path(feature_root)/kind/(sid+'.npz')


def get_features(index,sid,view,feature_root):
    z=load_song(index,sid)
    if view=='fusion':
        v=np.concatenate([z['acoustic'],z['embedding']],axis=-1)
    elif view=='mert_last':
        v=z['embedding']
    else:
        f=feature_archive_path(feature_root,view,sid)
        if not f.exists():raise FileNotFoundError(f'{f}: prepare {view} features first')
        with np.load(f,allow_pickle=False) as a:
            if 'times' not in a or 'features' not in a:raise ValueError(f'{f}: missing times/features')
            if not np.allclose(a['times'],z['times'],atol=1e-3):raise ValueError(f'{sid}: feature timestamps mismatch')
            v=a['features']
    if len(v)!=len(z['times']) or not np.isfinite(v).all():
        raise ValueError(f'{sid}: invalid {view} features')
    return np.asarray(v,dtype=np.float32),z


def train_normalizer(index, ids, view, features_dir):
    # Streaming train-only mean and SD; no leakage from held-out songs.
    n=0;total=None;total2=None
    for sid in ids:
        x,_=get_features(index,sid,view,features_dir)
        if x.ndim != 2:raise ValueError('Normalizer only for 2D descriptors')
        if total is None:total=np.zeros(x.shape[-1],np.float64);total2=total.copy()
        if x.shape[-1]!=len(total):raise ValueError('Feature dimension mismatch across songs')
        total+=x.sum(0,dtype=np.float64)
        total2+=(x.astype(np.float64)**2).sum(0)
        n+=len(x)
    mean=total/n;std=np.maximum(np.sqrt(np.maximum(total2/n-mean**2,0)),1e-5)
    return mean.astype('float32'),std.astype('float32')


def normalized(x,mean,std):
    return np.clip((x-mean)/std,-5.,5.).astype('float32')


def ccc(x,y):
    if len(x)<2:return np.nan
    mx,my=np.mean(x),np.mean(y)
    cov=np.mean((x-mx)*(y-my))
    denom=np.var(x)+np.var(y)+(mx-my)**2
    return float(2*cov/denom) if denom>1e-12 else np.nan


def pearson(x,y):
    if len(x)<3:return np.nan
    x=x-x.mean();y=y-y.mean()
    z=np.sqrt((x*x).sum()*(y*y).sum())
    return float((x*y).sum()/z) if z>1e-9 else np.nan


def per_song_metrics(pred,gt,mask):
    out={}
    for d,axis in enumerate(AXES):
        ok=mask[:,d].astype(bool)
        x=gt[ok,d].astype(np.float64);y=pred[ok,d].astype(np.float64)
        if len(x)==0:
            for name in ('mae','rmse','ccc','within_pearson'):out[f'{axis}_{name}']=np.nan
            continue
        out[f'{axis}_mae']=float(np.mean(np.abs(x-y)))
        out[f'{axis}_rmse']=float(np.sqrt(np.mean((x-y)**2)))
        out[f'{axis}_ccc']=ccc(x,y)
        out[f'{axis}_within_pearson']=pearson(x,y)
    both=mask.all(axis=1)
    if both.any():out['med']=float(np.linalg.norm(pred[both]-gt[both],axis=1).mean())
    else:out['med']=np.nan
    return out


def evaluate(predictions,index,ids):
    rec=[]; ys=[[],[]]; ps=[[],[]]
    for sid in ids:
        z=load_song(index,sid)
        if sid not in predictions:raise ValueError('Prediction missing song: '+sid)
        p=np.asarray(predictions[sid],dtype=np.float64)
        if p.shape!=z['mean'].shape or not np.isfinite(p).all():raise ValueError('Prediction shape/nonfinite: '+sid)
        m=z['mask'];truth=z['mean']
        rec.append({'song_id':sid,**per_song_metrics(p,truth,m)})
        for d in range(2):
            ok=m[:,d];ys[d].append(truth[ok,d]);ps[d].append(p[ok,d])
    df=pd.DataFrame(rec)
    out={col:float(df[col].mean(skipna=True)) for col in df if col!='song_id'}
    for d,axis in enumerate(AXES):
        out[axis+'_pooled_ccc']=ccc(np.concatenate(ys[d]),np.concatenate(ps[d]))
    return df,out


def save_report(root,predictions,index,ids,metadata):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    df,metrics=evaluate(predictions,index,ids)
    df.to_csv(root/'validation_by_song.csv',index=False)
    for sid in ids:
        z=load_song(index,sid)
        np.savez_compressed(root/'predictions'/f'{sid}.npz',
          pred=predictions[sid],mean=z['mean'],mask=z['mask'],times=z['times'])
    result={'metrics':metrics,'metadata':metadata}
    (root/'validation_metrics.json').write_text(json.dumps(result,indent=2))
    return metrics


def result_root(path):
    root=Path(path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Non-empty output directory; refusing overwrite: {root}')
    root.mkdir(parents=True,exist_ok=True)
    (root/'predictions').mkdir(exist_ok=True)
    return root
