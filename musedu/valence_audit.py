"""Training-fold delay selection and development annotation/cache diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .experiment import load_manifest,read_npz
from .v2_experiment import _reference_split


def corr(a,b):
    keep=np.isfinite(a)&np.isfinite(b);a=a[keep];b=b[keep]
    return float(np.corrcoef(a,b)[0,1]) if len(a)>=3 and np.ptp(a)>1e-7 and np.ptp(b)>1e-7 else np.nan


def delay_folds(manifest,reference,out):
    df=load_manifest(manifest);split=_reference_split(df,reference);lookup=dict(zip(df.song_id,df.file))
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261027);order=rng.permutation(split['train']);folds=np.array_split(order,3)
    projection=np.random.default_rng(2026).normal(size=(768,96)).astype('float32')/np.sqrt(96)
    delays=(0.,.5,1.,2.,3.,4.);data={};rows=[]
    for sid in split['train']:
        z=read_npz(lookup[sid]);features={}
        # An inexpensive fixed representation probes alignment without tuning the neural models.
        for delay in delays:
            ix=np.searchsorted(z['context_times'],z['times']-delay)
            if not np.allclose(z['context_times'][ix],z['times']-delay):raise ValueError('Lag mapping')
            e=z['context_layers'][ix].mean(1)@projection
            x=np.concatenate((z['context_acoustic'][ix],e),axis=1)
            features[delay]=x-x.mean(0)
        data[sid]=(features,z['mean'],z['mask'])
    for fold,valid in enumerate(folds):
        val_ids=set(valid);train=[sid for sid in split['train'] if sid not in val_ids]
        for delay in delays:
            x=np.concatenate([data[sid][0][delay] for sid in train]);mu=x.mean(0);sd=x.std(0).clip(1e-4)
            models=[]
            for axis in range(2):
                xx=[];yy=[];ww=[]
                for sid in train:
                    f,y,m=data[sid];keep=m[:,axis];target=y[keep,axis]
                    xx.append((f[delay][keep]-mu)/sd);yy.append(target-target.mean())
                    ww.append(np.full(keep.sum(),1/max(1,keep.sum())))
                model=Ridge(alpha=1.).fit(np.concatenate(xx),np.concatenate(yy),sample_weight=np.concatenate(ww))
                models.append(model)
            scores=[[],[]]
            for sid in valid:
                f,y,m=data[sid];p=np.column_stack([model.predict((f[delay]-mu)/sd) for model in models])
                for axis in range(2):scores[axis].append(corr(p[m[:,axis],axis],y[m[:,axis],axis]))
            rows.append({'fold':fold,'delay_seconds':delay,'valence_within_pearson':float(np.nanmean(scores[0])),
                         'arousal_within_pearson':float(np.nanmean(scores[1]))})
            print(f'lag fold={fold} delay={delay}: valence={rows[-1]["valence_within_pearson"]:.4f}',flush=True)
    table=pd.DataFrame(rows);table.to_csv(root/'delay_fold_metrics.csv',index=False)
    means=table.groupby('delay_seconds').valence_within_pearson.mean()
    candidate=float(means.idxmax());pivot=table.pivot(index='fold',columns='delay_seconds',values='valence_within_pearson')
    gain=float(means.loc[candidate]-means.loc[0.]);wins=int((pivot[candidate]>pivot[0.]).sum())
    selected=candidate if gain>=.01 and wins>=2 else 0.
    protocol={'candidate_delays':delays,'selected_delay_seconds':selected,'best_probe_delay':candidate,
              'mean_gain_over_zero':gain,'fold_wins_over_zero':wins,
              'rule':'choose nonzero only if mean valence gain >= .01 and wins in >=2/3 folds',
              'folds':[list(map(str,fold)) for fold in folds],
              'selection_song_ids':split['train'],'validation_used_for_delay_selection':False,
              'probe':'fixed random projection of mean frozen layers + acoustic; centered per song; Ridge alpha=1'}
    (root/'delay_selection.json').write_text(json.dumps(protocol,indent=2));return selected


def annotations_and_boundaries(old_manifest,new_manifest,reference,out,annotation_root):
    old=load_manifest(old_manifest);new=load_manifest(new_manifest);split=_reference_split(old,reference)
    a=old.set_index('song_id');b=new.set_index('song_id');root=Path(out);root.mkdir(parents=True,exist_ok=True)
    base=Path(annotation_root)/'DEAM_Annotations'/'annotations'/'annotations per each rater'/'dynamic (per second annotations)'
    rng=np.random.default_rng(20261027);reliability=[];boundary=[];identities=[]
    for sid in split['train']+split['val']:
        z=read_npz(a.loc[sid,'file']);zz=read_npz(b.loc[sid,'file'])
        identity=all(np.array_equal(z[k],zz[k],equal_nan=True) for k in ('times','mean','mask','std','std_mask','raters'))
        identity=identity and str(z['audio_sha256'].item())==str(zz['audio_sha256'].item())
        if not identity:raise ValueError(f'Cache changed source/labels {sid}')
        identities.append(sid)
        if sid in split['train']:
            for version,e in (('old',z['embedding']),('overlap',zz['embedding'])):
                # Per-song scale removes arbitrary feature-unit differences from boundary diagnostic.
                normalized=(e-e.mean(0))/e.std(0).clip(1e-4)
                change=np.linalg.norm(np.diff(normalized,axis=0),axis=1)
                near=(np.mod(z['times'][1:],5.)<=.5)
                if near.any() and (~near).any():
                    boundary.append({'song_id':sid,'version':version,'boundary_change':float(change[near].mean()),
                                     'interior_change':float(change[~near].mean())})
        for axis,name in enumerate(('valence','arousal')):
            f=base/name/f'{sid}.csv'
            if not f.is_file():continue
            frame=pd.read_csv(f);cols=[f'sample_{int(round(t*1000))}ms' for t in z['times']]
            missing=sum(c not in frame for c in cols)
            ratings=frame.reindex(columns=cols).apply(pd.to_numeric,errors='coerce').to_numpy();n=len(ratings)
            if n<4:continue
            def observed_mean(values):
                counts=np.isfinite(values).sum(0)
                return np.divide(np.nansum(values,axis=0),counts,
                                 out=np.full(values.shape[1],np.nan),where=counts>0)
            scores=[]
            for repeat in range(20):
                order=rng.permutation(n);mid=n//2
                scores.append(corr(observed_mean(ratings[order[:mid]]),observed_mean(ratings[order[mid:]])))
            target=z['mean'][z['mask'][:,axis],axis]
            reliability.append({'song_id':sid,'split':'train' if sid in split['train'] else 'val',
                'axis':name,'n_raters':n,'split_half_pearson':float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan,
                'target_sd':float(target.std()),
                'missing_rater_time_columns':missing,
                'rater_mean_vs_release_mae':float(np.nanmean(np.abs(observed_mean(ratings)-z['mean'][:,axis])))})
    pd.DataFrame(reliability).to_csv(root/'inter_rater_reliability.csv',index=False)
    bt=pd.DataFrame(boundary);bt.to_csv(root/'feature_boundary_diagnostics.csv',index=False)
    (root/'identity_audit.json').write_text(json.dumps({'verified_development_songs':len(identities),
        'source_audio_hashes_labels_masks_raters_times_unchanged':True,'scale_cal_test_labels_read':False,
        'annotation_delay_note':'Acoustic alignment can be probed; subjective response delay is not directly observed.'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True);p.add_argument('--old-manifest');p.add_argument('--annotation-root')
    a=p.parse_args();delay_folds(a.manifest,a.reference,a.out)
    if a.old_manifest and a.annotation_root:
        annotations_and_boundaries(a.old_manifest,a.manifest,a.reference,a.out,a.annotation_root)
