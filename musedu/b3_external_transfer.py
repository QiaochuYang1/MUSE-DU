"""Zero-update external evaluation of five nested DEAM B2/MERT/B3 ensembles."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .b3_level_fusion import fuse
from .experiment import read_npz
from .matched_development import make_model,output_mean
from .published_baselines.io import normalized,result_root,save_report
from .published_baselines.models import create_model
from .plan_matrix_2x2 import sha

MERP_EXCLUDED={'deam_115','deam_1334','deam_343','deam_745'}

def load_models(nested,calibration,device):
    models=[]
    for fold in range(1,6):
        b=torch.load(nested/f'fold_{fold}'/'B2'/'selected_checkpoint.pt',map_location='cpu',weights_only=False)
        m=torch.load(nested/f'fold_{fold}'/'MERT'/'selected_checkpoint.pt',map_location='cpu',weights_only=False)
        bm=make_model('v1',len(b['normalizer_mean']),96).to(device)
        bm.load_state_dict(b['state_dict']);bm.eval()
        mm=create_model('probe',(24,1024)).to(device)
        mm.load_state_dict(m['state_dict']);mm.eval()
        meta=json.loads((calibration/f'fold_{fold}'/'validation_metrics.json').read_text())['metadata']
        if meta['scale_valence']!=1. or meta['scale_arousal']!=1.:
            raise ValueError('Expected level-only nested calibration')
        params=[meta['alpha_valence'],meta['alpha_arousal'],1.,1.]
        models.append((bm,b['normalizer_mean'],b['normalizer_std'],
                       mm,m['normalizer_mean'],m['normalizer_std'],params))
    return models

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external-manifest',required=True);p.add_argument('--features',required=True)
    p.add_argument('--nested-root',required=True);p.add_argument('--calibration-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();out=Path(a.out).resolve();features=Path(a.features).resolve()
    nested=Path(a.nested_root).resolve();calibration=Path(a.calibration_root).resolve()
    with open(a.external_manifest,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    dataset={r['dataset'] for r in rows}
    if len(dataset)!=1 or 'deam' in dataset:raise ValueError('Expected one external dataset')
    dataset=dataset.pop()
    if dataset=='merp':rows=[r for r in rows if r['song_id'] not in MERP_EXCLUDED]
    ids=[r['song_id'] for r in rows]
    if len(ids)!=len(set(ids)) or len(ids)!=(767 if dataset=='pmemo' else 50):
        raise AssertionError('External song count or IDs changed')
    if not set(ids).issubset({p.stem for p in features.glob('*.npz')}):
        raise AssertionError('External MERT-330M features incomplete')
    out.mkdir(parents=True,exist_ok=True)
    protocol={'source':'DEAM nested five-fold ensembles','external_dataset':dataset,
              'external_song_count':len(ids),'excluded_merp_ids':sorted(MERP_EXCLUDED) if dataset=='merp' else [],
              'external_manifest_sha256':sha(a.external_manifest),
              'external_mert_feature_protocol_sha256':sha(features/'protocol.json'),
              'nested_protocol_sha256':sha(nested/'protocol.json'),
              'calibration':'DEAM nested level-only, one alpha per axis per fold',
              'external_labels_used_for_training_checkpoint_selection_or_calibration':False,
              'ensemble':'arithmetic mean of five fold-specific predictions'}
    file=out/'protocol.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('External protocol changed')
    file.write_text(json.dumps(protocol,indent=2))
    models=load_models(nested,calibration,a.device)
    predictions={name:{} for name in ('B2_ensemble','MERT_ensemble','B3_level_only')}
    index={r['song_id']:r['file'] for r in rows}
    with torch.inference_mode():
        for i,row in enumerate(rows,1):
            sid=row['song_id'];z=read_npz(row['file']);times=z['times']
            if dataset=='pmemo' and times[0]<15:raise ValueError('PMEmo timestamp alignment')
            with np.load(features/f'{sid}.npz',allow_pickle=False) as archive:
                if not np.allclose(archive['times'],times,atol=1e-3):
                    raise ValueError('External MERT timestamp mismatch')
                mert_features=archive['features']
            x_b=np.concatenate((z['acoustic'],z['embedding']),axis=-1)
            per={'B2_ensemble':[],'MERT_ensemble':[],'B3_level_only':[]}
            valid=torch.ones((1,len(times)),dtype=torch.bool,device=a.device)
            tt=torch.as_tensor(times[None],dtype=torch.float32,device=a.device)
            for bm,bmu,bsd,mm,mmu,msd,params in models:
                bx=torch.as_tensor(((x_b-bmu)/bsd)[None],dtype=torch.float32,device=a.device)
                mx=torch.as_tensor(normalized(mert_features,mmu,msd)[None],dtype=torch.float32,device=a.device)
                bp=output_mean(bm(bx,tt,valid))[0].cpu().numpy()
                mp=mm(mx,valid)[0].cpu().numpy()
                per['B2_ensemble'].append(bp);per['MERT_ensemble'].append(mp)
                per['B3_level_only'].append(fuse(bp,mp,params))
            for name,items in per.items():
                predictions[name][sid]=np.mean(np.stack(items),axis=0).astype(np.float32)
            if i%25==0 or i==len(rows):print(f'EXTERNAL {dataset} {i}/{len(rows)}',flush=True)
    all_rows=[];tables={}
    for name,pp in predictions.items():
        dest=out/name
        if not (dest/'validation_metrics.json').exists():
            result_root(dest)
            save_report(dest,pp,index,ids,{'method':name,'dataset':dataset,
                        'external_labels_used_for_training_checkpoint_selection_or_calibration':False})
        report=json.loads((dest/'validation_metrics.json').read_text())
        all_rows.append({'model':name,**report['metrics']})
        tables[name]=pd.read_csv(dest/'validation_by_song.csv',dtype={'song_id':str}).set_index('song_id')
    pd.DataFrame(all_rows).to_csv(out/'summary.csv',index=False)
    for axis in ('valence','arousal'):
        aa=tables['B3_level_only'][axis+'_within_pearson'].to_numpy()
        bb=tables['B2_ensemble'][axis+'_within_pearson'].to_numpy()
        if not np.allclose(aa,bb,atol=2e-5,equal_nan=True):
            raise AssertionError('External level-only Pearson changed')
    rng=np.random.default_rng(2026);boot=[]
    for other in ('B2_ensemble','MERT_ensemble'):
        for metric in ('med','valence_ccc','arousal_ccc','valence_within_pearson','arousal_within_pearson'):
            delta=(tables['B3_level_only'].loc[ids,metric]-tables[other].loc[ids,metric]).to_numpy(dtype=float)
            draws=rng.integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
            samples=np.nanmean(delta[draws],axis=1)
            boot.append({'contrast':'B3_level_only minus '+other,'metric':metric,
                         'difference':float(np.nanmean(delta)),
                         'ci95_lower':float(np.quantile(samples,.025)),
                         'ci95_upper':float(np.quantile(samples,.975)),
                         'bootstrap_draws':10000,'bootstrap_unit':'song'})
    pd.DataFrame(boot).to_csv(out/'paired_bootstraps.csv',index=False)
    print(pd.DataFrame(all_rows)[['model','med','valence_within_pearson',
                                   'arousal_within_pearson','valence_ccc','arousal_ccc']].to_string(index=False))

if __name__=='__main__':main()
