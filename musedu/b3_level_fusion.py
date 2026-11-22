"""B3-C pilot: fold-internal level/trajectory calibration of saved B2 and MERT models.

No outer-fold label is read while choosing alpha or trajectory scale. The same
internal songs were previously used to choose both base checkpoints; this
exploratory calibration therefore requires a later fully nested confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from .experiment import read_npz
from .matched_development import make_model, output_mean
from .published_baselines.io import (load_index, load_song, get_features,
                                      normalized, result_root, save_report, evaluate)
from .published_baselines.models import create_model
from .plan_matrix_2x2 import sha


def fuse(b2,mert,params):
    """Per-axis alpha and positive residual scale; no clipping preserves Pearson r."""
    alpha=np.asarray(params[:2],np.float64)
    scale=np.asarray(params[2:],np.float64)
    if np.any(scale<=0):raise ValueError('Trajectory scales must be positive')
    mu_b=b2.mean(axis=0,keepdims=True)
    mu_m=mert.mean(axis=0,keepdims=True)
    return (1-alpha)*mu_b+alpha*mu_m+scale*(b2-mu_b)


def b2_internal(index,ids,checkpoint,normalizer,device):
    with np.load(normalizer,allow_pickle=False) as z:mu=z['mean'];sd=z['std']
    model=make_model('v1',len(mu),96).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['state_dict'])
    model.eval();out={}
    with torch.inference_mode():
        for sid in ids:
            z=read_npz(index[sid])
            x=np.concatenate((z['acoustic'],z['embedding']),axis=-1)
            x=(x-mu)/sd
            xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
            tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device)
            valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
            out[sid]=output_mean(model(xx,tt,valid))[0].cpu().numpy().astype(np.float32)
    return out


def mert_internal(index,ids,checkpoint,features,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    mu=saved['normalizer_mean'];sd=saved['normalizer_std']
    model=create_model('probe',(24,1024)).to(device)
    model.load_state_dict(saved['state_dict']);model.eval();out={}
    with torch.inference_mode():
        for sid in ids:
            x,_=get_features(index,sid,'mert',features)
            x=normalized(x,mu,sd)
            xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
            valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
            out[sid]=model(xx,valid)[0].cpu().numpy().astype(np.float32)
    return out


def fit_calibrator(ids,b2,mert,index,scale_fixed=None):
    records=[]
    for sid in ids:
        z=load_song(index,sid)
        valid=z['mask'].all(axis=1)
        if valid.any():records.append((b2[sid],mert[sid],z['mean'],valid))
    def expanded(params):
        return np.asarray(params if scale_fixed is None else [params[0],params[1],scale_fixed,scale_fixed],np.float64)
    def objective(params):
        return float(np.mean([np.linalg.norm(fuse(b,m,params)[ok]-y[ok],axis=1).mean()
                              for b,m,y,ok in records]))
    def optimized_objective(params):return objective(expanded(params))
    start=np.array([.5,.5,1.,1.] if scale_fixed is None else [.5,.5],np.float64)
    bounds=[(0.,1.),(0.,1.),(.25,2.),(.25,2.)] if scale_fixed is None else [(0.,1.),(0.,1.)]
    result=minimize(optimized_objective,start,bounds=bounds,
                    method='L-BFGS-B',options={'maxiter':150,'ftol':1e-10,'eps':1e-5})
    if not result.success and not np.isfinite(result.fun):raise RuntimeError(result.message)
    return expanded(result.x),{'calibration_songs':len(records),'starting_med':optimized_objective(start),
                     'selected_med':optimized_objective(result.x),'optimizer_success':bool(result.success),
                     'optimizer_message':str(result.message)}


def predictions_on_disk(folder,ids):
    out={}
    for sid in ids:
        with np.load(folder/f'{sid}.npz',allow_pickle=False) as z:out[sid]=z['pred']
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--matrix-root',required=True);p.add_argument('--mert-root',required=True)
    p.add_argument('--features',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--repeats',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--fixed-scale',type=float,default=None,
                   help='Fix both trajectory scales, e.g. 1.0 for level-only fusion')
    a=p.parse_args();out=Path(a.out).resolve();matrix=Path(a.matrix_root).resolve()
    teacher=Path(a.mert_root).resolve();features=Path(a.features).resolve()
    index,_=load_index(a.manifest,a.reference_run)
    splits=json.loads((matrix/'frozen_splits.json').read_text())['folds']
    protocol={'name':'B3-C level-trajectory fusion pilot','base_b2':'full-budget shape/shape',
              'base_mert':'MERT-330M 2026 adapted probe','repeats':a.repeats,
              'calibration':'per-fold internal checkpoint songs; equal-song MED; L-BFGS-B',
              'alpha_bounds':[0.,1.],'scale_bounds':[.25,2.],
              'validation_reuse':'internal songs also selected base checkpoints',
              'outer_labels_used_for_calibration':False,
              'clipping':False,'fixed_scale':a.fixed_scale,
              'frozen_splits_sha256':sha(matrix/'frozen_splits.json')}
    out.mkdir(parents=True,exist_ok=True)
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Pilot protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    for split in splits:
        fold=split['fold'];inner=split['checkpoint_ids'];outer=split['outer_ids']
        mert_checkpoint=teacher/f'fold_{fold}'/'selected_checkpoint.pt'
        teacher_cache=out/'teacher_internal'/f'fold_{fold}'
        if {p.stem for p in teacher_cache.glob('*.npz')}!=set(inner):
            teacher_cache.mkdir(parents=True,exist_ok=True)
            pp=mert_internal(index,inner,mert_checkpoint,features,a.device)
            for sid,pred in pp.items():np.savez_compressed(teacher_cache/f'{sid}.npz',pred=pred)
        mert_val=predictions_on_disk(teacher_cache,inner)
        mert_outer=predictions_on_disk(teacher/f'fold_{fold}'/'predictions',outer)
        for repeat in a.repeats:
            base=matrix/f'repeat_{repeat}'/'train_shape'/f'fold_{fold}'
            result=out/f'repeat_{repeat}'/f'fold_{fold}'
            if (result/'validation_metrics.json').is_file():
                print(f'SKIP repeat={repeat} fold={fold}',flush=True);continue
            b2_val=b2_internal(index,inner,base/'select_shape'/'selected_checkpoint.pt',
                               base/'normalizer.npz',a.device)
            params,detail=fit_calibrator(inner,b2_val,mert_val,index,a.fixed_scale)
            b2_outer=predictions_on_disk(base/'select_shape'/'predictions',outer)
            pred={sid:fuse(b2_outer[sid],mert_outer[sid],params).astype(np.float32) for sid in outer}
            result_root(result)
            metadata={'repeat':repeat,'fold':fold,'seed':json.loads((base/'select_shape'/'validation_metrics.json').read_text())['metadata']['seed'],
                      'alpha_valence':float(params[0]),'alpha_arousal':float(params[1]),
                      'scale_valence':float(params[2]),'scale_arousal':float(params[3]),
                      'calibration':detail,'inner_ids':inner,
                      'outer_labels_used_for_calibration':False,
                      'b2_checkpoint_sha256':sha(base/'select_shape'/'selected_checkpoint.pt'),
                      'mert_checkpoint_sha256':sha(mert_checkpoint)}
            metrics=save_report(result,pred,index,outer,metadata)
            (result/'calibration.json').write_text(json.dumps(metadata,indent=2))
            print(f'B3-C repeat={repeat} fold={fold} MED={metrics["med"]:.6f} alpha={params[:2]} scale={params[2:]}',flush=True)

if __name__=='__main__':main()
