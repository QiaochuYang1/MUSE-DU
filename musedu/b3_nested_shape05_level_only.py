"""Disjoint-calibration level-only fusion of the nested shape-0.05 trajectory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .b3_level_fusion import fit_calibrator,fuse
from .experiment import read_npz
from .matched_development import make_model,output_mean
from .published_baselines.io import load_index,evaluate,result_root,save_report

def infer_calibration(index,ids,checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    mu=saved['normalizer_mean'];sd=saved['normalizer_std']
    model=make_model('v1',len(mu),96).to(device)
    model.load_state_dict(saved['state_dict']);model.eval();out={}
    with torch.inference_mode():
        for sid in ids:
            z=read_npz(index[sid]);x=np.concatenate((z['acoustic'],z['embedding']),axis=-1)
            x=(x-mu)/sd
            xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
            tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device)
            valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
            out[sid]=output_mean(model(xx,tt,valid))[0].cpu().numpy().astype(np.float32)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--nested-root',required=True);p.add_argument('--shape05-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();nested=Path(a.nested_root);shape=Path(a.shape05_root);out=Path(a.out)
    index,_=load_index(a.manifest,a.reference_run);out.mkdir(parents=True,exist_ok=True)
    protocol={'name':'nested shape-0.05 level-only exploratory fusion',
              'trajectory_scale':1.,'alpha_bounds':[0.,1.],
              'fit_checkpoint_calibration_outer_disjoint':True,
              'shape_weight_previously_selected_after_outer_results':True}
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    pp={name:{} for name in ('B2_05','MERT','B3_05_level_only')}
    for fold in range(1,6):
        split=json.loads((nested/f'fold_{fold}'/'split.json').read_text())
        cal=split['calibration_ids'];outer=split['outer_ids']
        b2_cal=infer_calibration(index,cal,shape/f'fold_{fold}'/'B2_05_selected_checkpoint.pt',a.device)
        mert_cal={}
        for sid in cal:
            with np.load(nested/f'fold_{fold}'/'calibration_predictions'/f'{sid}.npz',allow_pickle=False) as z:
                mert_cal[sid]=z['mert']
        params,detail=fit_calibrator(cal,b2_cal,mert_cal,index,scale_fixed=1.)
        b2_outer={};mert_outer={}
        for sid in outer:
            with np.load(shape/f'fold_{fold}'/'B2_05'/'evaluation'/'predictions'/f'{sid}.npz',allow_pickle=False) as z:
                b2_outer[sid]=z['pred']
            with np.load(nested/f'fold_{fold}'/'MERT'/'evaluation'/'predictions'/f'{sid}.npz',allow_pickle=False) as z:
                mert_outer[sid]=z['pred']
        fused={sid:fuse(b2_outer[sid],mert_outer[sid],params).astype(np.float32) for sid in outer}
        destination=out/f'fold_{fold}'
        if not (destination/'validation_metrics.json').exists():
            result_root(destination)
            save_report(destination,fused,index,outer,
                        {'fold':fold,'alpha_valence':float(params[0]),
                         'alpha_arousal':float(params[1]),'scale_valence':1.,'scale_arousal':1.,
                         'calibration':detail,'outer_labels_used_for_calibration':False,
                         'shape_weight_previously_selected_after_outer_results':True})
        pp['B2_05'].update(b2_outer);pp['MERT'].update(mert_outer)
        pp['B3_05_level_only'].update(fused)
        print(f'LEVEL05 fold={fold} MED={evaluate(fused,index,outer)[1]["med"]:.6f}',flush=True)
    rows=[];tables={}
    for name,pred in pp.items():
        if set(pred)!=set(index):raise AssertionError('Incomplete OOF set')
        table,metrics=evaluate(pred,index,sorted(index));tables[name]=table.set_index('song_id')
        rows.append({'model':name,'songs':1802,**metrics})
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    rng=np.random.default_rng(2026);ids=sorted(index);boot=[]
    for other in ('B2_05','MERT'):
        for metric in ('med','valence_ccc','arousal_ccc','valence_within_pearson','arousal_within_pearson'):
            delta=(tables['B3_05_level_only'].loc[ids,metric]-tables[other].loc[ids,metric]).to_numpy(dtype=float)
            draws=rng.integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
            sample=np.nanmean(delta[draws],axis=1)
            boot.append({'contrast':'B3_05_level_only minus '+other,'metric':metric,
                         'difference':float(np.nanmean(delta)),
                         'ci95_lower':float(np.quantile(sample,.025)),
                         'ci95_upper':float(np.quantile(sample,.975)),
                         'bootstrap_draws':10000})
    pd.DataFrame(boot).to_csv(out/'paired_bootstraps.csv',index=False)
    print(pd.DataFrame(rows)[['model','med','valence_within_pearson','arousal_within_pearson','valence_ccc','arousal_ccc']].to_string(index=False))

if __name__=='__main__':main()
