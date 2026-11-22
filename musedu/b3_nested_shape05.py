"""Exploratory nested B3-C variant with B2 shape weight 0.05 and fixed selector 0.02."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .b3_level_fusion import fit_calibrator,fuse
from .deam_stage1_cv import run_b2
from .experiment import load_manifest
from .published_baselines.io import load_index,result_root,save_report
from .plan_matrix_2x2 import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--nested-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();root=Path(a.out).resolve();nested=Path(a.nested_root).resolve()
    index,_=load_index(a.manifest,a.reference_run);df=load_manifest(a.manifest)
    protocol={'name':'nested shape-0.05 fusion exploratory variant',
              'shape_weight':.05,'checkpoint_shape_weight':.02,
              'prior_outer_results_informed_shape_weight_choice':True,
              'fit_checkpoint_calibration_outer_split':'identical to b3_nested_confirm',
              'nested_protocol_sha256':sha(nested/'protocol.json')}
    root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Shape-0.05 protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    for fold in range(1,6):
        split=json.loads((nested/f'fold_{fold}'/'split.json').read_text())
        fit=split['fit_ids'];stop=split['checkpoint_ids']
        cal=split['calibration_ids'];outer=split['outer_ids']
        destination=root/f'fold_{fold}'
        if (destination/'B3_C_05'/'evaluation'/'validation_metrics.json').exists():
            print(f'SKIP fold={fold}',flush=True);continue
        destination.mkdir(exist_ok=True)
        b2,detail=run_b2(df,index,fit,stop,cal+outer,2026+fold,25,a.device,8,
                         objective='point_shape',shape_weight=.05,
                         checkpoint_path=destination/'B2_05_selected_checkpoint.pt')
        teacher={}
        for sid in cal:
            with np.load(nested/f'fold_{fold}'/'calibration_predictions'/f'{sid}.npz',allow_pickle=False) as z:
                teacher[sid]=z['mert']
        for sid in outer:
            with np.load(nested/f'fold_{fold}'/'MERT'/'evaluation'/'predictions'/f'{sid}.npz',allow_pickle=False) as z:
                teacher[sid]=z['pred']
        params,calibration=fit_calibrator(cal,b2,teacher,index)
        fused={sid:fuse(b2[sid],teacher[sid],params).astype(np.float32) for sid in outer}
        metadata={'fold':fold,'seed':2026+fold,'training_shape_weight':.05,
                  'checkpoint_shape_weight':.02,'best_epoch':detail['best_epoch'],
                  'alpha_valence':float(params[0]),'alpha_arousal':float(params[1]),
                  'scale_valence':float(params[2]),'scale_arousal':float(params[3]),
                  'calibration':calibration,'b2_checkpoint_sha256':sha(destination/'B2_05_selected_checkpoint.pt'),
                  'outer_labels_used_for_calibration':False,
                  'prior_outer_results_informed_shape_weight_choice':True}
        for name,pred in (('B2_05',b2),('B3_C_05',fused)):
            output=destination/name/'evaluation';result_root(output)
            metrics=save_report(output,{sid:pred[sid] for sid in outer},index,outer,metadata)
            print(f'SHAPE05 fold={fold} {name} MED={metrics["med"]:.6f}',flush=True)
        (destination/'calibration.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
