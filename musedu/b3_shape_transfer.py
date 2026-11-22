"""Exploratory fixed-parameter transfer of B3-C calibration to prior shape-0.05 B2 ablation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .b3_level_fusion import fuse
from .published_baselines.io import load_index,result_root,save_report,evaluate

def load(folder,ids):
    if {p.stem for p in folder.glob('*.npz')}!=set(ids):raise AssertionError('Missing fold predictions')
    out={}
    for sid in ids:
        with np.load(folder/f'{sid}.npz',allow_pickle=False) as z:out[sid]=z['pred']
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--matrix-root',required=True);p.add_argument('--shape05-root',required=True)
    p.add_argument('--mert-root',required=True);p.add_argument('--calibration-root',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);matrix=Path(a.matrix_root)
    index,_=load_index(a.manifest,a.reference_run)
    splits=json.loads((matrix/'frozen_splits.json').read_text())['folds']
    fused_all={};shape_all={}
    for split in splits:
        fold=split['fold'];ids=split['outer_ids']
        shape=load(Path(a.shape05_root)/f'fold_{fold}'/'predictions',ids)
        mert=load(Path(a.mert_root)/f'fold_{fold}'/'predictions',ids)
        cal=json.loads((Path(a.calibration_root)/'repeat_0'/f'fold_{fold}'/'calibration.json').read_text())
        params=[cal['alpha_valence'],cal['alpha_arousal'],cal['scale_valence'],cal['scale_arousal']]
        fused={sid:fuse(shape[sid],mert[sid],params).astype(np.float32) for sid in ids}
        dest=out/f'fold_{fold}'
        if not (dest/'validation_metrics.json').exists():
            result_root(dest)
            save_report(dest,fused,index,ids,{'fold':fold,'parameters_from':'B3-C repeat-0 inner validation',
                'trajectory_model':'historical shape-weight-0.05 ablation',
                'parameters_refit_for_shape05':False,'outer_labels_used_for_calibration':False,
                'exploratory_due_to_prior_outer_result_inspection':True})
        fused_all.update(fused);shape_all.update(shape)
    if set(fused_all)!=set(index):raise AssertionError('Incomplete OOF set')
    result={'fused':evaluate(fused_all,index,sorted(index))[1],
            'shape05':evaluate(shape_all,index,sorted(index))[1]}
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({name:{k:v for k,v in metrics.items() if k in
          ('med','valence_within_pearson','arousal_within_pearson')}
          for name,metrics in result.items()},indent=2))

if __name__=='__main__':main()
