"""Five-fold B2 shape-weight and component sensitivity on frozen DEAM splits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .deam_stage1_cv import make_protocol, internal_split, run_b2, aggregate
from .experiment import load_manifest
from .published_baselines.io import load_index, result_root, save_report

VARIANTS={
    'shape_0p005':dict(shape_weight=.005),
    'shape_0p01':dict(shape_weight=.01),
    'shape_0p05':dict(shape_weight=.05),
    'shape_0p1':dict(shape_weight=.1),
    'mert_only':dict(view='mert'),
    'acoustic_only':dict(view='acoustic'),
    'framewise_fusion':dict(framewise=True),
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    index,_=load_index(a.manifest,a.reference_run)
    df=load_manifest(a.manifest)
    ids=sorted(index)
    protocol=make_protocol(ids,2026,a.epochs)
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    pf=root/'protocol.json'
    if pf.exists() and json.loads(pf.read_text())!=protocol:
        raise ValueError('Protocol does not match frozen five-fold split')
    pf.write_text(json.dumps(protocol,indent=2))
    for name in a.variants:
        params=VARIANTS[name]
        for fold in a.folds:
            dest=root/name/f'fold_{fold}'
            if (dest/'validation_metrics.json').exists():
                print(f'skip {name} fold={fold}',flush=True);continue
            outer=protocol['folds'][fold-1]
            training=[sid for i,part in enumerate(protocol['folds']) if i!=fold-1 for sid in part]
            fit,stop=internal_split(training,2026+fold)
            if set(outer)&(set(fit)|set(stop)):raise AssertionError('Outer fold leakage')
            result_root(dest)
            print(f'START {name} fold={fold}',flush=True)
            pred,detail=run_b2(df,index,fit,stop,outer,2026+fold,a.epochs,a.device,
                               a.batch_size,**params)
            metadata={'method':name,'fold':fold,'fit_song_count':len(fit),
                      'checkpoint_song_count':len(stop),'outer_song_count':len(outer),
                      'detail':detail,'outer_labels_used_for_selection':False,
                      'outer_labels_used_for_training':False,
                      'checkpoint_selector':'equal-song Huber + 0.02 stable centered correlation'}
            metrics=save_report(dest,pred,index,outer,metadata)
            print(f'DONE {name} fold={fold} MED={metrics["med"]:.6f}',flush=True)
        if all((root/name/f'fold_{i}'/'validation_metrics.json').exists() for i in range(1,6)):
            print(f'AGGREGATE {name} {aggregate(root,name,index,ids)}',flush=True)

if __name__=='__main__':main()
