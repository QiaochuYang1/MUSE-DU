"""Strictly disjoint fit/checkpoint/calibration/outer-fold confirmation of B3-C."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .deam_stage1_cv import run_b2
from .experiment import load_manifest
from .published_baselines.io import load_index,load_song,result_root,save_report,evaluate
from .published_baselines.run import run_neural
from .b3_level_fusion import fit_calibrator,fuse
from .plan_matrix_2x2 import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--matrix-root',required=True);p.add_argument('--features',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--repeat',type=int,default=0,choices=(0,1,2),
                   help='Independent initialization repeat; repeat 0 reproduces the archived run')
    a=p.parse_args();root=Path(a.out).resolve();matrix=Path(a.matrix_root).resolve()
    index,_=load_index(a.manifest,a.reference_run);df=load_manifest(a.manifest)
    splits=json.loads((matrix/'frozen_splits.json').read_text())['folds']
    protocol={'name':'B3-C nested confirmation','outer_folds':5,
              'split':'original checkpoint IDs 144; 144 calibration songs reserved from original fit IDs using seed 5000+fold',
              'b2':'B2 shape loss 0.02, 25 epoch cap, patience 5, shared shape selector',
              'mert':'frozen MERT-330M weighted layer MLP, Adam 0.001, effective batch 64, 50 epoch cap, patience 15',
              'calibration':'four parameters bounded alpha 0..1 and scale 0.25..2, equal-song MED on disjoint calibration songs',
              'outer_labels_used_for_fit_selection_or_calibration':False,
              'split_sha256':sha(matrix/'frozen_splits.json')}
    if a.repeat:
        protocol['repeat']=a.repeat
        protocol['seed_rule']='2026 + fold + 1000 * repeat'
    root.mkdir(parents=True,exist_ok=True)
    protocol_file=root/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text())!=protocol:
        raise ValueError('Nested protocol changed')
    protocol_file.write_text(json.dumps(protocol,indent=2))
    for split in splits:
        fold=split['fold'];outer=split['outer_ids'];stop=split['checkpoint_ids']
        shuffled=np.array(split['fit_ids'],dtype=object)
        np.random.default_rng(5000+fold).shuffle(shuffled)
        cal=shuffled[:144].tolist();fit=shuffled[144:].tolist()
        parts=(fit,stop,cal,outer)
        if (len(set(fit)|set(stop)|set(cal)|set(outer))!=1802 or
            any(set(x)&set(y) for i,x in enumerate(parts) for y in parts[i+1:])):
            raise AssertionError('Nested song split overlap or omission')
        fold_root=root/f'fold_{fold}';fold_root.mkdir(exist_ok=True)
        split_file=fold_root/'split.json'
        content={'fit_ids':fit,'checkpoint_ids':stop,'calibration_ids':cal,'outer_ids':outer}
        if split_file.exists() and json.loads(split_file.read_text())!=content:
            raise ValueError('Nested split changed')
        split_file.write_text(json.dumps(content,indent=2))
        if (fold_root/'B3_C'/'evaluation'/'validation_metrics.json').exists():
            print(f'SKIP fold={fold}',flush=True);continue
        b2_dir=fold_root/'B2';b2_dir.mkdir(exist_ok=True)
        mert_dir=fold_root/'MERT';mert_dir.mkdir(exist_ok=True)
        seed=2026+fold+1000*a.repeat
        b2_pred,b2_detail=run_b2(df,index,fit,stop,cal+outer,seed,25,a.device,8,
                                  objective='point_shape',checkpoint_path=b2_dir/'selected_checkpoint.pt')
        mert_pred,mert_detail,_=run_neural(index,fit,stop,cal+outer,'mert','probe',
                                           a.features,seed=seed,epochs=50,device=a.device,
                                           batch_size=64,lr=.001,patience=15,
                                           checkpoint_path=mert_dir/'selected_checkpoint.pt',
                                           optimizer_kind='Adam',cache_in_memory=True,
                                           microbatch_size=16)
        params,cal_detail=fit_calibrator(cal,b2_pred,mert_pred,index)
        cal_dir=fold_root/'calibration_predictions';cal_dir.mkdir(exist_ok=True)
        for sid in cal:
            z=load_song(index,sid)
            np.savez_compressed(cal_dir/f'{sid}.npz',b2=b2_pred[sid],mert=mert_pred[sid],
                                mean=z['mean'],mask=z['mask'],times=z['times'])
        metadata={'fold':fold,'seed':seed,'fit_songs':len(fit),'checkpoint_songs':len(stop),
                  'calibration_songs':len(cal),'outer_songs':len(outer),
                  'alpha_valence':float(params[0]),'alpha_arousal':float(params[1]),
                  'scale_valence':float(params[2]),'scale_arousal':float(params[3]),
                  'calibration':cal_detail,'b2_epoch':b2_detail['best_epoch'],
                  'mert_epoch':mert_detail['best_epoch'],
                  'b2_checkpoint_sha256':sha(b2_dir/'selected_checkpoint.pt'),
                  'mert_checkpoint_sha256':sha(mert_dir/'selected_checkpoint.pt'),
                  'outer_labels_used_for_fit_selection_or_calibration':False}
        b3={sid:fuse(b2_pred[sid],mert_pred[sid],params).astype(np.float32) for sid in outer}
        for name,pred in (('B2',b2_pred),('MERT',mert_pred),('B3_C',b3)):
            destination=fold_root/name/'evaluation';result_root(destination)
            selected={sid:pred[sid] for sid in outer}
            save_report(destination,selected,index,outer,metadata)
        (fold_root/'calibration.json').write_text(json.dumps(metadata,indent=2))
        print(f'NESTED fold={fold} B2={evaluate({sid:b2_pred[sid] for sid in outer},index,outer)[1]["med"]:.6f} '
              f'MERT={evaluate({sid:mert_pred[sid] for sid in outer},index,outer)[1]["med"]:.6f} '
              f'B3={evaluate(b3,index,outer)[1]["med"]:.6f}',flush=True)

if __name__=='__main__':main()
