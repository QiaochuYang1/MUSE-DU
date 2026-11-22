"""Full-budget 2x2 training-loss/checkpoint-selection experiment on frozen DEAM folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .deam_stage1_cv import internal_split, aggregate
from .experiment import load_manifest, on_device, pack, read_npz, seed_everything
from .matched_development import Songs, train_normalizer, make_model, output_mean
from .model import _equal_song_huber, centered_correlation_loss
from .v2_experiment import LengthBuckets
from .published_baselines.io import load_index, result_root, save_report

ORIGINAL=Path(r'E:\deam_cv5_stage1')
TRAIN_WEIGHTS={'point':0.,'shape':.02}
SELECT_WEIGHTS={'point':0.,'shape':.02}
REPEATS=((2027,2028,2029,2030,2031),
         (2032,2033,2034,2035,2036),
         (2037,2038,2039,2040,2041))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare(manifest,reference_run,out,original_run=ORIGINAL):
    original_run=Path(original_run)
    index,_=load_index(manifest,reference_run)
    df=load_manifest(manifest)
    protocol=json.loads((original_run/'protocol.json').read_text())
    folds=protocol['folds']
    ids=sorted(index)
    if len(ids)!=1802 or len(set(ids))!=1802 or set().union(*(set(x) for x in folds))!=set(ids):
        raise AssertionError('Invalid original outer folds')
    if len(set().union(*(set(x) for x in folds)))!=sum(map(len,folds)):
        raise AssertionError('Overlapping original outer folds')
    frozen=[]
    for fold in range(1,6):
        outer=folds[fold-1]
        training=[sid for j,part in enumerate(folds) if j!=fold-1 for sid in part]
        fit,stop=internal_split(training,2026+fold)
        if set(outer)&(set(fit)|set(stop)) or set(fit)&set(stop):
            raise AssertionError('Split overlap')
        frozen.append({'fold':fold,'outer_ids':outer,'fit_ids':fit,'checkpoint_ids':stop,
                       'initialization_seeds':[group[fold-1] for group in REPEATS]})
        for method in ('B2','B2_point'):
            folder=original_run/method/f'fold_{fold}'/'predictions'
            if {p.stem for p in folder.glob('*.npz')}!=set(outer):
                raise AssertionError(f'Historical predictions incomplete: {method} fold {fold}')
    out.mkdir(parents=True,exist_ok=True)
    target=out/'frozen_splits.json'
    content={'source_protocol_sha256':sha(original_run/'protocol.json'),
             'deam_manifest_sha256':sha(manifest),'source_manifest_sha256':
             sha(Path(reference_run)/'source_manifest.json'),
             'folds':frozen}
    if target.exists() and json.loads(target.read_text())!=content:
        raise ValueError('Frozen split file changed')
    target.write_text(json.dumps(content,indent=2))
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
                 'cuda_available':torch.cuda.is_available(),
                 'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 'mert_repository':'m-a-p/MERT-v1-95M',
                 'mert_revision':'12af15fef9d0ac838c3f475bfbbf26d2060dd4f5',
                 'code_sha256':{name:sha(Path(__file__).parent/name) for name in
                                 ('model.py','deam_stage1_cv.py','plan_matrix_2x2.py')}}
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    return df,index,frozen

def fit_one(df,index,split,seed,objective,out,epochs,device,batch_size):
    seed_everything(seed)
    norm=train_normalizer(df,split['fit_ids'],'fusion')
    train=Songs(df,split['fit_ids'],norm,'fusion')
    val=Songs(df,split['checkpoint_ids'],norm,'fusion')
    dl=DataLoader(train,batch_sampler=LengthBuckets(train.lengths,batch_size,True,seed),collate_fn=pack)
    dv=DataLoader(val,batch_sampler=LengthBuckets(val.lengths,batch_size,False,seed),collate_fn=pack)
    model=make_model('v1',len(norm[0]),96).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'normalizer.npz',mean=norm[0],std=norm[1])
    epoch_dir=out/'epoch_checkpoints';epoch_dir.mkdir(exist_ok=True)
    history=[]
    for epoch in range(1,epochs+1):
        model.train();train_loss=[]
        for data in dl:
            b=on_device(data,device)
            optimizer.zero_grad(set_to_none=True)
            pred=output_mean(model(b['x'],b['times'],b['valid']))
            point=_equal_song_huber(pred,b['mean'],b['mask'])
            loss=point+TRAIN_WEIGHTS[objective]*centered_correlation_loss(pred,b['mean'],b['mask'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite B2 loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();train_loss.append(float(loss.detach().cpu()))
        model.eval();points=[];shapes=[]
        with torch.inference_mode():
            for data in dv:
                b=on_device(data,device)
                for j in range(len(data['x'])):
                    one={k:(v[j:j+1] if torch.is_tensor(v) else v) for k,v in b.items()}
                    pred=output_mean(model(one['x'],one['times'],one['valid']))
                    points.append(float(_equal_song_huber(pred,one['mean'],one['mask']).cpu()))
                    shapes.append(float(centered_correlation_loss(pred,one['mean'],one['mask']).cpu()))
        point_score=float(np.mean(points));shape_score=point_score+.02*float(np.mean(shapes))
        history.append({'epoch':epoch,'train_objective':float(np.mean(train_loss)),
                        'validation_point':point_score,'validation_shape':shape_score})
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'epoch':epoch,'seed':seed,'training_loss':objective},
                   epoch_dir/f'epoch_{epoch:02d}.pt')
        print(f'FIT seed={seed} train={objective} epoch={epoch}/{epochs} point={point_score:.6f} shape={shape_score:.6f}',flush=True)
    (out/'history.json').write_text(json.dumps(history,indent=2))
    lookup=dict(zip(df.song_id,df.file))
    choices={rule:min(history,key=lambda row:(row['validation_'+rule],row['epoch']))
             for rule in SELECT_WEIGHTS}
    for rule,chosen in choices.items():
        destination=out/f'select_{rule}'
        if (destination/'validation_metrics.json').is_file():continue
        result_root(destination)
        epoch=chosen['epoch']
        state=torch.load(epoch_dir/f'epoch_{epoch:02d}.pt',map_location=device,weights_only=True)
        model.load_state_dict(state['state_dict']);model.eval()
        predictions={}
        with torch.inference_mode():
            for sid in split['outer_ids']:
                z=read_npz(lookup[sid])
                x=(np.concatenate((z['acoustic'],z['embedding']),axis=-1)-norm[0])/norm[1]
                xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
                tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device)
                valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
                predictions[sid]=output_mean(model(xx,tt,valid))[0].cpu().numpy().astype(np.float32)
        metadata={'training_loss':objective,'selection_loss':rule,'seed':seed,
                  'fold':split['fold'],'selected_epoch':epoch,
                  'selected_validation_score':chosen['validation_'+rule],
                  'training_epochs':epochs,'early_stopping':False,
                  'fit_song_count':len(split['fit_ids']),
                  'checkpoint_song_count':len(split['checkpoint_ids']),
                  'outer_song_count':len(split['outer_ids']),
                  'outer_labels_used_for_selection':False,
                  'checkpoint_sha256':sha(epoch_dir/f'epoch_{epoch:02d}.pt')}
        metrics=save_report(destination,predictions,index,split['outer_ids'],metadata)
        (destination/'selected_checkpoint.pt').write_bytes(
            (epoch_dir/f'epoch_{epoch:02d}.pt').read_bytes())
        print(f'SCORED seed={seed} fold={split["fold"]} {objective}/{rule} epoch={epoch} MED={metrics["med"]:.6f}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--original-run',default=str(ORIGINAL))
    p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.epochs!=25:raise ValueError('Frozen diagnostic requires exactly 25 epochs')
    root=Path(a.out).resolve()
    if os.name=='nt' and len(str(root/'repeat_2'/'train_shape'/'fold_5'/'select_shape'/'predictions'/'9999.npz'))>=240:
        raise ValueError('Output path too long for Windows; choose a short --out path')
    df,index,splits=prepare(a.manifest,a.reference_run,root,a.original_run)
    config={'full_epochs':25,'early_stopping':False,'batch_size':a.batch_size,
            'optimizer':'AdamW','learning_rate':3e-4,'weight_decay':.01,
            'gradient_clip_norm':1.,'huber_delta':.1,
            'training_weights':TRAIN_WEIGHTS,'selection_weights':SELECT_WEIGHTS,
            'tie_break':'earliest epoch for equal score',
            'repeat_seeds':REPEATS,
            'historical_early_stopped_results_separate':True}
    config_file=root/'matrix_protocol.json'
    if config_file.exists() and json.loads(config_file.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Matrix protocol changed')
    config_file.write_text(json.dumps(config,indent=2))
    for repeat in a.repeats:
        if repeat not in range(3):raise ValueError('Repeat must be 0,1,2')
        for split in splits:
            fold=split['fold']
            if fold not in a.folds:continue
            seed=REPEATS[repeat][fold-1]
            for objective in TRAIN_WEIGHTS:
                run=root/f'repeat_{repeat}'/f'train_{objective}'/f'fold_{fold}'
                if all((run/f'select_{rule}'/'validation_metrics.json').is_file()
                       for rule in SELECT_WEIGHTS):
                    print(f'SKIP repeat={repeat} fold={fold} train={objective}',flush=True);continue
                if (run/'history.json').exists() and len(list((run/'epoch_checkpoints').glob('epoch_*.pt')))==25:
                    raise RuntimeError(f'Incomplete selection output after completed fit: {run}; inspect before rerun')
                fit_one(df,index,split,seed,objective,run,a.epochs,a.device,a.batch_size)

if __name__=='__main__':main()
