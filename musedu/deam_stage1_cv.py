"""Five song-disjoint DEAM outer folds shared by B2 and three competitors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .experiment import load_manifest, on_device, pack, read_npz, seed_everything
from .matched_development import Songs, train_normalizer, make_model, loss_for, selector, output_mean
from .baselines import choose_view
from .model import _equal_song_huber, centered_correlation_loss
from torch import nn
from torch.nn import functional as F
from .v2_experiment import LengthBuckets
from .published_baselines.io import load_index, load_song, save_report, result_root, evaluate

METHODS=('B2','mert_probe','opensmile_lstm','conformer')
EXTRA_METHODS=('B2_point','mert_linear','opensmile_mlp','librosa_xgboost','librosa_rf',
               'opensmile_librosa_ridge_adapted')
ALL_METHODS=METHODS+EXTRA_METHODS
VIEWS={'mert_probe':('mert','probe'),'opensmile_lstm':('opensmile','lstm'),
       'conformer':('mel','conformer'),'opensmile_mlp':('opensmile','mlp'),
       'librosa_xgboost':('librosa','xgboost'),'librosa_rf':('librosa','rf')}
FOLD_SIZES=(360,360,360,360,362)


def make_protocol(ids,seed=2026,epochs=25):
    ids=list(map(str,ids))
    if len(ids)!=1802 or len(set(ids))!=1802:raise ValueError('Expected 1,802 unique DEAM songs')
    order=np.random.default_rng(seed).permutation(np.asarray(sorted(ids),dtype=object)).tolist()
    folds=[];start=0
    for size in FOLD_SIZES:
        folds.append(order[start:start+size]);start+=size
    assert start==len(order)
    return {'dataset':'DEAM','scope':'all 1,802 recordings','split_seed':seed,
            'outer_fold_sizes':list(FOLD_SIZES),'folds':folds,'epochs':epochs,
            'internal_checkpoint_fraction':0.1,'patience':5,
            'checkpoint_selector':'equal-song Huber + 0.02 stable centered correlation',
            'methods':list(METHODS),'prediction_step_seconds':0.5}


def internal_split(train_ids,seed):
    shuffled=list(train_ids)
    np.random.default_rng(seed).shuffle(shuffled)
    nstop=max(1,int(round(.1*len(shuffled))))
    stop=shuffled[:nstop];fit=shuffled[nstop:]
    assert set(fit).isdisjoint(stop)
    return fit,stop


def run_b2(df,index,fit_ids,stop_ids,eval_ids,seed,epochs,device,batch_size,
           objective='point_shape',predict_lookup=None,checkpoint_path=None,
           shape_weight=.02,view='fusion',framewise=False):
    seed_everything(seed)
    norm=train_normalizer(df,fit_ids,view)
    train=Songs(df,fit_ids,norm,view)
    stop=Songs(df,stop_ids,norm,view)
    tr=DataLoader(train,batch_sampler=LengthBuckets(train.lengths,batch_size,True,seed),collate_fn=pack)
    va=DataLoader(stop,batch_sampler=LengthBuckets(stop.lengths,batch_size,False,seed),collate_fn=pack)
    model=(FramewiseMUSEDU(len(norm[0]),96) if framewise else make_model('v1',len(norm[0]),96)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=float('inf');best_state=None;selected=0;bad=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for data in tr:
            batch=on_device(data,device)
            optimizer.zero_grad(set_to_none=True)
            output=model(batch['x'],batch['times'],batch['valid'])
            loss=(loss_for(output,batch,objective) if shape_weight==.02 else
                  _equal_song_huber(output_mean(output),batch['mean'],batch['mask'])+
                  shape_weight*centered_correlation_loss(output_mean(output),batch['mean'],batch['mask']))
            if not torch.isfinite(loss):raise FloatingPointError('B2 nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach().cpu()))
        model.eval();scores=[]
        with torch.inference_mode():
            for data in va:
                batch=on_device(data,device)
                # Select by an equal-song average, independent of batch length.
                for j in range(len(data['x'])):
                    one={k:(v[j:j+1] if torch.is_tensor(v) else v) for k,v in batch.items()}
                    scores.append(float(selector(model(one['x'],one['times'],one['valid']),one).cpu()))
        score=float(np.mean(scores))
        history.append({'epoch':epoch,'train_objective':float(np.mean(losses)),
                        'checkpoint_selector':score})
        print(f'B2 seed={seed} epoch={epoch} selector={score:.6f}',flush=True)
        if score<best-1e-7:
            best=score;selected=epoch;bad=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=5:break
    model.load_state_dict(best_state);model.eval()
    if checkpoint_path is not None:
        torch.save({'state_dict':best_state,'normalizer_mean':norm[0],
                    'normalizer_std':norm[1],'architecture':'v1','hidden':96,
                    'objective':objective,'seed':seed,'selected_epoch':selected},checkpoint_path)
    pred={};lookup=predict_lookup if predict_lookup is not None else dict(zip(df.song_id,df.file))
    with torch.inference_mode():
        for sid in eval_ids:
            z=read_npz(lookup[sid])
            x=choose_view(z,view)
            x=(x-norm[0])/norm[1]
            xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
            tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device)
            valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
            pred[sid]=output_mean(model(xx,tt,valid))[0].cpu().numpy().astype(np.float32)
    detail={'training_objective':objective,'shape_weight':shape_weight,'view':view,
            'framewise':framewise,'best_epoch':selected,'best_selector':best,'history':history,
            'stopped_early':len(history)<epochs,
            'improving_at_cap':len(history)==epochs and selected>=epochs-2}
    return pred,detail


class FramewiseMUSEDU(nn.Module):
    """Fusion projection and heads without cross-frame operations or time encoding."""
    def __init__(self,in_features,hidden=96):
        super().__init__()
        self.projection=nn.Sequential(nn.Linear(in_features,hidden),nn.LayerNorm(hidden))
        self.local=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),
                                 nn.Linear(hidden,hidden),nn.GELU())
        self.context=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),
                                   nn.Linear(hidden,hidden),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(hidden*2,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.mean_head=nn.Linear(hidden,2)
        self.dis_head=nn.Linear(hidden,2)

    def forward(self,x,time,valid):
        z=self.projection(x)
        h=self.fuse(torch.cat((self.local(z),self.context(z)),-1))
        return {'mean':torch.tanh(self.mean_head(h)),
                'tau':F.softplus(self.dis_head(h))+1e-4,'hidden':h}


def aggregate(root,method,index,all_ids):
    folder=root/method
    predictions={};rows=[]
    for i in range(1,6):
        fold_dir=folder/f'fold_{i}'
        table=pd.read_csv(fold_dir/'validation_by_song.csv',dtype={'song_id':str})
        table.insert(0,'fold',i)
        rows.append(table)
        for path in (fold_dir/'predictions').glob('*.npz'):
            with np.load(path,allow_pickle=False) as z:predictions[path.stem]=z['pred']
    if len(predictions)!=1802 or set(predictions)!=set(all_ids):
        raise AssertionError(f'{method}: incomplete or duplicate outer predictions')
    pd.concat(rows,ignore_index=True).to_csv(folder/'out_of_fold_by_song.csv',index=False)
    _,metrics=evaluate(predictions,index,all_ids)
    (folder/'crossval_summary.json').write_text(json.dumps({'method':method,'songs':1802,
        'outer_folds':5,'all_songs_scored_once':True,'metrics':metrics},indent=2))
    return metrics


def run(manifest,reference_run,features_dir,out,methods=METHODS,seed=2026,epochs=25,
        device='cuda',batch_size=8,fold_numbers=(1,2,3,4,5)):
    from .published_baselines.run import run_neural, run_classical, adapted_stack
    from .published_baselines.linear_probe import run_mert_linear
    methods=tuple(methods)
    if not set(methods)<=set(ALL_METHODS):raise ValueError('Unknown method')
    if any(i not in range(1,6) for i in fold_numbers):raise ValueError('Bad fold number')
    index,_=load_index(manifest,reference_run)
    df=load_manifest(manifest)
    all_ids=sorted(index)
    protocol=make_protocol(all_ids,seed,epochs)
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    p=root/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Existing protocol differs')
    p.write_text(json.dumps(protocol,indent=2))
    for fold in fold_numbers:
        outer=protocol['folds'][fold-1]
        training=[sid for j,part in enumerate(protocol['folds']) if j!=fold-1 for sid in part]
        fit,stop=internal_split(training,seed+fold)
        if set(outer)&(set(fit)|set(stop)):raise AssertionError('Outer label leakage')
        for method in methods:
            dest=root/method/f'fold_{fold}'
            if (dest/'validation_metrics.json').is_file():
                print(f'skip complete {method} fold={fold}',flush=True);continue
            print(f'START {method} fold={fold} fit={len(fit)} checkpoint={len(stop)} outer={len(outer)}',flush=True)
            result_root(dest)
            model_seed=seed+fold
            if method in ('B2','B2_point'):
                objective='point' if method=='B2_point' else 'point_shape'
                pp,detail=run_b2(df,index,fit,stop,outer,model_seed,epochs,device,batch_size,objective)
            elif method=='mert_linear':
                pp,detail=run_mert_linear(index,fit,stop,outer,features_dir)
            elif method=='opensmile_librosa_ridge_adapted':
                pp,detail,oof=adapted_stack(index,training,outer,features_dir,model_seed,epochs,
                                            device,batch_size,folds=3)
                oof_dir=dest/'training_oof';oof_dir.mkdir()
                for sid,x in oof.items():
                    np.savez_compressed(oof_dir/f'{sid}.npz',times=load_song(index,sid)['times'],pred=x)
            else:
                view,kind=VIEWS[method]
                if kind in ('xgboost','rf'):
                    pp,detail,_=run_classical(index,fit,outer,view,kind,features_dir,seed=model_seed)
                else:
                    pp,detail,_=run_neural(index,fit,stop,outer,view,kind,features_dir,
                                           seed=model_seed,epochs=epochs,device=device,batch_size=batch_size)
            metadata={'method':method,'fold':fold,'model_seed':model_seed,
                      'fit_song_count':len(fit),'checkpoint_song_count':len(stop),
                      'outer_song_count':len(outer),'detail':detail,
                      'outer_labels_used_for_selection':False,'outer_labels_used_for_training':False}
            metrics=save_report(dest,pp,index,outer,metadata)
            print(f'DONE {method} fold={fold} metrics={metrics}',flush=True)
    for method in methods:
        if all((root/method/f'fold_{i}'/'validation_metrics.json').is_file() for i in range(1,6)):
            print(f'AGGREGATE {method} {aggregate(root,method,index,all_ids)}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--features-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--methods',nargs='+',choices=ALL_METHODS,default=list(METHODS))
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    run(a.manifest,a.reference_run,a.features_dir,a.out,a.methods,a.seed,a.epochs,
        a.device,a.batch_size,a.folds)


if __name__=='__main__':main()
