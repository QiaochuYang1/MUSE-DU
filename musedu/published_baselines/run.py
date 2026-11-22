"""Reproduce paper-inspired reference baselines with matched MUSE-DU development splits.

No access to scale, calibration, or test labels is performed. Results are exploratory.
Use: python -m musedu.published_baselines.run --help
"""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from .io import (seed_all,load_index,load_song,get_features,train_normalizer,normalized,
                 evaluate,result_root,save_report)
from .models import create_model

METHODS={
 'opensmile_lstm':('opensmile','lstm'),
 'librosa_xgboost':('librosa','xgboost'),
 'librosa_rf':('librosa','rf'),
 'opensmile_mlp':('opensmile','mlp'),
 'hybrid_ridge':('multi','stack'),
 'opensmile_librosa_ridge_adapted':('multi','adapted_stack'),
 'mert_probe':('mert','probe'),
 'conformer':('mel','conformer'),
 'hubert_probe':('hubert','probe'),
}
STACK_FAMILIES=('opensmile','librosa','paa')
STACK_MODELS=('lstm','mlp','rf','xgboost')
ADAPTED_STACK_MEMBERS=(('opensmile','lstm'),('opensmile','mlp'),
                       ('librosa','xgboost'),('librosa','rf'))

class Songs(Dataset):
    def __init__(self,index,ids,view,featroot,normalizer,cache_in_memory=False):
        self.index=index;self.ids=list(ids);self.view=view;self.root=featroot
        self.mean,self.std=normalizer
        self.cached=[self._load(sid) for sid in self.ids] if cache_in_memory else None
    def __len__(self):return len(self.ids)
    def _load(self,sid):
        x,z=get_features(self.index,sid,self.view,self.root)
        return {'sid':sid,'x':normalized(x,self.mean,self.std),
                'y':np.nan_to_num(z['mean'],nan=0.).astype('float32'),
                'mask':z['mask'].astype(bool)}
    def __getitem__(self,i):
        return self.cached[i] if self.cached is not None else self._load(self.ids[i])


def batchify(rows):
    n=len(rows);T=max(len(x['x']) for x in rows)
    dim=rows[0]['x'].shape[1:]
    x=np.zeros((n,T,*dim),np.float32);y=np.zeros((n,T,2),np.float32)
    mask=np.zeros((n,T,2),bool);valid=np.zeros((n,T),bool)
    for j,r in enumerate(rows):
        t=len(r['x']);x[j,:t]=r['x'];y[j,:t]=r['y'];mask[j,:t]=r['mask'];valid[j,:t]=1
    return {'x':torch.from_numpy(x),'y':torch.from_numpy(y),
            'mask':torch.from_numpy(mask),'valid':torch.from_numpy(valid)}


def normalizer_for(index,ids,view,feature_root):
    if view not in ('mert','hubert'):
        return train_normalizer(index,ids,view,feature_root)
    count=0;mean=None;m2=None
    for sid in ids:
        x,_=get_features(index,sid,view,feature_root)
        if x.ndim!=3:raise ValueError(view+' must have [T,L,D] Transformer features')
        if mean is None:
            mean=np.zeros(x.shape[1:],np.float64);m2=np.zeros(x.shape[1:],np.float64)
        if mean.shape!=x.shape[1:]:raise ValueError('Encoder layer mismatch')
        mean+=x.sum(axis=0,dtype=np.float64)
        m2+=(x.astype(np.float64)**2).sum(axis=0)
        count+=len(x)
    avg=mean/count; sd=np.maximum(np.sqrt(np.maximum(m2/count-avg**2,0)),1e-5)
    return avg.astype(np.float32),sd.astype(np.float32)


def loss_equal_song(pred,y,mask,kind='mse'):
    if kind=='mse':err=(pred-y)**2
    else:err=torch.nn.functional.huber_loss(pred,y,delta=.1,reduction='none')
    per=(err*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)
    return per.mean()


def val_selector(pred,y,mask):
    """User's common selector: equal-song Huber + 0.02 stable centered correlation."""
    from musedu.model import _equal_song_huber, centered_correlation_loss
    return _equal_song_huber(pred,y,mask)+.02*centered_correlation_loss(pred,y,mask)


def run_neural(index,fit_ids,stop_ids,predict_ids,view,kind,features_dir,*,seed,epochs,device,batch_size=8,lr=None,patience=5,checkpoint_path=None,optimizer_kind='AdamW',cache_in_memory=False,microbatch_size=None):
    seed_all(seed)
    mu,sd=normalizer_for(index,fit_ids,view,features_dir)
    train=Songs(index,fit_ids,view,features_dir,(mu,sd),cache_in_memory)
    val=Songs(index,stop_ids,view,features_dir,(mu,sd),cache_in_memory)
    dl=DataLoader(train,batch_size=batch_size,shuffle=True,collate_fn=batchify,num_workers=0)
    dv=DataLoader(val,batch_size=batch_size,shuffle=False,collate_fn=batchify,num_workers=0)
    example=train[0]['x'];model=create_model(kind,example.shape[1:]).to(device)
    if lr is None:lr=1e-3 if kind in ('lstm','mlp','probe') else 3e-4
    if optimizer_kind=='AdamW':
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    elif optimizer_kind=='Adam':
        opt=torch.optim.Adam(model.parameters(),lr=lr)
    else:
        raise ValueError(optimizer_kind)
    best=float('inf');selected=0;state=None;bad=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();train_loss=[]
        for b in dl:
            opt.zero_grad(set_to_none=True)
            if microbatch_size is None:
                x=b['x'].to(device);y=b['y'].to(device);m=b['mask'].to(device);v=b['valid'].to(device)
                pred=model(x,v)
                loss=loss_equal_song(pred,y,m)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();train_loss.append(float(loss.detach().cpu()))
            else:
                total=0.
                for start in range(0,len(b['x']),microbatch_size):
                    stop=min(start+microbatch_size,len(b['x']))
                    x=b['x'][start:stop].to(device);y=b['y'][start:stop].to(device)
                    m=b['mask'][start:stop].to(device);v=b['valid'][start:stop].to(device)
                    pred=model(x,v)
                    loss=loss_equal_song(pred,y,m)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    weight=(stop-start)/len(b['x'])
                    (loss*weight).backward();total+=float(loss.detach().cpu())*weight
                train_loss.append(total)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval();val_loss=[]
        with torch.inference_mode():
            for b in dv:
                if microbatch_size is None:
                    x=b['x'].to(device);y=b['y'].to(device);m=b['mask'].to(device);v=b['valid'].to(device)
                    val_loss.extend([float(val_selector(model(x[j:j+1],v[j:j+1]),y[j:j+1],m[j:j+1]).cpu()) for j in range(len(x))])
                else:
                    for j in range(len(b['x'])):
                        x=b['x'][j:j+1].to(device);y=b['y'][j:j+1].to(device)
                        m=b['mask'][j:j+1].to(device);v=b['valid'][j:j+1].to(device)
                        val_loss.append(float(val_selector(model(x,v),y,m).cpu()))
        score=float(np.mean(val_loss))
        history.append({'epoch':epoch,'train_mse':float(np.mean(train_loss)),'validation_selector':score})
        if score<best-1e-7:
            best=score;selected=epoch;state=copy.deepcopy(model.state_dict());bad=0
        else:
            bad+=1
            if bad>=patience:break
    model.load_state_dict(state);model.eval()
    if checkpoint_path is not None:
        torch.save({'state_dict':state,'normalizer_mean':mu,'normalizer_std':sd,
                    'view':view,'kind':kind,'seed':seed,'selected_epoch':selected},checkpoint_path)
    output={}
    with torch.inference_mode():
        for sid in predict_ids:
            x,_=get_features(index,sid,view,features_dir)
            z=normalized(x,mu,sd)
            # Predict a full song, preserving noncausal context.
            xt=torch.from_numpy(z[None]).to(device)
            valid=torch.ones((1,len(z)),dtype=torch.bool,device=device)
            output[sid]=model(xt,valid)[0].cpu().numpy().astype(np.float32)
    return output, {'best_epoch':selected,'best_selector':best,'history':history,
                    'stopped_early':len(history)<epochs,
                    'improving_at_cap':len(history)==epochs and selected>=epochs-2},(mu,sd)


def _estimator(kind,seed):
    if kind=='rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300,max_depth=20,
              min_samples_split=10,min_samples_leaf=5,max_features='sqrt',random_state=seed,n_jobs=4)
    if kind=='xgboost':
        try:from xgboost import XGBRegressor
        except ImportError as e:raise RuntimeError('pip install xgboost>=2') from e
        return XGBRegressor(n_estimators=200,max_depth=6,learning_rate=.1,
             subsample=.8,colsample_bytree=.8,reg_lambda=1.,tree_method='hist',n_jobs=4,
             objective='reg:squarederror',random_state=seed)
    raise ValueError(kind)


def run_classical(index,fit_ids,predict_ids,view,kind,features_dir,*,seed):
    mu,sd=normalizer_for(index,fit_ids,view,features_dir)
    models=[]
    for axis in range(2):
        xx=[];yy=[];weights=[]
        for sid in fit_ids:
            x,z=get_features(index,sid,view,features_dir)
            mask=z['mask'][:,axis]
            if not mask.any():continue
            xx.append(normalized(x[mask],mu,sd));yy.append(z['mean'][mask,axis]);
            weights.append(np.ones(mask.sum(),np.float32)/mask.sum())
        x=np.concatenate(xx); y=np.concatenate(yy);w=np.concatenate(weights)
        model=_estimator(kind,seed+axis)
        model.fit(x,y,sample_weight=w*len(fit_ids))
        models.append(model)
    pred={}
    for sid in predict_ids:
        x,_=get_features(index,sid,view,features_dir)
        nx=normalized(x,mu,sd)
        pred[sid]=np.clip(np.column_stack([m.predict(nx) for m in models]),-1,1).astype(np.float32)
    return pred,{'best_epoch':None},(mu,sd)


def run_member(index,fit_ids,stop_ids,predict_ids,view,kind,features_dir,seed,epochs,device,batch_size):
    if kind in ('xgboost','rf'):
        return run_classical(index,fit_ids,predict_ids,view,kind,features_dir,seed=seed)
    return run_neural(index,fit_ids,stop_ids,predict_ids,view,kind,features_dir,seed=seed,
                      epochs=epochs,device=device,batch_size=batch_size)


def stack(index,train_ids,val_ids,features_dir,seed,epochs,device,batch_size,folds=3):
    """12 methods across three feature families, 3-fold train-only OOF ridge.

    OOF fold's labels never used for that fold's base model fit/checkpoint selection.
    This version uses twelve *late* models (not all sixteen early+late hybrid
    models of the publication). See README; addition of four early-fusion
    models is handled below with the 'early' view.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    for view in STACK_FAMILIES:
        _=get_features(index,train_ids[0],view,features_dir)  # fail fast on missing PAA
    # Train-only OOF meta features; each song contributes the same weight.
    idx=np.arange(len(train_ids));rng=np.random.default_rng(seed);rng.shuffle(idx)
    kf=KFold(n_splits=folds,shuffle=True,random_state=seed)
    cols=[(view,kind) for view in STACK_FAMILIES for kind in STACK_MODELS]
    cols += [('early',kind) for kind in STACK_MODELS]
    # Build the paper's early fusion from concatenating three descriptors via
    # an in-memory view, materialized to independent aligned archives.
    early_path=Path(features_dir)/'early';early_path.mkdir(parents=True,exist_ok=True)
    all_ids=set(train_ids)|set(val_ids)
    for sid in all_ids:
        dst=early_path/f'{sid}.npz'
        if dst.exists():
            # Existing early features are validated later by get_features.
            continue
        blocks=[];t=None
        for view in STACK_FAMILIES:
            x,z=get_features(index,sid,view,features_dir)
            blocks.append(x);t=z['times']
        np.savez_compressed(dst,times=t,features=np.concatenate(blocks,axis=1))
    oof={sid:np.full((len(load_song(index,sid)['times']),len(cols)*2),np.nan,np.float32) for sid in train_ids}
    for j,(view,kind) in enumerate(cols):
        for fold,(tr,ho) in enumerate(kf.split(idx)):
            # Inner 10% of fold-training songs exclusively for checkpoint choice.
            available=[train_ids[idx[k]] for k in tr]
            nstop=max(1,int(len(available)*.1))
            subtrain=available[nstop:];stop=available[:nstop]
            hold=[train_ids[idx[k]] for k in ho]
            pp,_,_=run_member(index,subtrain,stop,hold,view,kind,features_dir,seed+fold*100+j,
                               epochs,device,batch_size)
            for sid in hold:oof[sid][:,2*j:2*j+2]=pp[sid]
    x_tr=[];ys=[[],[]];ws=[[],[]]
    for sid in train_ids:
        z=load_song(index,sid);x=oof[sid]
        if not np.isfinite(x).all():raise RuntimeError('Incomplete OOF predictions')
        for d in range(2):
            mask=z['mask'][:,d]
            if mask.any():
                ys[d].append(z['mean'][mask,d]);
                ws[d].append(np.full(mask.sum(),1/mask.sum(),np.float32))
        x_tr.append(x)
    # one feature matrix per dimension because masks can differ
    meta=[]
    for d in range(2):
        feats=[oof[sid][load_song(index,sid)['mask'][:,d]] for sid in train_ids if load_song(index,sid)['mask'][:,d].any()]
        x=np.concatenate(feats);y=np.concatenate(ys[d]);w=np.concatenate(ws[d])
        estimator=Ridge(alpha=1.,fit_intercept=False,positive=True)
        estimator.fit(x,y,sample_weight=w*len(train_ids))
        meta.append(estimator)
    # Refit all 16 paper-inspired base models on all training songs. Validation
    # used only for checkpoint selection, never for meta-learner fitting.
    val_pred={sid:np.zeros((len(load_song(index,sid)['times']),2*len(cols)),np.float32) for sid in val_ids}
    histories={}
    for j,(view,kind) in enumerate(cols):
        pp,history,_=run_member(index,train_ids,val_ids,val_ids,view,kind,features_dir,seed+1000+j,
                              epochs,device,batch_size)
        histories[f'{view}_{kind}']=history
        for sid in val_ids:val_pred[sid][:,2*j:2*j+2]=pp[sid]
    prediction={}
    for sid in val_ids:
        x=val_pred[sid]
        prediction[sid]=np.clip(np.column_stack([e.predict(x) for e in meta]),-1,1).astype(np.float32)
    return prediction,{'components':len(cols),'meta_coefficients':[e.coef_.tolist() for e in meta],
           'oof_folds':folds,'histories':histories},None


def adapted_stack(index,train_ids,val_ids,features_dir,seed,epochs,device,batch_size,folds=3):
    """Train-only song-disjoint OOF Ridge over four OpenSMILE/Librosa models.

    Validation labels are read only by the caller's final evaluation. Neural
    checkpoint selection uses an inner song-disjoint subset of each fit fold.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    train_ids=list(train_ids);val_ids=list(val_ids)
    if set(train_ids)&set(val_ids):raise AssertionError('Split leakage')
    rng=np.random.default_rng(seed)
    shuffled=np.asarray(train_ids)[rng.permutation(len(train_ids))]
    splits=list(KFold(n_splits=folds,shuffle=True,random_state=seed).split(shuffled))
    width=2*len(ADAPTED_STACK_MEMBERS)
    oof={sid:np.full((len(load_song(index,sid)['times']),width),np.nan,np.float32)
         for sid in train_ids}
    fold_audit=[]
    for fold,(fit_idx,hold_idx) in enumerate(splits):
        available=shuffled[fit_idx].tolist();hold=shuffled[hold_idx].tolist()
        inner_rng=np.random.default_rng(seed+10000+fold)
        inner_rng.shuffle(available)
        nstop=max(1,int(round(.1*len(available))))
        stop=available[:nstop];fit=available[nstop:]
        if (set(fit)&set(stop)) or (set(fit)&set(hold)) or (set(stop)&set(hold)):
            raise AssertionError('OOF song leakage')
        fold_audit.append({'fold':fold,'fit_songs':len(fit),'checkpoint_songs':len(stop),
                           'holdout_songs':len(hold),'holdout_ids':hold})
        for j,(view,kind) in enumerate(ADAPTED_STACK_MEMBERS):
            print(f'adapted stack seed={seed} OOF fold={fold+1}/{folds} member={view}_{kind}',flush=True)
            pp,_,_=run_member(index,fit,stop,hold,view,kind,features_dir,
                               seed+fold*100+j,epochs,device,batch_size)
            for sid in hold:oof[sid][:,2*j:2*j+2]=pp[sid]
    if any(not np.isfinite(x).all() for x in oof.values()):
        raise RuntimeError('Incomplete OOF predictions')
    meta=[]
    for axis in range(2):
        xx=[];yy=[];ww=[]
        for sid in train_ids:
            z=load_song(index,sid);mask=z['mask'][:,axis]
            if not mask.any():continue
            xx.append(oof[sid][mask]);yy.append(z['mean'][mask,axis])
            ww.append(np.full(mask.sum(),1/mask.sum(),np.float32))
        x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww)
        model=Ridge(alpha=1.,fit_intercept=False,positive=True)
        model.fit(x,y,sample_weight=w*len(train_ids))
        meta.append(model)
    # A training-only checkpoint subset also selects the final neural models.
    full_ids=train_ids.copy();np.random.default_rng(seed+20000).shuffle(full_ids)
    nstop=max(1,int(round(.1*len(full_ids))))
    stop=full_ids[:nstop];fit=full_ids[nstop:]
    val_matrix={sid:np.empty((len(load_song(index,sid)['times']),width),np.float32)
                for sid in val_ids}
    histories={}
    for j,(view,kind) in enumerate(ADAPTED_STACK_MEMBERS):
        print(f'adapted stack seed={seed} final member={view}_{kind}',flush=True)
        pp,history,_=run_member(index,fit,stop,val_ids,view,kind,features_dir,
                                seed+1000+j,epochs,device,batch_size)
        histories[f'{view}_{kind}']=history
        for sid in val_ids:val_matrix[sid][:,2*j:2*j+2]=pp[sid]
    prediction={sid:np.clip(np.column_stack([m.predict(x) for m in meta]),-1,1).astype(np.float32)
                for sid,x in val_matrix.items()}
    detail={'label':'OpenSMILE–Librosa Ridge Stacking (adapted)',
            'members':[f'{view}_{kind}' for view,kind in ADAPTED_STACK_MEMBERS],
            'oof_folds':folds,'fold_audit':fold_audit,
            'ridge_alpha':1.,'ridge_positive':True,'ridge_fit_intercept':False,
            'ridge_training_songs':len(train_ids),'validation_labels_used_for_fit':False,
            'final_base_fit_songs':len(fit),'final_checkpoint_songs':len(stop),
            'meta_coefficients':[m.coef_.tolist() for m in meta],
            'histories':histories}
    return prediction,detail,oof


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True)
    p.add_argument('--features-dir',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028,2029,2030])
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--stack-folds',type=int,default=3)
    args=p.parse_args(argv)
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.methods))!=len(args.methods):raise ValueError('Duplicate seeds/methods')
    if args.epochs<1 or args.stack_folds<2:raise ValueError('Bad epochs/folds')
    index,splits=load_index(args.manifest,args.reference_run)
    train_ids=list(map(str,splits['train']));val_ids=list(map(str,splits['val']))
    if set(train_ids)&set(val_ids):raise AssertionError('Split leakage')
    if len(train_ids)!=902 or len(val_ids)!=270:
        raise ValueError('B2 comparison requires exactly 902 training and 270 validation songs')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in args.methods:
        view,kind=METHODS[name];all_predictions=[]
        for seed in args.seeds:
            dest=result_root(root/name/f'seed_{seed}')
            if kind=='stack':
                pp,detail,_=stack(index,train_ids,val_ids,args.features_dir,seed,args.epochs,args.device,
                                   args.batch_size,args.stack_folds)
            elif kind=='adapted_stack':
                pp,detail,oof=adapted_stack(index,train_ids,val_ids,args.features_dir,seed,args.epochs,
                                             args.device,args.batch_size,args.stack_folds)
                oof_dir=dest/'training_oof';oof_dir.mkdir()
                for sid,x in oof.items():
                    np.savez_compressed(oof_dir/f'{sid}.npz',times=load_song(index,sid)['times'],pred=x)
            else:
                pp,detail,_=run_member(index,train_ids,val_ids,val_ids,view,kind,args.features_dir,
                                       seed,args.epochs,args.device,args.batch_size)
            comparison=('adapted two-family OpenSMILE–Librosa Ridge stacking, development only'
                        if kind=='adapted_stack' else 'paper-inspired reproduction, development only')
            m=save_report(dest,pp,index,val_ids,{'method':name,'seed':seed,'detail':detail,
                     'comparison_type':comparison})
            rows.append({'method':name,'seed':seed,'evaluation':'individual',**m})
            all_predictions.append(pp)
            print(name,seed,m,flush=True)
        dest=result_root(root/name/'ensemble')
        ens={sid:np.stack([p[sid] for p in all_predictions]).mean(0) for sid in val_ids}
        m=save_report(dest,ens,index,val_ids,{'method':name,'seeds':args.seeds,'evaluation':'ensemble',
                     'note':'Repeated development-val metric, not confirmatory result'})
        rows.append({'method':name,'seed':'ensemble','evaluation':'ensemble',**m})
    pd.DataFrame(rows).to_csv(root/'all_baselines_development.csv',index=False)
    print('Saved:',root/'all_baselines_development.csv')

if __name__=='__main__':main()
