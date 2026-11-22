"""Exploratory v2 development restricted to saved training/validation songs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
from .experiment import (load_manifest,read_npz,pack,on_device,seed_everything,
                         stats_from_train,nominal_step)
from .model import MUSEDUV2,v2_supervised_loss,V2_LOSS_WEIGHTS,_equal_song_huber
from .metrics import song_metrics,aggregate,within_song_pearson
from .baselines import BaselineNetwork


class FullSongs(Dataset):
    def __init__(self,df,ids,norm):
        lookup=dict(zip(df.song_id,df.file));mu,sd=norm
        self.rows=[];self.lengths=[]
        for sid in ids:
            z=read_npz(lookup[sid])
            x=(np.concatenate([z['acoustic'],z['embedding']],axis=1)-mu)/sd
            self.rows.append({'x':x,'times':z['times'],'mean':z['mean'],
                              'std':z['std'],'mask':z['mask'],'std_mask':z['std_mask'],
                              'step_seconds':nominal_step(z)})
            self.lengths.append(len(x))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]


class LengthBuckets(Sampler):
    """Batch short songs together; keep long full recordings individually."""
    def __init__(self,lengths,batch_size=8,shuffle=True,seed=2026):
        self.short=[i for i,n in enumerate(lengths) if n<=120]
        self.long=[i for i,n in enumerate(lengths) if n>120]
        self.batch_size=batch_size;self.shuffle=shuffle;self.seed=seed;self.epoch=0
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);self.epoch+=1
        short=np.array(self.short,dtype=int)
        if self.shuffle:rng.shuffle(short)
        batches=[short[i:i+self.batch_size].tolist() for i in range(0,len(short),self.batch_size)]
        batches.extend([[i] for i in self.long])
        if self.shuffle:rng.shuffle(batches)
        yield from batches
    def __len__(self):return (len(self.short)+self.batch_size-1)//self.batch_size+len(self.long)


def _reference_split(df,reference_run):
    root=Path(reference_run)
    split=json.loads((root/'splits.json').read_text())
    if set(split)!={'train','val','scale','cal','test'}:raise ValueError('Invalid reference split')
    ids=[str(sid) for values in split.values() for sid in values]
    if len(ids)!=len(set(ids)) or set(ids)!=set(df.song_id):
        raise ValueError('Reference split overlaps or differs from manifest')
    source_file=root/'source_manifest.json'
    if not source_file.is_file():raise FileNotFoundError('Reference source_manifest.json required')
    saved={(str(x['dataset']),str(x['song_id'])):x['audio_sha256']
           for x in json.loads(source_file.read_text())}
    current={}
    for row in df.itertuples():
        # Read only identity metadata. Old test features/labels stay unopened.
        with np.load(row.file,allow_pickle=False) as z:
            digest=str(z['audio_sha256'].item()) if 'audio_sha256' in z else None
        current[(str(row.dataset),str(row.song_id))]=digest
    if saved!=current:raise ValueError('Reference audio source identities differ')
    return split


def load_v2_checkpoint(path,device='cpu'):
    state=torch.load(path,map_location=device,weights_only=True)
    if state.get('model_version')!='v2':raise ValueError('Expected v2 checkpoint')
    model=MUSEDUV2(state['in_features'],hidden=state['hidden']).to(device)
    model.load_state_dict(state['state_dict']);return model.eval()


def predict_v2(model,z,norm,device='cpu'):
    mu,sd=norm
    x=(np.concatenate([z['acoustic'],z['embedding']],axis=1)-mu)/sd
    with torch.inference_mode():
        xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
        tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device)
        valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
        out=model(xx,tt,valid)
    return {key:val[0].cpu().numpy() for key,val in out.items()}


def predict_v2_ensemble(models,z,norm,device='cpu'):
    outputs=[predict_v2(model,z,norm,device) for model in models]
    result={key:np.mean([out[key] for out in outputs],axis=0)
            for key in ('mean','tau','hidden','song_mean','song_tau')}
    members=np.stack([out['mean'] for out in outputs])
    result['std_model']=members.std(0,ddof=1) if len(models)>1 else np.zeros_like(members[0])
    return result


def _within_song_corr(pred,target,mask):
    result=within_song_pearson(pred,target,mask)
    return [result[f'{name}_within_pearson'] for name in ('valence','arousal')]


def evaluate_v2(model,df,ids,norm,device='cpu'):
    lookup=dict(zip(df.song_id,df.file));rows=[]
    for sid in ids:
        z=read_npz(lookup[sid]);p=predict_v2(model,z,norm,device)
        metric=song_metrics(p['mean'],z['mean'],z['mask'])
        corr=_within_song_corr(p['mean'],z['mean'],z['mask'])
        rows.append({'song_id':sid,**metric,'valence_within_pearson':corr[0],
                     'arousal_within_pearson':corr[1]})
    table=pd.DataFrame(rows)
    summary={k:float(table[k].mean()) for k in table if k!='song_id'}
    return table,summary


def evaluate_v2_ensemble(models,df,ids,norm,device='cpu'):
    lookup=dict(zip(df.song_id,df.file));rows=[]
    for sid in ids:
        z=read_npz(lookup[sid]);p=predict_v2_ensemble(models,z,norm,device)
        metric=song_metrics(p['mean'],z['mean'],z['mask'])
        corr=_within_song_corr(p['mean'],z['mean'],z['mask'])
        rows.append({'song_id':sid,**metric,'valence_within_pearson':corr[0],
                     'arousal_within_pearson':corr[1]})
    table=pd.DataFrame(rows)
    return table,{k:float(table[k].mean()) for k in table if k!='song_id'}


def run_v2_development(manifest,reference_run,out,*,weights=None,seed=2026,
                       epochs=15,hidden=96,device='cpu',batch_size=8):
    """Never reads held-out test features or labels to fit/select/report v2."""
    weights=dict(V2_LOSS_WEIGHTS if weights is None else weights)
    if set(weights)!=set(V2_LOSS_WEIGHTS) or any(v<0 for v in weights.values()):
        raise ValueError('Provide five nonnegative v2 loss weights')
    df=load_manifest(manifest)
    split=_reference_split(df,reference_run)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Choose an empty v2 development output directory')
    seed_everything(seed)
    norm=stats_from_train(df,split['train'])
    np.savez(out/'normalizer.npz',mean=norm[0],std=norm[1])
    (out/'development_protocol.json').write_text(json.dumps({
        'model_version':'v2','seed':seed,'weights':weights,
        'train_song_ids':split['train'],'validation_song_ids':split['val'],
        'excluded_from_all_v2_fitting_selection_and_reporting':{
            'scale':len(split['scale']),'cal':len(split['cal']),'test':len(split['test'])},
        'note':'Exploratory development on the original DEAM training/validation split; the old test set informed the redesign and is not confirmatory.'},indent=2))
    train=FullSongs(df,split['train'],norm);valid=FullSongs(df,split['val'],norm)
    tr=DataLoader(train,batch_sampler=LengthBuckets(train.lengths,batch_size,True,seed),collate_fn=pack)
    va=DataLoader(valid,batch_sampler=LengthBuckets(valid.lengths,batch_size,False,seed),collate_fn=pack)
    model=MUSEDUV2(len(norm[0]),hidden=hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=float('inf');bad=0;history=[]
    for epoch in range(epochs):
        model.train();training=[]
        for data in tr:
            batch=on_device(data,device)
            opt.zero_grad(set_to_none=True)
            pred=model(batch['x'],batch['times'],batch['valid'])
            loss,_=v2_supervised_loss(pred,batch,weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite v2 training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();training.append(float(loss.detach()))
        model.eval();validation=[]
        with torch.inference_mode():
            for data in va:
                batch=on_device(data,device);pred=model(batch['x'],batch['times'],batch['valid'])
                loss,_=v2_supervised_loss(pred,batch,weights)
                validation.append(float(loss))
        score=float(np.mean(validation))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(training)),'val_loss':score})
        print(f'v2 epoch {epoch+1}: train={history[-1]["train_loss"]:.5f} val={score:.5f}',flush=True)
        if score<best-1e-7:
            best=score;bad=0
            torch.save({'model_version':'v2','state_dict':model.state_dict(),
                        'in_features':len(norm[0]),'hidden':hidden,'weights':weights},out/'model.pt')
        else:
            bad+=1
            if bad>=5:break
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    chosen=load_v2_checkpoint(out/'model.pt',device)
    table,summary=evaluate_v2(chosen,df,split['val'],norm,device)
    table.to_csv(out/'validation_by_song.csv',index=False)
    result={'best_validation_loss':best,'epochs_ran':len(history),
            'validation_metrics':summary,'exploratory':True,
            'test_set_used':False}
    (out/'validation_summary.json').write_text(json.dumps(result,indent=2))
    return result


def run_hierarchical_control_development(manifest,reference_run,out,*,seed=2026,
                                         epochs=15,hidden=96,device='cpu',batch_size=8):
    """CNN–BiGRU with the same global/local output idea and point loss only."""
    df=load_manifest(manifest)
    split=_reference_split(df,reference_run)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Choose an empty control output directory')
    seed_everything(seed)
    norm=stats_from_train(df,split['train'])
    np.savez(out/'normalizer.npz',mean=norm[0],std=norm[1])
    train=FullSongs(df,split['train'],norm);valid=FullSongs(df,split['val'],norm)
    tr=DataLoader(train,batch_sampler=LengthBuckets(train.lengths,batch_size,True,seed),collate_fn=pack)
    va=DataLoader(valid,batch_sampler=LengthBuckets(valid.lengths,batch_size,False,seed),collate_fn=pack)
    model=BaselineNetwork('cnn_bigru_hierarchical',len(norm[0]),hidden=hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=float('inf');bad=0;history=[]
    for epoch in range(epochs):
        model.train();training=[]
        for data in tr:
            batch=on_device(data,device);opt.zero_grad(set_to_none=True)
            pred=model(batch['x'],batch['times'],batch['valid'])
            loss=_equal_song_huber(pred,batch['mean'],batch['mask'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();training.append(float(loss.detach()))
        model.eval();validation=[]
        with torch.inference_mode():
            for data in va:
                batch=on_device(data,device)
                pred=model(batch['x'],batch['times'],batch['valid'])
                validation.append(float(_equal_song_huber(pred,batch['mean'],batch['mask'])))
        score=float(np.mean(validation));history.append({'epoch':epoch+1,'train_loss':float(np.mean(training)),'val_loss':score})
        print(f'hierarchical control epoch {epoch+1}: val={score:.5f}',flush=True)
        if score<best-1e-7:
            best=score;bad=0
            torch.save({'state_dict':model.state_dict(),'method':'cnn_bigru_hierarchical',
                        'in_features':len(norm[0]),'hidden':hidden},out/'model.pt')
        else:
            bad+=1
            if bad>=5:break
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    state=torch.load(out/'model.pt',map_location=device,weights_only=True)
    model.load_state_dict(state['state_dict']);model.eval()
    lookup=dict(zip(df.song_id,df.file));rows=[]
    for sid in split['val']:
        z=read_npz(lookup[sid]);x=(np.concatenate([z['acoustic'],z['embedding']],axis=1)-norm[0])/norm[1]
        with torch.inference_mode():
            xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
            tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device)
            pred=model(xx,tt,torch.ones((1,len(x)),dtype=torch.bool,device=device))[0].cpu().numpy()
        corr=_within_song_corr(pred,z['mean'],z['mask'])
        rows.append({'song_id':sid,**song_metrics(pred,z['mean'],z['mask']),
                     'valence_within_pearson':corr[0],'arousal_within_pearson':corr[1]})
    table=pd.DataFrame(rows);table.to_csv(out/'validation_by_song.csv',index=False)
    result={'validation_metrics':{k:float(table[k].mean()) for k in table if k!='song_id'},
            'best_validation_loss':best,'exploratory':True,'test_set_used':False}
    (out/'validation_summary.json').write_text(json.dumps(result,indent=2))
    (out/'development_protocol.json').write_text(json.dumps({
        'train_song_ids':split['train'],'validation_song_ids':split['val'],
        'test_set_used':False,'architecture':'CNN–BiGRU hierarchical global/local control',
        'loss':'point Huber only, equal song weighting'},indent=2))
    return result
