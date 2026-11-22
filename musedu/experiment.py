"""Reproducible song-split training, independent scale fitting, calibration, evaluation.

All checkpoints and fit statistics come only from training/validation/scale/calibration
as explicitly specified; tests never contribute to selection or conformal quantiles.
"""
from __future__ import annotations
import json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from .model import MUSEDU,ErrorScale,supervised_loss,laplace_scale_loss
from .metrics import song_metrics,aggregate
from .calibration import song_score,conformal_quantile,evaluate_intervals

SPLITS={'train':.50,'val':.15,'scale':.10,'cal':.10,'test':.15}


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_manifest(path):
    df=pd.read_csv(path,dtype={'song_id':str})
    required={'dataset','song_id','file'}
    if not required.issubset(df):raise ValueError(f'Manifest missing {required-set(df)}')
    if df.duplicated(['dataset','song_id']).any():raise ValueError('Duplicate song IDs in manifest')
    df['file']=df.file.map(lambda p:str(Path(p).resolve()))
    for f in df.file:
        if not Path(f).is_file():raise FileNotFoundError(f)
    return df


def split_songs(df,seed=2026,ratios=SPLITS):
    if set(ratios)!=set(SPLITS) or not np.isclose(sum(ratios.values()),1):raise ValueError('Invalid split fractions')
    songs=df.song_id.astype(str).to_numpy()
    if len(songs)<len(ratios)*2:raise ValueError('Need at least 10 songs to create disjoint splits')
    rng=np.random.default_rng(seed);order=songs[rng.permutation(len(songs))]
    sizes={k:int(np.floor(len(order)*v)) for k,v in ratios.items()}
    # assign rounding leftovers to training split
    sizes['train']+=len(order)-sum(sizes.values())
    out={};pos=0
    for key in SPLITS:
        out[key]=order[pos:pos+sizes[key]].tolist();pos+=sizes[key]
        if not out[key]:raise ValueError(f'Empty {key} split')
    assert set.union(*(set(s) for s in out.values()))==set(songs)
    assert sum(map(len,out.values()))==len(set(songs))
    return out


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    for key in ('acoustic','embedding','mean','std','mask','std_mask','times'):
        if key not in d:raise ValueError(f'{path}: missing {key}')
    if d['mean'].ndim!=2 or d['mean'].shape[1]!=2:raise ValueError('Targets must have two axes')
    if len(d['times'])!=len(d['mean']):raise ValueError('Timestamp/label length mismatch')
    if 'step_seconds' in d and (np.asarray(d['step_seconds']).ndim or
                                 not np.isfinite(float(d['step_seconds'])) or float(d['step_seconds'])<=0):
        raise ValueError(f'{path}: invalid step_seconds metadata')
    return d


def nominal_step(z):
    if 'step_seconds' in z:return float(z['step_seconds'])
    # Legacy cached features can still be read, but rerun `prepare` to create
    # an unambiguous step and audio fingerprint for rigorous external validation.
    deltas=np.diff(z['times'])
    positive=deltas[np.isfinite(deltas)&(deltas>0)]
    return float(np.min(positive)) if len(positive) else 0.


def provenance(df):
    records=[]
    for row in df.itertuples(index=False):
        z=read_npz(row.file)
        digest=str(z['audio_sha256'].item()) if 'audio_sha256' in z else None
        records.append({'dataset':str(row.dataset),'song_id':str(row.song_id),
                        'audio_sha256':digest})
    return records


def stats_from_train(df,ids):
    pieces=[]
    lookup=dict(zip(df.song_id,df.file))
    for sid in ids:
        z=read_npz(lookup[sid]);pieces.append(np.concatenate([z['acoustic'],z['embedding']],axis=1))
    x=np.concatenate(pieces,axis=0)
    mean=x.mean(axis=0);std=x.std(axis=0);std=np.maximum(std,1e-4)
    return mean.astype('float32'),std.astype('float32')


class SequenceWindows(Dataset):
    def __init__(self,df,ids,normalizer,win=120):
        self.rows=[]; lookup=dict(zip(df.song_id,df.file));mu,sd=normalizer
        for sid in ids:
            z=read_npz(lookup[sid]);feat=(np.concatenate([z['acoustic'],z['embedding']],axis=1)-mu)/sd
            for start in range(0,len(feat),win):
                stop=min(start+win,len(feat))
                if not z['mask'][start:stop].any():continue
                self.rows.append({k:z[k][start:stop]
                                  for k in ('times','mean','std','mask','std_mask')}
                                 |{'x':feat[start:stop], 'step_seconds':nominal_step(z)})
        if not self.rows:raise ValueError('No labeled windows')

    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]


def pack(rows):
    maxlen=max(len(x['times']) for x in rows); b=len(rows)
    feat=rows[0]['x'].shape[-1]
    out={'step':np.zeros(b,'float32'),
         'x':np.zeros((b,maxlen,feat),'float32'),
         'times':np.zeros((b,maxlen),'float32'),
         'mean':np.zeros((b,maxlen,2),'float32'),
         'std':np.zeros((b,maxlen,2),'float32'),
         'mask':np.zeros((b,maxlen,2),'bool'),
         'std_mask':np.zeros((b,maxlen,2),'bool'),
         'valid':np.zeros((b,maxlen),'bool')}
    for k,row in enumerate(rows):
        n=len(row['times'])
        for name in ('x','times','mean','std','mask','std_mask'):
            out[name][k,:n]=np.nan_to_num(row[name],nan=0)
        out['valid'][k,:n]=True
        inferred=np.diff(row['times'])
        inferred=inferred[np.isfinite(inferred)&(inferred>0)]
        out['step'][k]=float(row.get('step_seconds',np.min(inferred) if len(inferred) else 0.))
        if out['step'][k]<0 or not np.isfinite(out['step'][k]):
            raise ValueError('Invalid temporal sampling interval')
    out={k:torch.from_numpy(v) for k,v in out.items()}
    return out


def on_device(data,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in data.items()}


def train_one(df,split,normalizer,out_dir,seed=2026,epochs=25,batch_size=8,lr=3e-4,window=120,
              hidden=96,device='cpu'):
    seed_everything(seed)
    train=SequenceWindows(df,split['train'],normalizer,window)
    valid=SequenceWindows(df,split['val'],normalizer,window)
    tr=DataLoader(train,batch_size=batch_size,shuffle=True,collate_fn=pack)
    va=DataLoader(valid,batch_size=batch_size,shuffle=False,collate_fn=pack)
    f=len(normalizer[0]);model=MUSEDU(f,hidden=hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    best=float('inf');patience=7;bad=0
    for epoch in range(epochs):
        model.train()
        for data in tr:
            batch=on_device(data,device)
            opt.zero_grad(set_to_none=True)
            out=model(batch['x'],batch['times'],batch['valid'])
            loss,_=supervised_loss(out,batch)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval();losses=[]
        with torch.inference_mode():
            for data in va:
                batch=on_device(data,device);out=model(batch['x'],batch['times'],batch['valid'])
                loss,_=supervised_loss(out,batch);losses.append(float(loss))
        score=float(np.mean(losses))
        if score<best-1e-7:
            best=score;bad=0
            torch.save({'state_dict':model.state_dict(),'in_features':f,'hidden':hidden},out_dir/f'model_{seed}.pt')
        else:
            bad+=1
            if bad>=patience:break
    return best


def load_models(directory,device='cpu'):
    paths=sorted(Path(directory).glob('model_*.pt'))
    if not paths:raise FileNotFoundError('No ensemble model_*.pt checkpoints')
    models=[]
    for p in paths:
        state=torch.load(p,map_location=device,weights_only=True)
        m=MUSEDU(state['in_features'],hidden=state['hidden']).to(device)
        m.load_state_dict(state['state_dict']);m.eval();models.append(m)
    return models


def predict_one(models,z,mu,sd,device='cpu',window=120):
    feat=np.concatenate([z['acoustic'],z['embedding']],axis=1)
    feat=(feat-mu)/sd
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(feat),window):
            stop=min(len(feat),start+window)
            x=torch.from_numpy(feat[start:stop][None].astype('float32')).to(device)
            times=torch.from_numpy(z['times'][start:stop][None].astype('float32')).to(device)
            valid=torch.ones((1,len(x[0])),device=device,dtype=torch.bool)
            preds=[m(x,times,valid) for m in models]
            mean=np.stack([p['mean'][0].cpu().numpy() for p in preds])
            tau=np.stack([p['tau'][0].cpu().numpy() for p in preds])
            h=np.stack([p['hidden'][0].cpu().numpy() for p in preds])
            chunks.append({'pred':mean.mean(0),'tau':tau.mean(0),'hidden':h.mean(0),
                           'std_model':np.sqrt(np.maximum(mean.var(axis=0,ddof=1),0))
                           if len(models)>1 else np.zeros_like(mean[0])})
    return {k:np.concatenate([x[k] for x in chunks],axis=0).astype('float32') for k in chunks[0]}


def fit_scale(df,ids,models,mu,sd,folder,device='cpu',window=120,epochs=50):
    lookup=dict(zip(df.song_id,df.file));seq=[]
    for sid in ids:
        z=read_npz(lookup[sid]);p=predict_one(models,z,mu,sd,device,window)
        windows=[]
        for start in range(0,len(p['pred']),window):
            stop=min(start+window,len(p['pred']))
            windows.append(({k:v[start:stop] for k,v in p.items()},
                            {k:z[k][start:stop] for k in ('times','mean','mask')},nominal_step(z)))
        seq.append(windows)
    net=ErrorScale(models[0].hidden_dim).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    for epoch in range(epochs):
        net.train();random.shuffle(seq)
        for windows in seq:
            # Exactly one optimizer step per song, independent of its duration.
            # Chunk losses are combined according to their number of valid bins.
            weights=[int(z['mask'].sum()) for _,z,_ in windows]
            total=sum(weights)
            if total==0:continue
            opt.zero_grad(set_to_none=True)
            for (p,z,step),weight in zip(windows,weights):
                if weight==0:continue
                def ten(x):return torch.as_tensor(x,device=device).unsqueeze(0)
                hidden=ten(p['hidden']);tau=ten(p['tau']);s=ten(p['std_model']);mean=ten(p['pred'])
                time=ten(z['times']);truth=ten(np.nan_to_num(z['mean'],nan=0));mask=ten(z['mask']).bool()
                scale=net(hidden,tau,s,mean,time,step_seconds=step)
                loss=laplace_scale_loss(scale,mean,truth,mask)
                (loss*(weight/total)).backward()
            opt.step()
    torch.save({'state_dict':net.state_dict(),'hidden':models[0].hidden_dim},folder/'error_scale.pt')
    return net.eval()


def score_set(df,ids,models,net,mu,sd,device='cpu',window=120):
    lookup=dict(zip(df.song_id,df.file));scores=[];metrics=[];items=[]
    with torch.inference_mode():
        for sid in ids:
            z=read_npz(lookup[sid]);p=predict_one(models,z,mu,sd,device,window)
            scales=[]
            for start in range(0,len(p['pred']),window):
                stop=min(start+window,len(p['pred']))
                def t(k):return torch.from_numpy(p[k][start:stop][None]).to(device)
                tt=torch.from_numpy(z['times'][start:stop][None]).to(device)
                scales.append(net(t('hidden'),t('tau'),t('std_model'),t('pred'),tt,
                                  step_seconds=nominal_step(z))[0].cpu().numpy())
            scale=np.concatenate(scales)
            scores.append(song_score(p['pred'],z['mean'],scale,z['mask']))
            metrics.append(song_metrics(p['pred'],z['mean'],z['mask']))
            items.append({'song_id':sid,'pred':p['pred'],'ref':z['mean'],'scale':scale,'mask':z['mask']})
    return scores,metrics,items


def run_training(manifest,output,*,device='cpu',ensemble=5,epochs=25,scale_epochs=50,window=120,
                 hidden=96,seed=2026,split_override=None):
    df=load_manifest(manifest)
    if df.dataset.nunique()!=1:raise ValueError('Training must use one dataset; external testing is separate')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('model_*.pt')):
        raise FileExistsError('Existing model checkpoints in output directory: choose a clean --out to avoid mixing ensembles')
    # Persist all source songs, not just training split, since all participate
    # in training/selection/calibration/reporting and must remain independent
    # from zero-shot external tests.
    source_records=provenance(df)
    if any(r['audio_sha256'] is None for r in source_records):
        raise ValueError('Missing decoded-audio SHA256 in training features; rerun `prepare`')
    all_hashes=[r['audio_sha256'] for r in source_records]
    if len(all_hashes)!=len(set(all_hashes)):
        raise ValueError('Duplicate decoded audio in development manifest; resolve before song-disjoint splitting')
    (out/'source_manifest.json').write_text(json.dumps(source_records,indent=2))
    split=split_songs(df,seed=seed) if split_override is None else split_override
    if set(split)!=set(SPLITS) or any(not split[k] for k in SPLITS):
        raise ValueError('Split override requires five nonempty named splits')
    ids=[str(sid) for key in SPLITS for sid in split[key]]
    if len(ids)!=len(set(ids)) or set(ids)!=set(df.song_id.astype(str)):
        raise ValueError('Split override must partition every song exactly once')
    (out/'splits.json').write_text(json.dumps(split,indent=2))
    mu,sd=stats_from_train(df,split['train'])
    np.savez(out/'normalizer.npz',mean=mu,std=sd)
    for k in range(ensemble):
        train_one(df,split,(mu,sd),out,seed=seed+k,epochs=epochs,window=window,
                  hidden=hidden,device=device)
    models=load_models(out,device)
    net=fit_scale(df,split['scale'],models,mu,sd,out,device,window,scale_epochs)
    cal_scores,_,_=score_set(df,split['cal'],models,net,mu,sd,device,window)
    q=conformal_quantile(cal_scores,alpha=.1)
    (out/'calibration.json').write_text(json.dumps({'alpha':.1,'q':q,'n_cal':len(cal_scores)},indent=2))
    _,rep,items=score_set(df,split['test'],models,net,mu,sd,device,window)
    result={'split_sizes':{k:len(v) for k,v in split.items()},'point_metrics':aggregate(rep),
            'interval_metrics':evaluate_intervals(items,q),'experimental':True}
    (out/'test_metrics.json').write_text(json.dumps(result,indent=2))
    return result


def run_external(manifest,trained,device='cpu',window=120):
    df=load_manifest(manifest)
    folder=Path(trained)
    source_file=folder/'source_manifest.json'
    if not source_file.exists():
        raise FileNotFoundError('Missing source_manifest.json; retrain with provenance tracking before external evaluation')
    source=json.loads(source_file.read_text())
    if df.dataset.nunique()!=1:raise ValueError('External evaluation must use one dataset')
    if df.dataset.iloc[0] in {r['dataset'] for r in source}:
        raise ValueError('External evaluation dataset matches the development corpus; use the held-out test split instead')
    # MERP contains a small set of DEAM anchor recordings, known a priori.
    excluded=[]
    if set(df.dataset)=={'merp'}:
        excluded=df.loc[df.song_id.str.startswith('deam_'),'song_id'].tolist()
        df=df[~df.song_id.str.startswith('deam_')].copy()
        if df.empty:raise ValueError('Only overlapping MERP anchor songs remain')
    source_hashes={r.get('audio_sha256') for r in source if r.get('audio_sha256')}
    ext=provenance(df)
    if any(not r.get('audio_sha256') for r in source+ext):
        raise ValueError('Missing decoded-audio SHA256: rerun `prepare` on BOTH datasets for leakage-safe evaluation')
    duplicate_hashes=source_hashes&{r['audio_sha256'] for r in ext}
    if duplicate_hashes:
        raise ValueError(f'Audio overlap between development and external sets: {len(duplicate_hashes)} matching SHA256 hashes')
    # Hash equality does not rule out remastered, cropped or otherwise modified
    # copies of the same song; an additional perceptual fingerprint audit is needed.
    models=load_models(folder,device)
    normalizer=np.load(folder/'normalizer.npz');mu=normalizer['mean'];sd=normalizer['std']
    state=torch.load(folder/'error_scale.pt',weights_only=True,map_location=device)
    net=ErrorScale(state['hidden']).to(device);net.load_state_dict(state['state_dict']);net.eval()
    q=json.loads((folder/'calibration.json').read_text())['q']
    _,reps,items=score_set(df,df.song_id.tolist(),models,net,mu,sd,device,window)
    return {'note':'External-domain coverage is empirical, not guaranteed; identical decoded-audio hashes checked, perceptual overlap not excluded',
            'excluded_known_anchor_ids':excluded, 'audio_hash_overlap':0,
            'point_metrics':aggregate(reps),'interval_metrics':evaluate_intervals(items,q)}
