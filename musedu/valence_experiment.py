"""Controlled development experiments on versioned MERT features."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .experiment import load_manifest,read_npz,pack,on_device,seed_everything,nominal_step
from .matched_development import make_model,output_mean,loss_for,selector
from .metrics import song_metrics,within_song_pearson,transition_metrics,event_f1_from_counts
from .v2_experiment import LengthBuckets,_reference_split


def input_features(z,representation='last',delay=0.,harmonic=False):
    if 'context_times' in z:
        requested=z['times']-delay;ix=np.searchsorted(z['context_times'],requested)
        if np.any(ix>=len(z['context_times'])) or not np.allclose(z['context_times'][ix],requested,atol=1e-4):
            raise ValueError('Delay not supported by cached audio context')
        a=z['context_acoustic'][ix];layers=z['context_layers'][ix]
        harmonic_data=z['context_harmonic'][ix]
    else:
        if delay or harmonic or representation=='mix':raise ValueError('Enhanced cache required')
        a=z['acoustic'];layers=z['embedding'][:,None,:];harmonic_data=None
    if harmonic:a=np.concatenate((a,harmonic_data),axis=1)
    e=layers[:,-1] if representation=='last' else layers.reshape(len(a),-1)
    return np.concatenate((a,e),axis=1).astype('float32')


class LayerRegressor(nn.Module):
    def __init__(self,architecture,acoustic_dim,layer_count,embedding_dim=768,hidden=96,long_valence=False):
        super().__init__();self.acoustic_dim=acoustic_dim;self.layer_count=layer_count
        self.embedding_dim=embedding_dim
        if layer_count>1:self.layer_logits=nn.Parameter(torch.zeros(layer_count))
        if architecture in ('trace','trace_no_cue_loss','trace_no_cues'):
            from .trace_mer import TRACEMER
            self.model=TRACEMER(embedding_dim=embedding_dim,acoustic_dim=acoustic_dim,hidden=hidden,
                               cue_decomposition=architecture!='trace_no_cues')
        elif long_valence:
            from .echo_mer import ECHOMER
            self.model=ECHOMER(acoustic_dim+embedding_dim,hidden=hidden,
                               event_conditioned=True,long_valence=True)
        else:self.model=make_model(architecture,acoustic_dim+embedding_dim,hidden)
    def forward(self,x,times,valid):
        a=x[:,:,:self.acoustic_dim]
        layers=x[:,:,self.acoustic_dim:].reshape(*x.shape[:2],self.layer_count,self.embedding_dim)
        if self.layer_count>1:e=(layers*self.layer_logits.softmax(0)[None,None,:,None]).sum(2)
        else:e=layers[:,:,0]
        return self.model(torch.cat((a,e),dim=-1),times,valid)


class FeatureSongs:
    def __init__(self,df,ids,config,norm):
        lookup=dict(zip(df.song_id,df.file));self.rows=[];self.lengths=[]
        for sid in ids:
            z=read_npz(lookup[sid]);x=input_features(z,config['representation'],config['delay'],config['harmonic'])
            self.rows.append({'x':(x-norm[0])/norm[1],**{k:z[k] for k in ('times','mean','std','mask','std_mask')},
                              'step_seconds':nominal_step(z)})
            self.lengths.append(len(x))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]


def normalizer(df,ids,config):
    lookup=dict(zip(df.song_id,df.file));total=None;square=None;n=0
    for sid in ids:
        x=input_features(read_npz(lookup[sid]),config['representation'],config['delay'],config['harmonic']).astype('float64')
        if total is None:total=np.zeros(x.shape[1]);square=total.copy()
        total+=x.sum(0);square+=(x*x).sum(0);n+=len(x)
    mean=total/n;sd=np.sqrt(np.maximum(square/n-mean*mean,0)).clip(1e-4)
    return mean.astype('float32'),sd.astype('float32')


def run(manifest,reference,out,architecture='v1',representation='last',delay=0.,harmonic=False,
        long_valence=False,seed=2026,epochs=15,hidden=96,device='cuda',split_override=None):
    df=load_manifest(manifest);split=_reference_split(df,reference)
    if split_override is not None:
        allowed=set(split['train'])
        if not set(split_override['train']+split_override['val'])<=allowed:
            raise ValueError('Inner fold must use original training songs only')
        if set(split_override['train'])&set(split_override['val']):raise ValueError('Fold overlap')
        split={**split,**split_override}
    root=Path(out)
    if (root/'validation_summary.json').is_file():return json.loads((root/'validation_summary.json').read_text())
    root.mkdir(parents=True,exist_ok=True)
    resume_training=(root/'model.pt').is_file() and (root/'training_history.csv').is_file()
    if any(root.iterdir()) and not resume_training:raise FileExistsError(f'Incomplete run requires inspection: {root}')
    seed_everything(seed)
    config={'architecture':architecture,'representation':representation,'delay':delay,'harmonic':harmonic,
            'long_valence':long_valence,'seed':seed,'epochs':epochs,'hidden':hidden,
            'shape_version':'stable','optimizer':'AdamW lr=3e-4 weight_decay=.01',
            'selector':'equal-song Huber + .02 stable Pearson loss','patience':5,
            'train_ids':split['train'],'val_ids':split['val'],'test_set_used':False}
    if resume_training and json.loads((root/'protocol.json').read_text())!=config:
        raise ValueError('Resume configuration differs')
    (root/'protocol.json').write_text(json.dumps(config,indent=2))
    norm=normalizer(df,split['train'],config);np.savez(root/'normalizer.npz',mean=norm[0],std=norm[1])
    train=FeatureSongs(df,split['train'],config,norm);val=FeatureSongs(df,split['val'],config,norm)
    tr=DataLoader(train,batch_sampler=LengthBuckets(train.lengths,8,True,seed),collate_fn=pack)
    va=DataLoader(val,batch_sampler=LengthBuckets(val.lengths,8,False,seed),collate_fn=pack)
    acoustic_dim=75 if harmonic else 18
    layer_count=(len(norm[0])-acoustic_dim)//768
    model=LayerRegressor(architecture,acoustic_dim,layer_count,hidden=hidden,long_valence=long_valence).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    history=pd.read_csv(root/'training_history.csv').to_dict('records') if resume_training else []
    best=min(h['selector'] for h in history) if history else float('inf');bad=0
    for epoch in ([] if resume_training else range(epochs)):
        model.train();train_loss=[]
        for data in tr:
            b=on_device(data,device);opt.zero_grad(set_to_none=True)
            output=model(b['x'],b['times'],b['valid'])
            if architecture in ('trace','trace_no_cue_loss','trace_no_cues'):
                from .trace_mer import trace_supervised_loss,trace_variant_weights
                variant={'trace':'T3_cue_sufficiency','trace_no_cue_loss':'T1_no_cue_sufficiency',
                         'trace_no_cues':'T2_no_cues'}[architecture]
                loss=trace_supervised_loss(output,b,trace_variant_weights(variant))[0]
            else:loss=loss_for(output,b,'point_shape')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();train_loss.append(float(loss.detach()))
        model.eval();total=0.;n=0
        with torch.inference_mode():
            for data in va:
                b=on_device(data,device);score=selector(model(b['x'],b['times'],b['valid']),b)
                total+=float(score)*len(b['x']);n+=len(b['x'])
        score=total/n;history.append({'epoch':epoch+1,'train_loss':float(np.mean(train_loss)),'selector':score})
        print(f'{root.name} {architecture} {representation} delay={delay} epoch={epoch+1} selector={score:.5f}',flush=True)
        if score<best-1e-7:
            best=score;bad=0
            torch.save({'state_dict':model.state_dict(),'config':config,'acoustic_dim':acoustic_dim,
                        'layer_count':layer_count,'hidden':hidden},root/'model.pt')
        else:
            bad+=1
            if bad>=5:break
    pd.DataFrame(history).to_csv(root/'training_history.csv',index=False)
    model.load_state_dict(torch.load(root/'model.pt',map_location=device,weights_only=True)['state_dict']);model.eval()
    lookup=dict(zip(df.song_id,df.file));rows=[]
    pred_dir=root/'vp';pred_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for sid in split['val']:
            z=read_npz(lookup[sid]);x=(input_features(z,representation,delay,harmonic)-norm[0])/norm[1]
            xx=torch.as_tensor(x[None],dtype=torch.float32,device=device)
            tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device)
            valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
            pred=output_mean(model(xx,tt,valid))[0].cpu().numpy()
            rows.append({'song_id':sid,**song_metrics(pred,z['mean'],z['mask']),
                         **within_song_pearson(pred,z['mean'],z['mask'])})
            np.savez_compressed(pred_dir/f'{sid}.npz',pred=pred,truth=z['mean'],mask=z['mask'],times=z['times'])
    table=pd.DataFrame(rows);table.to_csv(root/'validation_by_song.csv',index=False)
    summary={'validation_metrics':{k:float(table[k].mean()) for k in table if k!='song_id'},
             'best_selector':best,'selected_epoch':int(np.argmin([h['selector'] for h in history])+1),
             'layer_weights':model.layer_logits.softmax(0).detach().cpu().tolist() if layer_count>1 else [1.],
             'exploratory':True,'test_set_used':False}
    (root/'validation_summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True);p.add_argument('--architecture',
        choices=('v1','echo_event','trace','trace_no_cue_loss','trace_no_cues'),default='v1')
    p.add_argument('--representation',choices=('last','mix'),default='last');p.add_argument('--delay',type=float,default=0.)
    p.add_argument('--harmonic',action='store_true');p.add_argument('--long-valence',action='store_true')
    p.add_argument('--seeds',nargs='+',type=int,default=[2026]);p.add_argument('--epochs',type=int,default=15)
    a=p.parse_args()
    for seed in a.seeds:
        run(a.manifest,a.reference,Path(a.out)/f'seed{seed}',a.architecture,a.representation,
            a.delay,a.harmonic,a.long_valence,seed,a.epochs)
