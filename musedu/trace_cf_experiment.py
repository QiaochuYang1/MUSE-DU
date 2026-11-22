"""Matched CF0-CF3 fine-tuning from seed-matched supervised TRACE checkpoints."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .experiment import load_manifest,read_npz,pack,on_device,seed_everything,nominal_step
from .metrics import song_metrics,within_song_pearson
from .matched_development import selector,output_mean
from .trace_mer import trace_supervised_loss,trace_counterfactual_loss
from .valence_experiment import input_features,FeatureSongs,LayerRegressor
from .v2_experiment import LengthBuckets,_reference_split

CONDITIONS=('CF0','CF1','CF2','CF3')


def _row(z,norm):
    x=input_features(z,'mix',1.,True)
    return {'x':(x-norm[0])/norm[1],**{k:z[k] for k in ('times','mean','std','mask','std_mask')},
            'step_seconds':nominal_step(z)}


class Pairs:
    def __init__(self,table,group,norm):
        self.rows=[]
        for row in table[table.target_group==group].itertuples():
            original=read_npz(row.original_file);modified=read_npz(row.modified_file)
            self.rows.append({'factual':_row(original,norm),'altered':_row(modified,norm),
                              'region':modified['intervention_region'].astype(bool),'group':group})
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]


def collate_pairs(rows):
    factual=pack([r['factual'] for r in rows]);altered=pack([r['altered'] for r in rows])
    region=torch.zeros_like(factual['valid'])
    for i,row in enumerate(rows):region[i,:len(row['region'])]=torch.from_numpy(row['region'])
    return {'factual':factual,'altered':altered,'region':region,'group':rows[0]['group']}


def run(manifest,reference,interventions,base,out,condition,seed,epochs=5,device='cuda',cf_weight=.1):
    if condition not in CONDITIONS:raise ValueError(condition)
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    if (root/'validation_summary.json').is_file():return json.loads((root/'validation_summary.json').read_text())
    df=load_manifest(manifest);split=_reference_split(df,reference);seed_everything(seed)
    base=Path(base)/f'seed{seed}'
    with np.load(base/'normalizer.npz') as z:norm=(z['mean'],z['std'])
    train=FeatureSongs(df,split['train'],{'representation':'mix','delay':1.,'harmonic':True},norm)
    val=FeatureSongs(df,split['val'],{'representation':'mix','delay':1.,'harmonic':True},norm)
    tr=DataLoader(train,batch_sampler=LengthBuckets(train.lengths,8,True,seed),collate_fn=pack)
    va=DataLoader(val,batch_sampler=LengthBuckets(val.lengths,8,False,seed),collate_fn=pack)
    pair_table=pd.read_csv(interventions,dtype={'song_id':str});pair_table=pair_table[pair_table.partition=='train']
    pair_loaders={g:DataLoader(Pairs(pair_table,g,norm),batch_size=8,shuffle=True,collate_fn=collate_pairs)
                  for g in ('energy','spectrum')}
    model=LayerRegressor('trace',75,4,hidden=96).to(device)
    model.load_state_dict(torch.load(base/'model.pt',map_location=device,weights_only=True)['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    protocol={'condition':condition,'base_checkpoint':str(base.resolve()),'seed':seed,'epochs':epochs,
              'cf_weight':cf_weight,'interventions':len(pair_table),'architecture':'T3',
              'pair_step':'full factual loss for every condition; altered forward for CF1-CF3; correct CF loss CF2; swapped cue CF3',
              'test_set_used':False}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));history=[];best=float('inf')
    mismatch={'energy':'spectrum','spectrum':'energy'}
    for epoch in range(epochs):
        model.train();losses=[]
        for data in tr:
            b=on_device(data,device);optimizer.zero_grad(set_to_none=True);output=model(b['x'],b['times'],b['valid'])
            loss=trace_supervised_loss(output,b)[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        for group,loader in pair_loaders.items():
            for data in loader:
                f=on_device(data['factual'],device);a=on_device(data['altered'],device);region=data['region'].to(device)
                optimizer.zero_grad(set_to_none=True);fo=model(f['x'],f['times'],f['valid']);loss=trace_supervised_loss(fo,f)[0]
                if condition!='CF0':
                    ao=model(a['x'],a['times'],a['valid'])
                    if condition in ('CF2','CF3'):
                        assigned=group if condition=='CF2' else mismatch[group]
                        cf,_=trace_counterfactual_loss(fo,ao,assigned,region,torch.ones(len(region),dtype=torch.bool,device=device),leak_weight=0.)
                        loss=loss+cf_weight*cf
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        model.eval();total=0.;n=0
        with torch.inference_mode():
            for data in va:
                b=on_device(data,device);score=selector(model(b['x'],b['times'],b['valid']),b);total+=float(score)*len(b['x']);n+=len(b['x'])
        score=total/n;history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'selector':score})
        print(f'{condition} seed{seed} epoch={epoch+1} selector={score:.5f}',flush=True)
        if score<best:
            best=score;torch.save({'state_dict':model.state_dict(),'condition':condition,'seed':seed},root/'model.pt')
    pd.DataFrame(history).to_csv(root/'training_history.csv',index=False)
    model.load_state_dict(torch.load(root/'model.pt',map_location=device,weights_only=True)['state_dict']);model.eval()
    lookup=dict(zip(df.song_id,df.file));rows=[];pred_dir=root/'vp';pred_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for sid in split['val']:
            z=read_npz(lookup[sid]);x=(input_features(z,'mix',1.,True)-norm[0])/norm[1]
            xx=torch.as_tensor(x[None],dtype=torch.float32,device=device);tt=torch.as_tensor(z['times'][None],dtype=torch.float32,device=device);valid=torch.ones((1,len(x)),dtype=torch.bool,device=device)
            pred=output_mean(model(xx,tt,valid))[0].cpu().numpy();rows.append({'song_id':sid,**song_metrics(pred,z['mean'],z['mask']),**within_song_pearson(pred,z['mean'],z['mask'])})
            np.savez_compressed(pred_dir/f'{sid}.npz',pred=pred,truth=z['mean'],mask=z['mask'],times=z['times'])
    table=pd.DataFrame(rows);table.to_csv(root/'validation_by_song.csv',index=False)
    summary={'validation_metrics':{k:float(table[k].mean()) for k in table if k!='song_id'},'best_selector':best,
             'selected_epoch':int(np.argmin([h['selector'] for h in history])+1),'test_set_used':False}
    (root/'validation_summary.json').write_text(json.dumps(summary,indent=2));return summary


def main(manifest,reference,interventions,base,out,device='cuda'):
    for condition in CONDITIONS:
        for seed in range(2026,2031):run(manifest,reference,interventions,base,Path(out)/condition/f'seed{seed}',condition,seed,device=device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--reference',required=True);p.add_argument('--interventions',required=True)
    p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();main(a.manifest,a.reference,a.interventions,a.base,a.out,a.device)
