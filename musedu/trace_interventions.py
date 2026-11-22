"""Duration-preserving TRACE intervention generation and selectivity audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from .experiment import load_manifest,read_npz
from .features import _audio,_bin_features
from .valence_features import harmonic_descriptors
from .v2_experiment import _reference_split
from .trace_mer import DEFAULT_GROUPS


def _fade(n,sr,seconds=.1):
    length=min(int(seconds*sr),n//2);w=np.ones(n,dtype='float32')
    if length:
        ramp=np.sin(np.linspace(0,np.pi/2,length,dtype='float32'))**2
        w[:length]=ramp;w[-length:]=ramp[::-1]
    return w


def intervene(y,sr,start,end,kind):
    """Return altered waveform without changing length or alignment."""
    altered=np.asarray(y,dtype='float32').copy();a=max(0,int(start*sr));b=min(len(y),int(end*sr))
    if b-a<sr:raise ValueError('Intervention region is too short')
    original=altered[a:b].copy();blend=_fade(len(original),sr)
    if kind=='energy':
        changed=.5*original
    elif kind=='spectrum':
        changed=sosfiltfilt(butter(6,3000,btype='lowpass',fs=sr,output='sos'),original).astype('float32')
        changed*=np.sqrt(np.mean(original**2)+1e-10)/np.sqrt(np.mean(changed**2)+1e-10)
    else:raise ValueError(kind)
    altered[a:b]=original*(1-blend)+changed*blend
    return altered


def select_region(acoustic,times,kind,seconds=4.):
    bins=max(2,int(round(seconds/.5)));score=acoustic[:,0] if kind=='energy' else acoustic[:,4]
    rolling=np.convolve(score,np.ones(bins)/bins,mode='valid')
    index=int(np.argmax(rolling));return float(times[index]-.25),float(times[min(len(times)-1,index+bins-1)]+.25)


def descriptor_matrix(y,sr,times):
    acoustic=_bin_features(y,sr,times,2.)
    return np.concatenate((acoustic,harmonic_descriptors(acoustic)),axis=1)


def audit(manifest,reference,normalizer,out,train_count=40,val_count=20,seed=2026):
    df=load_manifest(manifest);split=_reference_split(df,reference);lookup=df.set_index('song_id')
    with np.load(normalizer) as z:mean=z['mean'][:75];sd=z['std'][:75]
    rng=np.random.default_rng(seed);selection={
        'train':list(map(str,rng.choice(split['train'],min(train_count,len(split['train'])),replace=False))),
        'evaluation':list(map(str,rng.choice(split['val'],min(val_count,len(split['val'])),replace=False)))}
    rows=[]
    for partition,ids in selection.items():
        for sid in ids:
            row=lookup.loc[sid];z=read_npz(row.file);times=z['context_times'];base=np.concatenate((z['context_acoustic'],z['context_harmonic']),1)
            y=_audio(row.audio,24000)
            for target in ('energy','spectrum'):
                # Choose edits only on the labeled timeline; context_times also
                # contains four seconds of preroll that cannot be evaluated.
                start,end=select_region(z['acoustic'],z['times'],target);changed=intervene(y,24000,start,end,target)
                modified=descriptor_matrix(changed,24000,times)
                region=(times>=start)&(times<end);deltas={}
                for group,indices in DEFAULT_GROUPS.items():
                    difference=(modified[:,indices]-base[:,indices])/sd[list(indices)]
                    deltas[group]=float(np.sqrt(np.mean(difference[region]**2)))
                ratio=sum(v for k,v in deltas.items() if k!=target)/(deltas[target]+1e-8)
                rows.append({'partition':partition,'song_id':sid,'target_group':target,
                             'start_seconds':start,'end_seconds':end,'off_target_ratio':ratio,**deltas})
            print(f'intervention audit {partition} {sid}',flush=True)
    root=Path(out);root.mkdir(parents=True,exist_ok=True);table=pd.DataFrame(rows)
    table.to_csv(root/'selectivity_measurements.csv',index=False)
    training=table[table.partition=='train'];thresholds={}
    for target in ('energy','spectrum'):
        part=training[training.target_group==target]
        thresholds[target]={'delta_min':float(part[target].quantile(.25)),
                            'eta_max':float(part.off_target_ratio.quantile(.5))}
    accepted=[]
    for _,row in table.iterrows():
        rule=thresholds[row.target_group]
        accepted.append(bool(row[row.target_group]>=rule['delta_min'] and row.off_target_ratio<=rule['eta_max']))
    table['accepted']=accepted;table.to_csv(root/'selectivity_measurements.csv',index=False)
    summary={'selection':selection,'threshold_rule':'training target-change Q25 and off-target-ratio median, fixed before evaluation subset',
             'thresholds':thresholds,'acceptance':{f'{p}_{g}':float(table[(table.partition==p)&(table.target_group==g)].accepted.mean())
                for p in selection for g in ('energy','spectrum')},
             'interventions':{'energy':'local 0.5 gain with 100 ms squared-sine fades',
                              'spectrum':'local 3 kHz sixth-order low-pass, RMS matched, 100 ms fades'},
             'group_change':'RMS standardized descriptor change within edited region',
             'mert_recomputed':False,'purpose':'pre-MERT selectivity gate only'}
    (root/'selectivity_summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--normalizer',required=True);p.add_argument('--out',required=True)
    p.add_argument('--train-count',type=int,default=40);p.add_argument('--val-count',type=int,default=20)
    a=p.parse_args();print(json.dumps(audit(a.manifest,a.reference,a.normalizer,a.out,a.train_count,a.val_count),indent=2))
