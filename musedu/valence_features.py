"""Versioned development-only overlap/layer cache; original caches stay intact."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .features import _audio, _bin_features, _load_mert, MERT_REVISION
from .experiment import load_manifest, read_npz
from .v2_experiment import _reference_split

LAYERS=(3,6,9,12)


def convolution_centers(config,n,sr):
    stride=1; receptive=1
    for kernel,step in zip(config.conv_kernel,config.conv_stride):
        receptive+=(kernel-1)*stride;stride*=step
    return ((receptive-1)/2+np.arange(n)*stride)/sr


def pool_frames(values,frame_times,bin_times,fps=2.):
    """Pool real frame centers inside each label bin; no point interpolation."""
    half=.5/fps
    result=np.zeros((len(bin_times),)+values.shape[1:],dtype='float32')
    weights=np.zeros(len(bin_times),dtype='float32')
    for i,t in enumerate(bin_times):
        keep=(frame_times>=t-half)&(frame_times<t+half)
        if keep.any():
            result[i]=values[keep].mean(0);weights[i]=1
    return result,weights


def overlap_layers(y,sr,times,device='cuda',layers=LAYERS,seconds=5.,hop_seconds=2.5):
    model,processor=_load_mert('m-a-p/MERT-v1-95M',device)
    if sr!=processor.sampling_rate:raise ValueError('MERT sample rate mismatch')
    totals=None;denom=np.zeros(len(times),dtype='float64')
    size=int(seconds*sr);hop=int(hop_seconds*sr)
    with torch.inference_mode():
        for start in range(0,len(y),hop):
            chunk=y[start:start+size]
            if len(chunk)<sr//10:continue
            begin=start/sr;end=(start+len(chunk))/sr
            if end<times[0]-.25 or begin>times[-1]+.25:continue
            inp=processor(chunk,sampling_rate=sr,return_tensors='pt',padding=False)
            out=model(**{k:v.to(device) for k,v in inp.items()},output_hidden_states=True)
            z=torch.stack([out.hidden_states[layer][0] for layer in layers],dim=1).float().cpu().numpy()
            frame_t=begin+convolution_centers(model.config,len(z),sr)
            pooled,present=pool_frames(z,frame_t,times)
            # Favor the context-rich center when multiple windows cover a bin.
            taper=np.maximum(.05,np.minimum(times-begin,end-times)/(seconds/2)).clip(0,1)
            weight=present*taper
            if totals is None:totals=np.zeros_like(pooled,dtype='float64')
            totals+=pooled*weight[:,None,None];denom+=weight
    if totals is None or np.any(denom==0):raise ValueError('Uncovered annotation bins')
    return (totals/denom[:,None,None]).astype('float32')


def harmonic_descriptors(acoustic):
    """Transparent chord/key template similarities, tonal centroids and changes.

    Template margins are acoustic evidence, not calibrated key probabilities.
    """
    c=np.maximum(acoustic[:,6:18],0).astype('float64')
    c=c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-8)
    chord=[];keys=[]
    profiles=[np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]),
              np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])]
    for minor in (False,True):
        base=np.zeros(12);base[[0,3 if minor else 4,7]]=1
        for root in range(12):
            chord.append(np.roll(base,root));keys.append(np.roll(profiles[int(minor)],root))
    chord=np.stack(chord);chord/=np.linalg.norm(chord,axis=1,keepdims=True)
    keys=np.stack(keys);keys-=keys.mean(1,keepdims=True)
    keys/=np.linalg.norm(keys,axis=1,keepdims=True)
    cs=c@chord.T
    slow=np.stack([c[max(0,i-19):i+1].mean(0) for i in range(len(c))])
    slow-=slow.mean(1,keepdims=True);slow/=np.maximum(np.linalg.norm(slow,axis=1,keepdims=True),1e-8)
    ks=slow@keys.T
    pitch=np.arange(12)
    tonal_basis=np.stack([np.cos(7*np.pi*pitch/6),np.sin(7*np.pi*pitch/6),
                          np.cos(3*np.pi*pitch/2),np.sin(3*np.pi*pitch/2),
                          .5*np.cos(2*np.pi*pitch/3),.5*np.sin(2*np.pi*pitch/3)])
    tonal=(c/np.maximum(c.sum(1,keepdims=True),1e-8))@tonal_basis.T
    novelty=np.r_[0,np.linalg.norm(np.diff(c,axis=0),axis=1)]
    sorted_chord=np.sort(cs,axis=1)
    margin=sorted_chord[:,-1]-sorted_chord[:,-2]
    changed=np.r_[0,(np.diff(cs.argmax(1))!=0).astype(float)]*margin
    result=np.column_stack([cs,ks,tonal,novelty,margin,changed]).astype('float32')
    if not np.isfinite(result).all():raise ValueError('Nonfinite harmonic descriptors')
    return result


def prepare(manifest,reference_run,out,device='cuda',limit=None,shard_index=0,shards=1):
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('NUMBA_DISABLE_JIT','1')
    df=load_manifest(manifest);split=_reference_split(df,reference_run)
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    runtime=(root/'runtime').resolve();runtime.mkdir(exist_ok=True)
    os.environ['NUMBA_CACHE_DIR']=str(runtime/'numba')
    tempfile.tempdir=str(runtime)
    torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    ids=(split['train']+split['val'])[shard_index::shards];lookup=df.set_index('song_id')
    if limit is not None:ids=ids[:limit]
    records=[]
    for i,sid in enumerate(ids):
        target=root/'features'/f'{sid}.npz';target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file():
            with np.load(target,allow_pickle=False) as cached:
                if 'context_layers' not in cached:raise ValueError(f'Incomplete cached feature {target}')
        if not target.is_file():
            row=lookup.loc[sid];old=read_npz(row.file);y=_audio(row.audio,24000)
            digest=hashlib.sha256(np.asarray(y,dtype='<f4').tobytes()).hexdigest()
            if digest!=str(old['audio_sha256'].item()):raise ValueError(f'Audio identity changed {sid}')
            times=np.arange(max(0,float(old['times'][0])-4),float(old['times'][-1])+.01,.5,dtype='float32')
            layers=overlap_layers(y,24000,times,device)
            acoustic=_bin_features(y,24000,times,2.)
            harmonic=harmonic_descriptors(acoustic)
            index=np.searchsorted(times,old['times'])
            if not np.allclose(times[index],old['times']):raise ValueError('Timestamp mapping mismatch')
            data={**old,'acoustic':acoustic[index],'embedding':layers[index,-1],
                  'embedding_layers':layers[index], 'harmonic':harmonic[index],
                  'context_times':times,'context_layers':layers,'context_acoustic':acoustic,
                  'context_harmonic':harmonic,'mert_layers':np.array(LAYERS),
                  'mert_revision':np.array(MERT_REVISION),'feature_version':np.array('overlap_bin_v1')}
            temporary=target.with_suffix('.partial.npz')
            np.savez_compressed(temporary,**data);temporary.replace(target)
        records.append({'song_id':sid,'file':str(target.resolve())})
        if (i+1)%25==0 or i==0:print(f'cache {i+1}/{len(ids)}: {sid}',flush=True)
    mapping={r['song_id']:r['file'] for r in records}
    result=df.copy();result['file']=[mapping.get(sid,file) for sid,file in zip(df.song_id,df.file)]
    suffix='' if shards==1 else f'_shard{shard_index}'
    result.to_csv(root/f'deam_manifest{suffix}.csv',index=False)
    (root/f'feature_protocol{suffix}.json').write_text(json.dumps({
        'layers':LAYERS,'mert_revision':MERT_REVISION,'chunk_seconds':5,'hop_seconds':2.5,
        'pooling':'annotation bins, tapered across overlapping windows',
        'new_feature_song_ids':ids,'context_preroll_seconds':4,
        'heldout_features':'original files retained; no scale/cal/test re-extraction',
        'harmonic':'24 chord and 24 key template similarities; 6 tonal centroids; novelty/margin/change'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();prepare(a.manifest,a.reference_run,a.out,a.device,a.limit,a.shard_index,a.shards)
