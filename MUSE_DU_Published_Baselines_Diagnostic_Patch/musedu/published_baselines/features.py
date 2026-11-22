"""Extract 2 Hz paper-inspired descriptors / frozen Transformer layer features.

Mazzola et al. report 988 OpenSMILE emobase_v2 functionals, 77 Librosa
features, 20 MoSQITo-derived PAA features. We reproduce available recipes,
not claim bitwise exact original feature sets or author trained checkpoints.
The full 20-D PAA recipe requires its separately validated MoSQITo extractor;
we import supplied, aligned NPZs rather than fabricate psychoacoustics.
"""
from __future__ import annotations
import argparse
import os
import time
import faulthandler
from pathlib import Path
import numpy as np
from .io import load_index,load_song,feature_archive_path


def _msg(message):
    print(f"[feature-extract {time.strftime('%H:%M:%S')}] {message}", flush=True)


def decode_audio(path, target_sr=44100, decoder='soundfile'):
    """Decode full audio with explicit backend; resample via SoXR (if needed).

    SoundFile avoids initializing Librosa merely for audio I/O. Output is
    floating-point mono audio. Set --decoder librosa to retain old I/O.
    """
    if decoder == 'librosa':
        _msg('START import librosa (decoder)')
        import librosa
        _msg('DONE import librosa (decoder)')
        _msg('START librosa.load')
        y,sr=librosa.load(str(path),sr=target_sr,mono=True)
        _msg('DONE librosa.load')
        return y,sr
    if decoder != 'soundfile': raise ValueError(decoder)
    _msg('START soundfile.read')
    import soundfile as sf
    y,sr=sf.read(str(path),dtype='float32',always_2d=True)
    y=y.mean(axis=1,dtype=np.float32)
    _msg(f'DONE soundfile.read sr={sr}, n={len(y)}')
    if sr!=target_sr:
        _msg(f'START soxr resample {sr} -> {target_sr}')
        try: import soxr
        except ImportError as exc:
            raise RuntimeError('Install soxr for non-44.1 kHz source files: pip install soxr') from exc
        y=soxr.resample(y,sr,target_sr).astype('float32',copy=False)
        _msg('DONE soxr resample')
    if not np.isfinite(y).all() or not len(y):
        raise ValueError(f'{path}: empty or non-finite decoded audio')
    return y,target_sr


def find_audio(audio_root, sid):
    from musedu.datasets import find_audio as resolve
    return resolve(Path(audio_root),sid)


def frame_features_librosa(audio,sr,times):
    """77-D deterministic adaptation, not a bitwise reconstruction of Table 1."""
    _msg('START import librosa (77-D features)')
    import librosa
    _msg(f'DONE import librosa version={librosa.__version__}')
    hop=512; nfft=2048
    _msg('START stft')
    stft=librosa.stft(audio,n_fft=nfft,hop_length=hop)
    _msg('DONE stft')
    mag=np.abs(stft)
    power=mag**2
    _msg('START mel spectrogram')
    mel=librosa.feature.melspectrogram(S=power,sr=sr,n_mels=128)
    _msg('DONE mel spectrogram')
    _msg('START MFCC')
    mfcc=librosa.feature.mfcc(S=librosa.power_to_db(mel+1e-9),n_mfcc=13)
    _msg('DONE MFCC')
    _msg('START chroma')
    chroma=librosa.feature.chroma_stft(S=mag,sr=sr,n_fft=nfft,hop_length=hop)
    _msg('DONE chroma')
    _msg('START spectral contrast')
    contrast=librosa.feature.spectral_contrast(S=mag,sr=sr)
    _msg('DONE spectral contrast')
    _msg('START spectral centroid')
    centroid=librosa.feature.spectral_centroid(S=mag,sr=sr)
    _msg('DONE spectral centroid')
    _msg('START spectral bandwidth')
    bandwidth=librosa.feature.spectral_bandwidth(S=mag,sr=sr)
    _msg('DONE spectral bandwidth')
    _msg('START spectral rolloff')
    rolloff=librosa.feature.spectral_rolloff(S=mag,sr=sr)
    _msg('DONE spectral rolloff')
    _msg('START spectral flatness')
    flat=librosa.feature.spectral_flatness(S=mag)
    _msg('DONE spectral flatness')
    _msg('START zero crossing')
    zcr=librosa.feature.zero_crossing_rate(audio,frame_length=nfft,hop_length=hop)
    _msg('DONE zero crossing')
    _msg('START RMS')
    rms=librosa.feature.rms(S=mag)
    _msg('DONE RMS')
    _msg('START onset')
    onset=librosa.onset.onset_strength(y=audio,sr=sr,hop_length=hop)
    _msg('DONE onset')
    _msg('START 77-D aggregation over label timestamps')
    # Causal-ish local estimated tempo from two-second onset buffer.
    frame_t=librosa.frames_to_time(np.arange(mag.shape[1]),sr=sr,hop_length=hop)
    result=[]
    for t in times:
        idx=np.flatnonzero((frame_t>=t-.25)&(frame_t<t+.25))
        if not len(idx):idx=np.array([int(np.argmin(np.abs(frame_t-t)))])
        def m(x):return x[...,np.clip(idx,0,x.shape[-1]-1)].mean(-1)
        def s(x):return x[...,np.clip(idx,0,x.shape[-1]-1)].std(-1)
        def ran(x):
            y=x[...,np.clip(idx,0,x.shape[-1]-1)];return y.max(-1)-y.min(-1)
        # 39 MFCC mean/std/range, 24 chroma mean/std, 7 spectral contrast,
        # 7 spectral/energy features (incl. local onset strength).
        v=np.concatenate([m(mfcc),s(mfcc),ran(mfcc),m(chroma),s(chroma),
            m(contrast),[float(m(q)[0]) for q in
                 [centroid,bandwidth,rolloff,flat,zcr,rms,onset.reshape(1,-1)]]])
        if len(v)!=77:raise AssertionError('Librosa feature dimension not 77')
        result.append(v)
    arr=np.asarray(result,dtype=np.float32)
    _msg(f'DONE 77-D aggregation shape={arr.shape}')
    return arr


def opensmile_features(audio,sr,times):
    """emobase 988-D functionals on label-centered 0.5s segments."""
    try:import opensmile
    except ImportError as e:raise RuntimeError('Install opensmile for OpenSMILE baseline') from e
    smile=opensmile.Smile(feature_set=opensmile.FeatureSet.emobase,
                          feature_level=opensmile.FeatureLevel.Functionals)
    rows=[]
    for t in times:
        a=max(0,int(round((float(t)-.25)*sr)))
        b=min(len(audio),int(round((float(t)+.25)*sr)))
        if b-a<sr//20:raise ValueError('Audio segment too short: alignment problem')
        value=smile.process_signal(audio[a:b],sr).to_numpy(dtype=np.float32)
        if value.shape[0]!=1:raise ValueError('Unexpected OpenSMILE output')
        rows.append(value[0])
    if len(rows[0])!=988:
        raise ValueError(f'OpenSMILE returned {len(rows[0])} features, paper uses 988; verify opensmile/emobase version')
    return np.asarray(rows,dtype=np.float32)


def pretrained_layer_features(audio,sr,times,kind,model_name,device='cuda',chunk_s=5.):
    """Frozen all-hidden-layer features per song [T,L,D]. Not fit to labels."""
    import torch
    from transformers import AutoModel,AutoFeatureExtractor
    target_sr=24000 if kind=='mert' else 16000
    import librosa
    if sr!=target_sr:audio=librosa.resample(audio,orig_sr=sr,target_sr=target_sr)
    processor=AutoFeatureExtractor.from_pretrained(model_name,trust_remote_code=(kind=='mert'))
    model=AutoModel.from_pretrained(model_name,trust_remote_code=(kind=='mert'))
    model=model.to(device).eval()
    model.requires_grad_(False)
    samples=int(target_sr*chunk_s)
    stamps=[];blocks=[]
    with torch.inference_mode():
        for start in range(0,len(audio),samples):
            segment=audio[start:start+samples]
            if len(segment)<int(target_sr*.15):continue
            batch=processor(segment,sampling_rate=target_sr,return_tensors='pt',padding=False)
            x={k:v.to(device) for k,v in batch.items()}
            res=model(**x,output_hidden_states=True)
            if not res.hidden_states:raise RuntimeError('Encoder did not expose hidden_states')
            # Exclude embedding layer at index 0, keep all learned transformer layers.
            arr=torch.stack(res.hidden_states[1:],dim=0) # L,1,S,D
            arr=arr[:,0].permute(1,0,2).float().cpu().numpy() # S,L,D
            stamp=start/target_sr+(np.arange(len(arr))+.5)*len(segment)/target_sr/len(arr)
            stamps.append(stamp);blocks.append(arr)
    if not blocks:raise ValueError('No valid encoder chunks')
    all_frames=np.concatenate(blocks,axis=0);all_times=np.concatenate(stamps)
    out=np.empty((len(times),all_frames.shape[1],all_frames.shape[2]),np.float32)
    for l in range(out.shape[1]):
        for d in range(out.shape[2]):
            out[:,l,d]=np.interp(times,all_times,all_frames[:,l,d],left=all_frames[0,l,d],right=all_frames[-1,l,d])
    del model
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return out


def mel_features(audio,sr,times):
    import librosa
    mel=librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=2048,hop_length=512,n_mels=128)
    db=librosa.power_to_db(mel+1e-10,ref=1.0)
    ft=librosa.frames_to_time(np.arange(db.shape[1]),sr=sr,hop_length=512)
    out=[]
    for t in times:
        ix=(ft>=t-.25)&(ft<t+.25)
        if not ix.any():ix[np.argmin(abs(ft-t))]=True
        out.append(db[:,ix].mean(axis=1))
    return np.array(out,np.float32)


def prepare(manifest,reference_run,audio_root,features_dir,views,device='cpu',mert_model='m-a-p/MERT-v1-330M',hubert_model='facebook/hubert-base-ls960',overwrite=False,decoder='soundfile',limit_songs=0):
    _msg('START loading manifest and B2 song splits')
    index,splits=load_index(manifest,reference_run)
    _msg('DONE loading manifest and B2 song splits')
    # Only train+val are processed. Keep old test, scale and calibration untouched.
    ids=list(splits['train'])+list(splits['val'])
    if limit_songs>0:ids=ids[:limit_songs]
    _msg(f'Will inspect {len(ids)} songs, views={views}; skipping existing files')
    for i,sid in enumerate(ids,1):
        z=load_song(index,sid)
        pending=[view for view in views if overwrite or not feature_archive_path(features_dir,view,sid).exists()]
        if not pending:continue
        _msg(f'START song={sid} ({i}/{len(ids)}) pending={pending}')
        audio_path=find_audio(audio_root,sid)
        audio,sr=decode_audio(audio_path,target_sr=44100,decoder=decoder)
        for kind in pending:
            _msg(f'START kind={kind} song={sid}')
            if kind=='librosa':feat=frame_features_librosa(audio,sr,z['times'])
            elif kind=='opensmile':feat=opensmile_features(audio,sr,z['times'])
            elif kind=='mel':feat=mel_features(audio,sr,z['times'])
            elif kind in ('mert','hubert'):
                name=mert_model if kind=='mert' else hubert_model
                feat=pretrained_layer_features(audio,sr,z['times'],kind,name,device=device,chunk_s=5. if kind=='mert' else 10.)
            elif kind=='paa':
                raise RuntimeError('PAA requires independently validated 20-D MoSQITo-derived features: add <features-dir>/paa/<song_id>.npz with times and features; do not fabricate PAA numbers')
            else:raise ValueError('Unknown feature family: '+kind)
            dest=feature_archive_path(features_dir,kind,sid)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not np.isfinite(feat).all(): raise ValueError(f'{kind} {sid}: nonfinite features')
            if len(feat)!=len(z['times']): raise ValueError(f'{kind} {sid}: wrong number of labels')
            np.savez_compressed(dest,times=z['times'],features=feat)
            _msg(f'DONE kind={kind} song={sid} shape={feat.shape} saved={dest}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True)
    p.add_argument('--audio-root',required=True)
    p.add_argument('--features-dir',required=True)
    p.add_argument('--views',nargs='+',choices=['opensmile','librosa','mel','mert','hubert'],required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--mert-model',default='m-a-p/MERT-v1-330M')
    p.add_argument('--hubert-model',default='facebook/hubert-base-ls960')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--decoder',choices=['soundfile','librosa'],default='soundfile',
                   help='SoundFile+SoXR avoids Librosa in audio loading; Librosa features are unchanged')
    p.add_argument('--limit-songs',type=int,default=0,help='Debug first N train/val songs only; 0 means all')
    p.add_argument('--trace-after',type=int,default=60,help='Dump stack trace every N seconds if stuck; 0 disables')
    p.add_argument('--numba-cache-dir',default=None,help='Explicit writable Numba cache directory')
    a=p.parse_args()
    if a.numba_cache_dir:
        c=Path(a.numba_cache_dir).expanduser().resolve()
        c.mkdir(parents=True,exist_ok=True)
        os.environ['NUMBA_CACHE_DIR']=str(c)
        _msg(f'NUMBA_CACHE_DIR={c}')
    if a.trace_after>0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(a.trace_after,repeat=True)
    try:
        prepare(a.manifest,a.reference_run,a.audio_root,a.features_dir,a.views,a.device,
                a.mert_model,a.hubert_model,a.overwrite,a.decoder,a.limit_songs)
    finally:
        faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
