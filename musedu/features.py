"""Offline audio feature preparation. MERT is optional, frozen, and cached to disk."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import hashlib
import numpy as np

FEATURE_NAMES=['rms','centroid','bandwidth','rolloff','flux','onset']+[f'chroma_{i}' for i in range(12)]
MERT_REVISION='12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'


def _audio(path, sr=24000):
    import librosa
    y,_=librosa.load(str(path),sr=sr,mono=True)
    if y.size==0 or not np.isfinite(y).all():raise ValueError(f'{path}: invalid audio')
    return y


def _bin_features(y,sr,times,fps):
    import librosa
    hop=512
    s=np.abs(librosa.stft(y,n_fft=2048,hop_length=hop,center=True))
    rms=librosa.feature.rms(S=s).ravel()
    centroid=librosa.feature.spectral_centroid(S=s,sr=sr).ravel()
    width=librosa.feature.spectral_bandwidth(S=s,sr=sr).ravel()
    roll=librosa.feature.spectral_rolloff(S=s,sr=sr).ravel()
    flux=np.r_[0,np.maximum(0,np.diff(s,axis=1)).mean(axis=0)]
    onset=librosa.onset.onset_strength(S=librosa.amplitude_to_db(s+1e-8,ref=np.max),sr=sr,hop_length=hop)
    chroma=librosa.feature.chroma_stft(S=s,sr=sr)
    n=s.shape[1]
    def match(x):
        z=np.asarray(x)
        if z.ndim==1:z=z[None,:]
        if z.shape[1]<n:z=np.pad(z,((0,0),(0,n-z.shape[1])),mode='edge')
        return z[:,:n]
    frame=match(np.vstack([rms,centroid,width,roll,flux]))
    frame=np.vstack([frame,match(onset),match(chroma)])
    frame_t=librosa.frames_to_time(np.arange(n),sr=sr,hop_length=hop)
    out=np.full((len(times),len(FEATURE_NAMES)),np.nan,np.float32)
    half=.5/fps
    for k,t in enumerate(times):
        ix=(frame_t>=t-half)&(frame_t<t+half)
        if not np.any(ix):ix[np.argmin(np.abs(frame_t-t))]=True
        out[k]=frame[:,ix].mean(axis=1)
    return out


@lru_cache(maxsize=2)
def _load_mert(model_name,device):
    import torch
    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise RuntimeError('Install optional extra: pip install -e .[mert]') from exc
    model=AutoModel.from_pretrained(model_name,revision=MERT_REVISION,
                                    trust_remote_code=True).eval().to(device)
    processor=Wav2Vec2FeatureExtractor.from_pretrained(model_name,revision=MERT_REVISION,
                                                       trust_remote_code=True)
    for p in model.parameters():p.requires_grad_(False)
    return model,processor


def _mert_embeddings(y,sr,times,chunk_seconds=5.,device='cpu',model_name='m-a-p/MERT-v1-95M'):
    """Run frozen MERT in short nonoverlapping chunks; timestamp-aligned interpolation.

    Model card permits 5s context; avoid treating padded tail representations as valid.
    Requires explicit model trust; user controls installed environment and model revision.
    """
    import torch
    model,processor=_load_mert(model_name,device)
    if int(processor.sampling_rate)!=sr:
        raise ValueError(f'MERT expects {processor.sampling_rate}, got {sr}')
    blocks=[]; stamps=[]
    samples=int(chunk_seconds*sr)
    with torch.inference_mode():
        for start in range(0,len(y),samples):
            chunk=y[start:start+samples]
            if len(chunk)<sr//10:continue
            inp=processor(chunk,sampling_rate=sr,return_tensors='pt',padding=False)
            inp={k:v.to(device) for k,v in inp.items()}
            z=model(**inp).last_hidden_state[0].detach().float().cpu().numpy()
            block_start=start/sr; duration=len(chunk)/sr
            # output vectors approximate equal-spaced centers of this chunk
            tm=block_start+(np.arange(len(z))+.5)*duration/len(z)
            blocks.append(z);stamps.append(tm)
    if not blocks:raise ValueError('Audio is too short for MERT')
    e=np.concatenate(blocks); t=np.concatenate(stamps)
    return np.stack([np.interp(times,t,e[:,j],left=e[0,j],right=e[-1,j])
                     for j in range(e.shape[1])],axis=-1).astype(np.float32)


def prepare_one(ann,audio_path,out_path,*,backend='acoustic',sr=24000,fps=2.,device='cpu'):
    y=_audio(audio_path,sr=sr)
    # Preserve all annotation timestamps; mark labels beyond available audio as missing.
    time_valid=(ann.times>=0)&(ann.times<len(y)/sr)
    if not time_valid.any():raise ValueError(f'{ann.song_id}: annotation/audio intervals disjoint')
    idx=np.where(time_valid)[0]
    times=ann.times[idx].astype(np.float32)
    acoustic=_bin_features(y,sr,times,fps)
    if backend=='mert':embedding=_mert_embeddings(y,sr,times,device=device)
    elif backend=='acoustic':embedding=np.empty((len(times),0),np.float32)
    else:raise ValueError(backend)
    if not np.isfinite(acoustic).all() or not np.isfinite(embedding).all():
        raise ValueError('Non-finite audio features')
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    # Fingerprint canonical decoded 24 kHz mono audio, rather than container bytes.
    # This detects identical audio exported in different lossless containers, not
    # remastered, cropped or perceptually similar tracks.
    audio_sha256=hashlib.sha256(np.asarray(y,dtype='<f4').tobytes()).hexdigest()
    np.savez_compressed(out_path,times=times,acoustic=acoustic,embedding=embedding,
        mean=ann.mean[idx].astype('float32'), std=ann.std[idx].astype('float32'),
        mask=ann.mask[idx], std_mask=ann.std_mask[idx],raters=ann.raters[idx].astype('float32'),
        step_seconds=np.float32(1./fps),audio_sha256=np.array(audio_sha256))
    return len(times)
