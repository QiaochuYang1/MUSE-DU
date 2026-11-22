"""Post-hoc grouped acoustic-feature occlusion, with matched feature replacement.

Attributions describe model behavior, not causal effects on human emotion.
"""
from __future__ import annotations
import numpy as np


def grouped_occlusion(predict_fn,features,segment,groups,baseline=None,axis=1):
    """predict_fn(features) -> [T,2] predictions; segment is a slice of output times.

    Features are already standardized on train songs. Replacing with zero equals
    training-feature mean. Segment and feature group are perturbation coordinates.
    """
    x=np.asarray(features)
    if x.ndim!=2:raise ValueError('Features must be [T,F]')
    if baseline is None:baseline=np.zeros_like(x)
    base=np.asarray(predict_fn(x))[segment,axis].mean()
    result={}
    for name,cols in groups.items():
        alt=x.copy();alt[:,cols]=baseline[:,cols]
        result[name]=float(base-np.asarray(predict_fn(alt))[segment,axis].mean())
    return result


def musical_event_scores(acoustic,kind):
    """Aligned 2 Hz acoustic indicators; score changes, not emotion labels."""
    x=np.asarray(acoustic,dtype=float)
    if x.ndim!=2 or x.shape[1]!=18:raise ValueError('Expected [time,18] acoustic descriptors')
    if kind=='energy':raw=np.abs(np.diff(np.log1p(np.maximum(x[:,0],0)),prepend=0))
    elif kind=='onset':raw=np.maximum(x[:,5],0)
    elif kind=='spectral':raw=np.abs(np.diff(x[:,1],prepend=x[0,1]))
    elif kind=='tonal':raw=np.linalg.norm(np.diff(x[:,6:18],axis=0,
                                                 prepend=x[:1,6:18]),axis=1)
    else:raise ValueError('Event kind must be energy/onset/spectral/tonal')
    return raw/(np.nanmedian(np.abs(raw-np.nanmedian(raw)))+1e-6)


def event_regions(acoustic,kind,width=4,count=3):
    """Choose nonoverlapping high-event time regions without labels."""
    scores=musical_event_scores(acoustic,kind)
    if width<1 or len(scores)<width:raise ValueError('Invalid event width')
    candidates=[]
    for center in np.argsort(-scores):
        start=int(np.clip(center-width//2,0,len(scores)-width))
        stop=start+width
        if all(stop<=a or start>=b for a,b,_ in candidates):
            candidates.append((start,stop,float(scores[center])))
        if len(candidates)>=count:break
    return candidates


def matched_control_region(acoustic,event,kind,width=4):
    """Low-event, equal-duration region with similar mean RMS in same song."""
    x=np.asarray(acoustic);scores=musical_event_scores(x,kind)
    start,stop=event[:2]
    if stop-start!=width:raise ValueError('Event width mismatch')
    rms=x[:,0];target=float(rms[start:stop].mean())
    candidates=[]
    for a in range(0,len(x)-width+1):
        b=a+width
        if not (b<=start-width or a>=stop+width):continue
        candidates.append((a,b,float(scores[a:b].mean()),
                           abs(float(rms[a:b].mean())-target)))
    if not candidates:raise ValueError('No independent control region')
    low=np.median([v[2] for v in candidates])
    options=[v for v in candidates if v[2]<=low]
    scale=np.std(rms)+1e-6
    chosen=min(options,key=lambda v:(v[3]/scale+.25*v[2],v[0]))
    return chosen[:2]


def attenuate_audio_region(audio,sr,times,region,db=-6.,fade_seconds=.1):
    """Apply a smooth equal-duration gain intervention to an audio time region."""
    y=np.asarray(audio,dtype=np.float32)
    start,stop=region
    if not 0<=start<stop<=len(times):raise ValueError('Invalid region')
    step=float(np.median(np.diff(times))) if len(times)>1 else .5
    left=max(0,int(round((times[start]-step/2)*sr)))
    right=min(len(y),int(round((times[stop-1]+step/2)*sr)))
    if right<=left:raise ValueError('Empty audio intervention')
    gain=10**(db/20);envelope=np.ones(right-left,dtype=np.float32)*gain
    fade=min(int(round(fade_seconds*sr)),len(envelope)//2)
    if fade:
        envelope[:fade]=np.linspace(1,gain,fade)
        envelope[-fade:]=np.linspace(gain,1,fade)
    out=y.copy();out[left:right]*=envelope
    return out
