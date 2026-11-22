"""Split-conformal simultaneous song-level mean-trajectory coverage."""
from __future__ import annotations
import math
import numpy as np


def song_score(pred,reference,scale,mask):
    pred=np.asarray(pred);reference=np.asarray(reference);scale=np.asarray(scale);mask=np.asarray(mask,bool)
    if pred.shape!=reference.shape or reference.shape!=scale.shape or scale.shape!=mask.shape:
        raise ValueError('Mismatched conformal arrays')
    valid=mask & np.isfinite(reference)&np.isfinite(pred)&np.isfinite(scale)&(scale>0)
    if not valid.any():raise ValueError('No valid label for song')
    if (mask&~valid).any():raise ValueError('Some masked conformal entries invalid')
    return float(np.max(np.abs(reference[valid]-pred[valid])/scale[valid]))


def conformal_quantile(scores,alpha=.1):
    scores=np.sort(np.asarray(scores,dtype=float))
    if scores.ndim!=1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError('Calibration scores must be finite 1-D nonempty')
    if not 0<alpha<1:raise ValueError('alpha must be in (0,1)')
    k=math.ceil((len(scores)+1)*(1-alpha))
    return float(scores[k-1]) if k<=len(scores) else float('inf')


def interval(pred,scale,q):
    pred=np.asarray(pred);scale=np.asarray(scale)
    if q==float('inf'):
        return np.full_like(pred,-1.,dtype=float),np.full_like(pred,1.,dtype=float)
    return np.maximum(-1,pred-q*scale),np.minimum(1,pred+q*scale)


def evaluate_intervals(items,q):
    song_cover=[];width=[];frame_covered=[]
    for z in items:
        lo,hi=interval(z['pred'],z['scale'],q)
        mask=z['mask']
        if not mask.any():continue
        inside=(z['ref']>=lo)&(z['ref']<=hi)
        song_cover.append(float(inside[mask].all()))
        width.append(float((hi-lo)[mask].mean()))
        frame_covered.extend(inside[mask].tolist())
    if not song_cover:raise ValueError('No valid test songs')
    return {'song_coverage':float(np.mean(song_cover)), 'mean_width':float(np.mean(width)),
            'frame_coverage':float(np.mean(frame_covered)), 'songs':len(song_cover)}


def calibration_curve(calibration_items,evaluation_items,nominals=(.7,.8,.85,.9,.95)):
    """Song-level simultaneous curve for supplied uniform or learned scales."""
    if not calibration_items or not evaluation_items:raise ValueError('Need calibration and evaluation songs')
    scores=[song_score(z['pred'],z['ref'],z['scale'],z['mask']) for z in calibration_items]
    rows=[]
    for nominal in nominals:
        if not 0<nominal<1:raise ValueError('Nominal coverage must lie in (0,1)')
        q=conformal_quantile(scores,1-nominal)
        rows.append({'nominal':float(nominal),'q':q,'n_cal':len(scores),
                     **evaluate_intervals(evaluation_items,q)})
    return rows


def matched_coverage_width(curve_a,curve_b,coverage_levels):
    """Interpolate width at common empirical coverage; exploratory diagnostic."""
    def prepare(curve):
        points=sorted((float(r['song_coverage']),float(r['mean_width'])) for r in curve)
        x=[];y=[]
        for cover,width in points:
            if x and cover==x[-1]:y[-1]=min(y[-1],width)
            else:x.append(cover);y.append(width)
        return np.asarray(x),np.asarray(y)
    xa,ya=prepare(curve_a);xb,yb=prepare(curve_b)
    lo=max(xa.min(),xb.min());hi=min(xa.max(),xb.max())
    rows=[]
    for coverage in coverage_levels:
        if not lo<=coverage<=hi:continue
        rows.append({'empirical_coverage':float(coverage),
                     'width_a':float(np.interp(coverage,xa,ya)),
                     'width_b':float(np.interp(coverage,xb,yb))})
    return rows
