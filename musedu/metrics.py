import numpy as np


def song_metrics(pred,truth,mask):
    """Dimensionwise equal-song metrics; CCC undefined for fewer than two labels."""
    p=np.asarray(pred); y=np.asarray(truth);m=np.asarray(mask,bool)
    report={}
    for d,axis in enumerate(('valence','arousal')):
        x=y[m[:,d],d];z=p[m[:,d],d]
        if len(x)==0:
            report.update({f'{axis}_mae':float('nan'),f'{axis}_rmse':float('nan'),f'{axis}_ccc':float('nan')});continue
        mae=np.mean(np.abs(x-z));rmse=np.sqrt(np.mean((x-z)**2))
        vx=np.var(x);vz=np.var(z)
        cov=np.mean((x-x.mean())*(z-z.mean()))
        denom=vx+vz+(x.mean()-z.mean())**2
        ccc=2*cov/denom if len(x)>1 and denom>0 else float('nan')
        report.update({f'{axis}_mae':float(mae),f'{axis}_rmse':float(rmse),f'{axis}_ccc':float(ccc)})
    return report


def aggregate(reports):
    if not reports:raise ValueError('Empty evaluation set')
    return {k:float(np.nanmean([r[k] for r in reports])) for k in reports[0]}


def within_song_pearson(pred,truth,mask):
    """Per-axis within-recording correlation; undefined for flat trajectories."""
    p=np.asarray(pred);y=np.asarray(truth);m=np.asarray(mask,bool)
    result={}
    for axis,name in enumerate(('valence','arousal')):
        a=y[m[:,axis],axis];b=p[m[:,axis],axis]
        if len(a)<3 or np.ptp(a)<=1e-7:value=float('nan')
        elif np.ptp(b)<=1e-7:value=0. # A failed constant predictor must not disappear from the mean.
        else:value=float(np.corrcoef(a,b)[0,1])
        result[f'{name}_within_pearson']=value
    return result


def transition_metrics(pred,truth,mask,times,step_seconds,reference_thresholds,
                       prediction_thresholds):
    """Exact-bin changes and event counts with thresholds fixed outside evaluation."""
    p=np.asarray(pred);y=np.asarray(truth);m=np.asarray(mask,bool)
    t=np.asarray(times);adjacent=np.isclose(np.diff(t),step_seconds,rtol=.01,atol=1e-4)
    result={}
    for axis,name in enumerate(('valence','arousal')):
        keep=adjacent&m[1:,axis]&m[:-1,axis]
        dy=np.diff(y[:,axis])[keep];dp=np.diff(p[:,axis])[keep]
        actual=np.abs(dy)>reference_thresholds[axis]
        predicted=np.abs(dp)>prediction_thresholds[axis]
        result.update({f'{name}_first_difference_mae':float(np.mean(np.abs(dp-dy)))
                       if len(dy) else float('nan'),
                       f'{name}_event_tp':int(np.sum(actual&predicted)),
                       f'{name}_event_fp':int(np.sum(~actual&predicted)),
                       f'{name}_event_fn':int(np.sum(actual&~predicted))})
    return result


def event_f1_from_counts(tp,fp,fn):
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.
