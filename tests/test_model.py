import numpy as np
import torch
from musedu.model import MUSEDU,ErrorScale,supervised_loss,laplace_scale_loss
from musedu.metrics import song_metrics
from musedu.calibration import song_score,conformal_quantile,interval,evaluate_intervals
from musedu.experiment import split_songs,pack
from musedu.explain import grouped_occlusion
import pandas as pd

torch.set_num_threads(2)


def test_model_loss_backward_padding_and_scale():
    batch=pack([{'x':np.ones((7,18),'float32'), 'times':np.arange(7,dtype='float32')*.5,
             'mean':np.ones((7,2),'float32')*.2,'std':np.ones((7,2),'float32')*.1,
             'mask':np.ones((7,2),bool),'std_mask':np.ones((7,2),bool)},
             {'x':np.zeros((4,18),'float32'),'times':np.arange(4,dtype='float32')*.5,
             'mean':np.zeros((4,2),'float32'),'std':np.zeros((4,2),'float32'),
             'mask':np.ones((4,2),bool),'std_mask':np.zeros((4,2),bool)}])
    m=MUSEDU(18,hidden=16,layers=1)
    o=m(batch['x'],batch['times'],batch['valid'])
    assert o['mean'].shape==(2,7,2) and o['tau'].min()>0
    loss,d=supervised_loss(o,batch)
    assert torch.isfinite(loss);loss.backward()
    net=ErrorScale(16);out=net(o['hidden'].detach(),o['tau'].detach(),torch.zeros_like(o['tau']),o['mean'].detach(),batch['times'])
    assert out.min()>0
    l=laplace_scale_loss(out,o['mean'],batch['mean'],batch['mask'])
    assert torch.isfinite(l);l.backward()


def test_conformal_math_no_fake_coverage():
    pred=np.array([[0.,.2],[.3,.2]])
    y=pred+.1; scale=np.ones_like(pred)*.2;mask=np.ones_like(pred,bool)
    assert np.isclose(song_score(pred,y,scale,mask),.5)
    assert np.isclose(conformal_quantile(np.arange(1,20),alpha=.1),18)
    assert np.isinf(conformal_quantile([.3],alpha=.1))
    low,high=interval(pred,scale,float('inf'))
    assert np.all(low==-1) and np.all(high==1)
    res=evaluate_intervals([{'pred':pred,'ref':y,'scale':scale,'mask':mask}],.5)
    assert res['song_coverage']==1


def test_splits_disjoint():
    df=pd.DataFrame({'song_id':[str(i) for i in range(100)]})
    s=split_songs(df)
    assert sum(len(v) for v in s.values())==100
    assert len(set().union(*[set(v) for v in s.values()]))==100
    assert split_songs(df)==s


def test_metrics_and_grouped_explain():
    x=np.zeros((3,2)); y=np.array([[.1,.2],[.2,.3],[.3,.4]])
    rep=song_metrics(x,y,np.ones_like(y,bool))
    assert rep['valence_mae']>0
    base=np.ones((3,3))
    r=grouped_occlusion(lambda z: np.tile(z[:,0,None],(1,2)),base,slice(0,2),{'first':[0], 'last':[2]})
    assert r['first']==1 and r['last']==0
