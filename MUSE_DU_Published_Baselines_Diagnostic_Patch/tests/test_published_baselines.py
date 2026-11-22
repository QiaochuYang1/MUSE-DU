import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from musedu.published_baselines.io import ccc,pearson,per_song_metrics,normalized
from musedu.published_baselines.models import LSTM,MLP,Conformer,MultiLayerProbe
from musedu.published_baselines.run import loss_equal_song,val_selector,batchify


def test_metrics():
    assert ccc(np.array([1.,2.,3.]),np.array([1.,2.,3.]))==pytest.approx(1.)
    assert pearson(np.array([1.,2.,3.]),np.array([3.,2.,1.]))==pytest.approx(-1.)
    a=np.arange(8,dtype=np.float32).reshape(4,2)*.1
    d=per_song_metrics(a,a,np.ones_like(a,bool))
    assert d['med']==pytest.approx(0.)
    assert d['valence_mae']==0.
    assert d['arousal_ccc']==pytest.approx(1.)

@pytest.mark.parametrize('architecture', ['lstm','mlp','conformer','probe'])
def test_neural_forward_backward_padding(architecture):
    torch.manual_seed(2026)
    b,t,d=2,7,8
    if architecture=='lstm':net=LSTM(d)
    elif architecture=='mlp':net=MLP(d)
    elif architecture=='conformer':net=Conformer(d,hidden=32,blocks=1)
    else:net=MultiLayerProbe(d,4)
    x=torch.randn(b,t,4,d) if architecture=='probe' else torch.randn(b,t,d)
    valid=torch.tensor([[1]*7,[1]*5+[0]*2],dtype=torch.bool)
    y=torch.tanh(torch.randn(b,t,2))
    mask=valid[:,:,None].expand(-1,-1,2)
    pred=net(x,valid)
    assert pred.shape==(b,t,2)
    assert torch.all(pred[1,5:]==0)
    loss=loss_equal_song(pred,y,mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())


def test_variable_length_collate_selector():
    a={'sid':'x','x':np.ones((4,6),np.float32),'y':np.ones((4,2),np.float32)*.1,'mask':np.ones((4,2),bool)}
    b={'sid':'y','x':np.ones((2,6),np.float32),'y':np.ones((2,2),np.float32)*-.2,'mask':np.ones((2,2),bool)}
    batch=batchify([a,b]);assert batch['valid'].shape==(2,4)
    assert not batch['mask'][1,2:].any()
    assert torch.isfinite(val_selector(torch.zeros_like(batch['y']),batch['y'],batch['mask']))


def test_normalization():
    x=np.array([[100.,2.]],dtype=np.float32)
    assert normalized(x,np.zeros(2),np.ones(2))[0,0]==5.


def test_end_to_end_training_and_reporting(tmp_path,monkeypatch):
    """A tiny synthetic test of the repository NPZ interface; never DEAM scores."""
    import sys,types
    from musedu.published_baselines.run import run_neural,run_classical
    from musedu.published_baselines.io import evaluate,save_report,result_root
    stub=types.ModuleType('musedu.experiment')
    def read_npz(path):
        with np.load(path,allow_pickle=False) as x:return {k:x[k] for k in x.files}
    stub.read_npz=read_npz
    monkeypatch.setitem(sys.modules,'musedu.experiment',stub)
    features=tmp_path/'features';index={};train=[];val=[]
    rng=np.random.default_rng(42)
    for i in range(9):
        sid=f'song_{i}'
        if i<6:train.append(sid)
        else:val.append(sid)
        n=8+i%3
        times=np.arange(n)*.5+15.
        raw=rng.standard_normal((n,77)).astype(np.float32)
        targets=np.clip(raw[:,:2]*.1,-1,1).astype(np.float32)
        path=tmp_path/(sid+'.npz');index[sid]=str(path)
        np.savez(path,times=times,mean=targets,mask=np.ones((n,2),bool),
                 acoustic=np.ones((n,18),np.float32),embedding=np.ones((n,8),np.float32))
        d=features/'librosa';d.mkdir(exist_ok=True,parents=True)
        np.savez(d/(sid+'.npz'),times=times,features=raw)
    pred,h,_=run_neural(index,train,val,val,'librosa','mlp',features,
                        seed=2026,epochs=2,device='cpu',batch_size=3)
    assert 1<=h['best_epoch']<=2
    metrics=evaluate(pred,index,val)[1]
    assert np.isfinite(metrics['valence_mae'])
    folder=result_root(tmp_path/'results')
    save_report(folder,pred,index,val,{'method':'mlp','seed':2026})
    assert (folder/'validation_metrics.json').exists()
    assert len(list((folder/'predictions').glob('*.npz')))==3
    rf,_,_=run_classical(index,train,val,'librosa','rf',features,seed=2026)
    assert all(np.isfinite(x).all() for x in rf.values())


def test_hybrid_oof_and_16_component_assembly(tmp_path,monkeypatch):
    """Stack orchestration smoke test with fake base predictors; not real scores."""
    import types
    import musedu.published_baselines.run as bm
    import musedu.published_baselines.io as bi
    stub=types.ModuleType('musedu.experiment')
    def read_npz(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    stub.read_npz=read_npz
    monkeypatch.setitem(sys.modules,'musedu.experiment',stub)
    index={};train=[];val=[];root=tmp_path/'features'
    rng=np.random.default_rng(22)
    for i in range(11):
        sid='s'+str(i);(train if i<8 else val).append(sid)
        t=np.arange(7)*.5+15
        y=(rng.normal(size=(7,2))*.1).astype(np.float32)
        p=tmp_path/(sid+'.npz');index[sid]=str(p)
        np.savez(p,times=t,mean=y,mask=np.ones((7,2),bool),
                 acoustic=np.ones((7,18),np.float32),embedding=np.ones((7,8),np.float32))
        for view,d in [('opensmile',988),('librosa',77),('paa',20)]:
            (root/view).mkdir(parents=True,exist_ok=True)
            np.savez(root/view/(sid+'.npz'),times=t,features=rng.normal(size=(7,d)).astype(np.float32))
    def fake_member(index,fit_ids,stop_ids,predict_ids,view,kind,features_dir,seed,epochs,device,batch_size):
        assert not (set(fit_ids)&set(predict_ids))
        # OOF folds have separate stopping data; final base refits may use
        # development-validation songs for checkpoint selection (exploratory).
        assert not (set(stop_ids)&set(predict_ids)) or set(stop_ids)==set(val)
        data={}
        for sid in predict_ids:
            t=bi.load_song(index,sid)['times'];base=(hash((view,kind,sid))%50)/1000
            data[sid]=np.stack([.1*np.sin(t)+base,.1*np.cos(t)-base],1).astype(np.float32)
        return data,{},None
    monkeypatch.setattr(bm,'run_member',fake_member)
    p,detail,_=bm.stack(index,train,val,root,2026,1,'cpu',2,folds=2)
    assert detail['components']==16
    assert len(detail['meta_coefficients'])==2
    assert all(z.shape==(7,2) for z in p.values())


def test_soundfile_extract_first_song_resume(tmp_path, monkeypatch):
    """Synthetic audio: exercise SoundFile decoding and saved Librosa 77-D cache."""
    import soundfile as sf
    from musedu.published_baselines import features as f
    sr=44100
    t=np.arange(int(2.2*sr))/sr
    wave=(0.08*np.sin(2*np.pi*220*t)).astype(np.float32)
    audio=tmp_path/'song_0.wav'
    sf.write(audio,wave,sr)
    idx={'song_0':str(tmp_path/'song_0.npz')}
    splits={'train':['song_0'],'val':[]}
    labels=np.array([.5,1.,1.5,2.],np.float32)
    monkeypatch.setattr(f,'load_index',lambda *args:(idx,splits))
    monkeypatch.setattr(f,'load_song',lambda *args:{'times':labels})
    monkeypatch.setattr(f,'find_audio',lambda *args:audio)
    root=tmp_path/'cache'
    f.prepare('fake.csv','fake-ref',tmp_path,root,['librosa'],decoder='soundfile',limit_songs=1)
    file=root/'librosa'/'song_0.npz'
    assert file.exists()
    with np.load(file) as z:
        assert z['features'].shape==(4,77)
        assert np.isfinite(z['features']).all()
        assert np.allclose(z['times'],labels)
    # A second call should re-use the cache without decoding/recomputing.
    monkeypatch.setattr(f,'decode_audio',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('cache miss')))
    f.prepare('fake.csv','fake-ref',tmp_path,root,['librosa'],decoder='soundfile',limit_songs=1)
