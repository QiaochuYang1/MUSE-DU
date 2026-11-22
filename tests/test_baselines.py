"""Synthetic-only controlled comparisons; real licensed corpora remain to be audited."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from musedu.baselines import METHODS, NEURAL_METHODS, BaselineNetwork, choose_view
from musedu.baseline_experiment import run_baseline_training, run_baseline_external
from musedu.experiment import load_manifest, split_songs, provenance


@pytest.fixture
def tiny_corpus(tmp_path):
    rng=np.random.default_rng(123)
    root=tmp_path/'features';root.mkdir()
    rows=[]
    for i in range(15):
        t=np.arange(10,dtype='float32')*.5
        acoustic=rng.normal(size=(len(t),18)).astype('float32')
        embedding=rng.normal(size=(len(t),8)).astype('float32')
        y=np.column_stack((.2*np.sin(t+i*.1),.35*np.cos(t+i*.1))).astype('float32')
        mask=np.ones_like(y,dtype=bool)
        if i%3==0:mask[1,1]=False
        dest=root/f'song{i}.npz'
        np.savez(dest,acoustic=acoustic,embedding=embedding,times=t,mean=y,
                 std=np.full_like(y,.2),mask=mask,std_mask=mask,
                 step_seconds=np.float32(.5),audio_sha256=np.array(f'{i+1:064x}'))
        rows.append({'dataset':'deam','song_id':f'song{i}','file':str(dest)})
    manifest=tmp_path/'deam.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    return manifest


def test_all_twelve_baseline_architectures_and_gradients():
    assert len(METHODS)==12
    torch.set_num_threads(2)
    x=torch.randn(2,7,26)
    times=torch.arange(7).float().repeat(2,1)
    valid=torch.ones(2,7,dtype=torch.bool)
    valid[1,-2:]=False
    for method in NEURAL_METHODS:
        net=BaselineNetwork(method,26,hidden=16)
        out=net(x,times,valid)
        assert out.shape==(2,7,2),method
        assert torch.isfinite(out).all(),method
        assert (out.abs()<=1+1e-6).all(),method
        assert (out[1,-2:]==0).all(),method
        out.sum().backward()
        if method=='cnn_bigru_hierarchical':
            assert net.global_head.weight.grad is not None,method
            assert net.local_head.weight.grad is not None,method
        else:assert net.head.weight.grad is not None,method


def test_mert_methods_require_real_embedding_features():
    z={'acoustic':np.ones((8,18),np.float32),
       'embedding':np.empty((8,0),np.float32)}
    assert choose_view(z,'acoustic').shape==(8,18)
    with pytest.raises(ValueError,match='requires --backend mert'):
        choose_view(z,'mert')
    with pytest.raises(ValueError,match='requires --backend mert'):
        choose_view(z,'fusion')


def test_ridge_and_neural_training_same_splits_and_separate_external(tiny_corpus,tmp_path):
    torch.set_num_threads(2)
    train=tmp_path/'ridge'
    result=run_baseline_training(tiny_corpus,train,'ridge',seed=17)
    assert result['method']=='ridge'
    assert result['uniform_conformal']['songs']>=1
    assert (train/'test_song_metrics.csv').exists()
    # Reference matches the exact source manifest and split assignments.
    match=tmp_path/'mert_tcn'
    r=run_baseline_training(tiny_corpus,match,'mert_tcn',reference_run=train,
                             hidden=16,epochs=1,window=5,batch_size=2,seed=999)
    assert r['method']=='mert_tcn'
    assert json.loads((train/'splits.json').read_text())==json.loads((match/'splits.json').read_text())
    with pytest.raises(FileExistsError):
        run_baseline_training(tiny_corpus,match,'ridge')
    with pytest.raises(ValueError,match='development corpus'):
        run_baseline_external(tiny_corpus,train)
    # Re-label an identical cached song as an external domain: must be rejected.
    df=pd.read_csv(tiny_corpus)
    df.loc[0,'dataset']='pmemo'
    bad=tmp_path/'external_bad.csv';df.iloc[[0]].to_csv(bad,index=False)
    with pytest.raises(ValueError,match='Audio overlap'):
        run_baseline_external(bad,train)
    # Truly distinct audio ID/hash permits external evaluation with a frozen model.
    with np.load(df.loc[0,'file']) as z:source={k:z[k] for k in z.files}
    source['audio_sha256']=np.array('f'*64)
    newfile=tmp_path/'external_new.npz';np.savez(newfile,**source)
    df.loc[0,'file']=str(newfile)
    good=tmp_path/'external_good.csv';df.iloc[[0]].to_csv(good,index=False)
    ev=run_baseline_external(good,train)
    assert ev['external_dataset']=='pmemo'
    assert ev['uniform_conformal']['songs']==1


def test_xgboost_and_random_forest_classical_implementation(tiny_corpus,tmp_path):
    pytest.importorskip('xgboost')
    for method in ('xgboost','random_forest'):
        result=run_baseline_training(tiny_corpus,tmp_path/method,method)
        assert np.isfinite(result['point_metrics']['valence_mae'])
        assert result['point_metrics']['arousal_mae']>=0


def test_reference_splits_disallow_source_mutations(tiny_corpus,tmp_path):
    train=tmp_path/'ridge';run_baseline_training(tiny_corpus,train,'ridge')
    df=pd.read_csv(tiny_corpus)
    with np.load(df.loc[0,'file']) as z:source={k:z[k] for k in z.files}
    source['audio_sha256']=np.array('b'*64)
    new=tmp_path/'replaced.npz';np.savez(new,**source)
    df.loc[0,'file']=str(new)
    modified=tmp_path/'modified.csv';df.to_csv(modified,index=False)
    with pytest.raises(ValueError,match='different source audio'):
        run_baseline_training(modified,tmp_path/'modified_run','ridge',reference_run=train)


def test_suite_writes_comparison_csv_and_a_true_linear_probe(tiny_corpus,tmp_path):
    from musedu.benchmark_suite import run_suite
    x=torch.randn(2,3,26)
    m=BaselineNetwork('mert_linear',26,hidden=16)
    assert sum(p.numel() for p in m.parameters())==26*2+2
    result=run_suite(tiny_corpus,tmp_path/'suite',methods=['ridge','mert_linear'],
                     hidden=16,epochs=1,window=5,batch_size=2)
    assert result['n_methods']==2
    df=pd.read_csv(tmp_path/'suite'/'baseline_comparison.csv')
    assert set(df.method)=={'ridge','mert_linear'}
    assert set(df.evaluation)=={'in_domain_test'}
