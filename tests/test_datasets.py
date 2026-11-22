from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from musedu.datasets import read_wide_table,load_averaged_csv,load_merp,find_audio


def build_csv(root):
    root.mkdir(parents=True,exist_ok=True)
    for ax in ('valence','arousal'):
        for typ in ('average','std'):
            df=pd.DataFrame({'song_id':[11,12], 'sample_1':[.2,.1], 'sample_2':[.3,.4],
                             'sample_3':[.1,.2]})
            if typ=='std':df.loc[:,['sample_1','sample_2','sample_3']]=.25
            df.to_csv(root/f'{ax}_cont_{typ}.csv',index=False)


def test_deam_and_pmemo_dynamic_csv(tmp_path):
    build_csv(tmp_path)
    for ds,scale in [('deam','minus_one_one'),('pmemo','zero_one')]:
        a=load_averaged_csv(tmp_path,ds,scale=scale)
        assert len(a)==2
        assert np.allclose(a[0].times,[15,15.5,16])
        assert a[0].mask.all() and a[0].std_mask.all()
        assert a[0].mean.shape==(3,2)
        assert np.isclose(a[0].mean[0,0], .2 if ds=='deam' else -.6)
        assert np.isclose(a[0].std[0,0], .25 if ds=='deam' else .5)


def test_refuses_invalid_scale_and_ambiguous_schema(tmp_path):
    df=pd.DataFrame({'song_id':[1],'1':[.1],'2':[.2]})
    p=tmp_path/'arousal_cont_average.csv';df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='numeric-only'):
        read_wide_table(p,fps=2.,start=15,scale='minus_one_one')
    build_csv(tmp_path/'ok')
    for ax in ('valence','arousal'):
        p=tmp_path/'ok'/f'{ax}_cont_average.csv'
        df=pd.read_csv(p);df.iloc[0,1]=1.9;df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='outside'):
        load_averaged_csv(tmp_path/'ok','deam')


def test_merp_rater_arrays_and_std(tmp_path,monkeypatch):
    df=pd.DataFrame([
        {'song_id':'00_01','rater_id':f'rater_{i:02}', 'valence':[.2+i*.1]*20,
         'arousal':[.4+i*.1]*20} for i in range(3)])
    (tmp_path/'annotations_filtered.parquet').touch()
    monkeypatch.setattr(pd,'read_parquet',lambda _:df)
    z=load_merp(tmp_path,fps=2.)[0]
    assert len(z.times)==20
    assert np.allclose(z.times[:3],[15.,15.5,16.])
    assert np.allclose(z.mean[:,0],.3)
    assert np.allclose(z.std[:,0],.1)
    assert np.all(z.raters==3)
    assert z.std_mask.all()
    (tmp_path/'audio').mkdir()
    (tmp_path/'audio'/'00_01.wav').touch()
    assert find_audio(tmp_path/'audio','00_01').name=='00_01.wav'


def test_merp_raw_10hz_starts_at_zero(tmp_path,monkeypatch):
    (tmp_path/'annotations_raw.parquet').touch()
    df=pd.DataFrame([{'song_id':'00_01','rater_id':'rater_01',
                      'valence':[.2]*20,'arousal':[.4]*20}])
    monkeypatch.setattr(pd,'read_parquet',lambda _:df)
    z=load_merp(tmp_path,filtered=False,fps=2.)[0]
    assert np.allclose(z.times,[0.,.5,1.,1.5])


def test_merp_rater_duplicates_rejected(tmp_path,monkeypatch):
    (tmp_path/'annotations_filtered.parquet').touch()
    df=pd.DataFrame([{'song_id':'deam_115','rater_id':'x','valence':[0]*10,'arousal':[0]*10}]*2)
    monkeypatch.setattr(pd,'read_parquet',lambda _:df)
    with pytest.raises(ValueError,match='repeated'):
        load_merp(tmp_path)


def test_pmemo_2019_long_annotations(tmp_path):
    pd.DataFrame({'musicId':[7,7],'frameTime':[15.5,16.],
                  'Valence(mean)':[.25,.75],'Arousal(mean)':[.5,.6]}).to_csv(
                      tmp_path/'dynamic_annotations.csv',index=False)
    pd.DataFrame({'musicId':[7,7],'frameTime':[15.5,16.],
                  'Valence(std)':[.1,.2],'Arousal(std)':[.3,.4]}).to_csv(
                      tmp_path/'dynamic_annotations_std.csv',index=False)
    with pytest.raises(ValueError,match='zero_one'):
        load_averaged_csv(tmp_path,'pmemo')
    item=load_averaged_csv(tmp_path,'pmemo',scale='zero_one')[0]
    assert item.song_id=='7'
    assert np.allclose(item.times,[15.5,16.])
    assert np.allclose(item.mean,[[-.5,0.],[.5,.2]])
    assert np.allclose(item.std,[[.2,.6],[.4,.8]])


def test_deam_release_absolute_times_and_rater_dispersion(tmp_path):
    base=tmp_path/'DEAM_Annotations'/'annotations'
    avg=base/'annotations averaged per song'/'dynamic (per second annotations)'
    individual=base/'annotations per each rater'/'dynamic (per second annotations)'
    avg.mkdir(parents=True)
    for axis in ('valence','arousal'):
        pd.DataFrame({'song_id':[10],'sample_15000ms':[.2],
                      'sample_15500ms':[.4]}).to_csv(avg/f'{axis}.csv',index=False)
        folder=individual/axis;folder.mkdir(parents=True)
        pd.DataFrame({'sample_15000ms':[.1,.3],
                      'sample_15500ms':[.2,.6]}).to_csv(folder/'10.csv',index=False)
    item=load_averaged_csv(tmp_path,'deam')[0]
    assert item.song_id=='10'
    assert np.allclose(item.times,[15.,15.5])
    assert np.allclose(item.mean,[[.2,.2],[.4,.4]])
    assert np.allclose(item.std[0],[np.sqrt(.02)]*2)
    assert np.all(item.raters==2)
