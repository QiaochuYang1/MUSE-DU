from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import pytest
from dataclasses import replace
from musedu.datasets import load_averaged_csv,find_audio
from musedu.features import prepare_one,FEATURE_NAMES
from musedu.experiment import run_training,run_external


def test_end_to_end_synthetic_audio_not_real_benchmark(tmp_path):
    torch.set_num_threads(2)
    ann=tmp_path/'ann';ann.mkdir()
    audio=tmp_path/'audio';audio.mkdir()
    ids=[str(x) for x in range(100,115)]
    n=12;sr=24000
    for ax in ('valence','arousal'):
        df=pd.DataFrame({'song_id':ids,**{f'sample_{i+1}':[.15+.02*np.sin(i+j)]*1 for j in [] for i in []}})
        for i in range(n):df[f'sample_{i+1}']=[float(.25*np.sin(.2*i+k*.1)) for k in range(len(ids))]
        df.to_csv(ann/f'{ax}_cont_average.csv',index=False)
        sd=df.copy()
        for i in range(n):sd[f'sample_{i+1}']=.2
        sd.to_csv(ann/f'{ax}_cont_std.csv',index=False)
    anns=load_averaged_csv(ann,'deam',start=0.)
    rows=[]
    for j,a in enumerate(anns):
        tt=np.arange(int(sr*7))/sr
        sf.write(audio/f'{a.song_id}.wav',(.04*np.sin(tt*2*np.pi*(220+j))).astype('float32'),sr)
        dest=tmp_path/f'{a.song_id}.npz'
        prepare_one(a,find_audio(audio,a.song_id),dest,backend='acoustic')
        rows.append({'dataset':'deam','song_id':a.song_id,'file':str(dest)})
    manifest=tmp_path/'manifest.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    out=tmp_path/'exp'
    r=run_training(manifest,out,device='cpu',ensemble=2,epochs=1,
                   scale_epochs=2,window=6,hidden=16)
    assert len(FEATURE_NAMES)==18
    assert (out/'calibration.json').exists() and (out/'test_metrics.json').exists()
    assert r['interval_metrics']['songs']>=1
    # A development manifest must never be accepted as external evaluation.
    with pytest.raises(ValueError, match='development corpus'):
        run_external(manifest,out,device='cpu',window=6)

    # A different dataset using the SAME audio (even with a changed song ID)
    # must be rejected by decoded-audio fingerprint comparison.
    copy_manifest=tmp_path/'bad_pmemo.csv'
    pd.DataFrame([{'dataset':'pmemo','song_id':'unrelated_id','file':rows[0]['file']}]).to_csv(copy_manifest,index=False)
    with pytest.raises(ValueError,match='Audio overlap'):
        run_external(copy_manifest,out,device='cpu',window=6)

    # Generate an actual independent synthetic audio file for the positive path.
    different=replace(anns[0],dataset='pmemo',song_id='independent_test')
    tt=np.arange(int(sr*7))/sr
    sf.write(audio/'independent_test.wav',(.04*np.sin(tt*2*np.pi*1303)).astype('float32'),sr)
    different_npz=tmp_path/'independent_test.npz'
    prepare_one(different,audio/'independent_test.wav',different_npz,backend='acoustic')
    good_manifest=tmp_path/'good_pmemo.csv'
    pd.DataFrame([{'dataset':'pmemo','song_id':'independent_test','file':str(different_npz)}]).to_csv(good_manifest,index=False)
    ext=run_external(good_manifest,out,device='cpu',window=6)
    assert ext['interval_metrics']['songs']==1
    assert ext['audio_hash_overlap']==0
