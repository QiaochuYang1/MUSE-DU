import numpy as np
from musedu.b3_level_fusion import fuse,fit_calibrator
from musedu.published_baselines.io import pearson


def test_positive_scale_preserves_b2_within_song_pearson():
    b2=np.array([[.1,-.2],[.2,.4],[-.3,.1],[.5,.3]],np.float32)
    mert=np.array([[.4,.1],[.5,.0],[.5,.2],[.3,.1]],np.float32)
    combined=fuse(b2,mert,[.7,.2,1.5,.5])
    for axis in range(2):
        assert np.isclose(pearson(b2[:,axis],np.arange(4,dtype=float)),
                          pearson(combined[:,axis],np.arange(4,dtype=float)),atol=1e-6)
    assert np.isclose(combined[:,0].mean(),.3*b2[:,0].mean()+.7*mert[:,0].mean())


def test_calibration_parameters_are_numerically_learned(monkeypatch):
    rng=np.random.default_rng(12)
    ids=[str(i) for i in range(24)]
    b2={sid:rng.normal(0,.2,(12,2)).astype(np.float32) for sid in ids}
    mert={sid:rng.normal(0,.2,(12,2)).astype(np.float32) for sid in ids}
    true=np.array([.8,.3,1.3,.7])
    targets={sid:fuse(b2[sid],mert[sid],true) for sid in ids}
    monkeypatch.setattr('musedu.b3_level_fusion.load_song',lambda index,sid:
                        {'mean':targets[sid],'mask':np.ones((12,2),bool)})
    fitted,detail=fit_calibrator(ids,b2,mert,{})
    assert detail['selected_med']<detail['starting_med']*.1
    assert np.allclose(fitted,true,atol=.03)
    level_only,_=fit_calibrator(ids,b2,mert,{},scale_fixed=1.)
    assert np.allclose(level_only[2:],[1.,1.])
