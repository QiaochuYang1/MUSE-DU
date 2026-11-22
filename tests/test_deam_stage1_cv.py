from musedu.deam_stage1_cv import make_protocol, internal_split


def test_outer_and_internal_splits_cover_once_without_overlap():
    ids=[str(i) for i in range(1802)]
    protocol=make_protocol(ids,seed=2026,epochs=25)
    folds=protocol['folds']
    assert [len(fold) for fold in folds]==[360,360,360,360,362]
    assert set().union(*(set(fold) for fold in folds))==set(ids)
    assert len(set().union(*(set(fold) for fold in folds)))==sum(map(len,folds))
    for i,outer in enumerate(folds):
        training=[sid for j,part in enumerate(folds) if j!=i for sid in part]
        fit,stop=internal_split(training,2027+i)
        assert not (set(fit)&set(stop))
        assert not (set(outer)&(set(fit)|set(stop)))
        assert set(fit)|set(stop)==set(training)
