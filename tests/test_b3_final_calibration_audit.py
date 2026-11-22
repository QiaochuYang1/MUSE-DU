import numpy as np

from musedu.b3_final_calibration_audit import ccc_from_sufficient, fit_single, pooled_sufficient
from musedu.published_baselines.io import ccc


def test_single_model_level_calibration_recovers_song_means_and_keeps_shape():
    rng = np.random.default_rng(17)
    calibration, predictions = {}, {}
    for i in range(144):
        sid = str(i)
        pred = rng.normal(size=(8, 2)) * .02 + rng.uniform(-.5, .5, size=(1, 2))
        truth = pred - pred.mean(axis=0) + (.1 + .7 * pred.mean(axis=0))
        predictions[sid] = pred
        calibration[sid] = (truth, np.ones_like(truth, dtype=bool))
    params, apply = fit_single(calibration, predictions)
    assert np.allclose(params['gamma'], [.7, .7], atol=.02)
    probe = predictions['0']
    calibrated = apply(probe)
    assert np.allclose(calibrated-calibrated.mean(axis=0), probe-probe.mean(axis=0))


def test_pooled_sufficient_stats_match_direct_ccc():
    rng = np.random.default_rng(8)
    pairs = []
    for length in (6, 12, 17):
        truth = rng.normal(size=(length, 2))
        pred = truth * .4 + rng.normal(size=(length, 2)) * .3
        mask = rng.random((length, 2)) > .2
        pairs.append((pred, truth, mask))
    sufficient = sum((pooled_sufficient(*pair) for pair in pairs))
    calculated = ccc_from_sufficient(sufficient)
    for axis in range(2):
        x = np.concatenate([truth[mask[:, axis], axis] for _, truth, mask in pairs])
        y = np.concatenate([pred[mask[:, axis], axis] for pred, _, mask in pairs])
        assert np.isclose(calculated[axis], ccc(x, y), atol=1e-12)
