"""Frozen last-layer MERT linear Ridge probe for song-disjoint DEAM CV."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from pathlib import Path


ALPHAS = (1.0, 100.0, 10000.0)


def _last(index, sid, features_dir):
    with np.load(Path(features_dir) / 'mert' / f'{sid}.npz', allow_pickle=False) as archive:
        features = archive['features']
        feature_times = archive['times']
    with np.load(index[sid], allow_pickle=False) as archive:
        song = {name: archive[name] for name in ('times', 'mean', 'mask')}
    if not np.allclose(feature_times, song['times'], atol=1e-3):
        raise ValueError(f'MERT timestamps do not match song {sid}')
    if features.ndim != 3:
        raise ValueError('MERT feature cache must contain [time, layer, dimension]')
    return features[:, -1, :].copy(), song


def _score(predictions, songs):
    values = []
    correlations = []
    for sid, song in songs.items():
        pred = predictions[sid]
        target = np.nan_to_num(song['mean'], nan=0).astype(np.float32)
        mask = song['mask'].astype(bool)
        error = np.abs(pred - target)
        huber = np.where(error < .1, .5 * error ** 2, .1 * (error - .05))
        if mask.any():
            values.append(float(huber[mask].mean()))
        for axis in range(2):
            use = mask[:, axis]
            if use.sum() < 3:
                continue
            y = target[use, axis] - target[use, axis].mean()
            if np.dot(y, y) <= 1e-6:
                continue
            x = pred[use, axis] - pred[use, axis].mean()
            corr = np.dot(x, y) / (max(np.linalg.norm(x), 1e-6) * max(np.linalg.norm(y), 1e-6))
            correlations.append(1 - float(corr))
    return float(np.mean(values) + .02 * np.mean(correlations))


def run_mert_linear(index, fit_ids, stop_ids, predict_ids, features_dir):
    """Choose one shared alpha on internal songs, then predict outer songs.

    The probe sees only the frozen final MERT layer. Each song contributes unit
    total weight per target axis, regardless of its number of annotated bins.
    """
    train = {sid: _last(index, sid, features_dir) for sid in fit_ids}
    valid = {sid: _last(index, sid, features_dir) for sid in stop_ids}
    n = sum(len(x) for x, _ in train.values())
    total = sum((x.sum(0, dtype=np.float64) for x, _ in train.values()))
    total2 = sum(((x.astype(np.float64) ** 2).sum(0) for x, _ in train.values()))
    mean = total / n
    std = np.maximum(np.sqrt(np.maximum(total2 / n - mean ** 2, 0)), 1e-5)
    def normalize(x):
        return np.clip((x - mean) / std, -5, 5).astype(np.float32)

    xfit = np.concatenate([normalize(x) for x, _ in train.values()])
    yfit = np.concatenate([np.nan_to_num(z['mean'], nan=0) for _, z in train.values()]).astype(np.float32)
    mask = np.concatenate([z['mask'] for _, z in train.values()]).astype(bool)
    xvalid = {sid: normalize(x) for sid, (x, _) in valid.items()}
    scores = {}
    best = float('inf')
    selected_models = None
    chosen_alpha = None
    for alpha in ALPHAS:
        models = []
        for axis in range(2):
            sample_weights = np.concatenate([
                np.full(len(x), 1.0 / max(1, z['mask'][:, axis].sum()), np.float32)
                for x, z in train.values()])
            use = mask[:, axis]
            model = Ridge(alpha=alpha, solver='lsqr', tol=1e-4)
            model.fit(xfit[use], yfit[use, axis], sample_weight=sample_weights[use] * len(fit_ids))
            models.append(model)
        prediction = {sid: np.clip(np.column_stack([m.predict(x) for m in models]), -1, 1).astype(np.float32)
                      for sid, x in xvalid.items()}
        score = _score(prediction, {sid: z for sid, (_, z) in valid.items()})
        scores[str(alpha)] = score
        if score < best:
            best = score
            chosen_alpha = alpha
            selected_models = models
    outer = {}
    for sid in predict_ids:
        x, _ = _last(index, sid, features_dir)
        z = normalize(x)
        outer[sid] = np.clip(np.column_stack([m.predict(z) for m in selected_models]), -1, 1).astype(np.float32)
    detail = {'feature': 'frozen MERT final layer', 'model': 'Ridge linear probe',
              'alpha_candidates': list(ALPHAS), 'selected_alpha': chosen_alpha,
              'internal_validation_scores': scores, 'best_selector': best,
              'song_equal_training_weights': True, 'outer_labels_used_for_selection': False}
    return outer, detail
