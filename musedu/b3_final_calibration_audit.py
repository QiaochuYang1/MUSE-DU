"""Calibration-matched DEAM controls and fold-stratified paired interval audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .b3_level_fusion import fit_calibrator, fuse
from .published_baselines.io import evaluate, load_index, load_song, per_song_metrics, result_root, save_report
from .plan_matrix_2x2 import sha

MODELS = ('B2_raw', 'MERT_raw', 'B2_calibrated', 'MERT_calibrated', 'B3_level_only')
CONTRASTS = ('B2_raw', 'MERT_raw', 'B2_calibrated', 'MERT_calibrated')
METRICS = ('med', 'valence_ccc', 'arousal_ccc', 'valence_pooled_ccc', 'arousal_pooled_ccc')
GAMMA_BOUNDS = ((0., 2.), (0., 2.))
OPT_OPTIONS = {'maxiter': 150, 'ftol': 1e-10, 'eps': 1e-5}


def fit_single(calibration, axis_predictions):
    """Fit two axis gains jointly using the same equal-song MED objective as B3."""
    ids = sorted(calibration)
    true_means = []
    predicted_means = []
    records = []
    for sid in ids:
        truth, mask = calibration[sid]
        pred = np.asarray(axis_predictions[sid], dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if pred.shape != truth.shape or mask.shape != truth.shape:
            raise ValueError('Calibration shape mismatch: ' + sid)
        if not mask.all(axis=1).any():
            continue
        true_means.append([truth[mask[:, axis], axis].mean() for axis in range(2)])
        predicted_means.append(pred.mean(axis=0))
        records.append((pred, truth, mask.all(axis=1)))
    if len(records) != 144:
        raise AssertionError('Calibration song count changed')
    global_true = np.mean(true_means, axis=0)
    global_pred = np.mean(predicted_means, axis=0)

    def apply(pred, gamma):
        song_mean = pred.mean(axis=0, keepdims=True)
        target_mean = global_true + gamma * (song_mean - global_pred)
        return target_mean + (pred - song_mean)

    def objective(gamma):
        return float(np.mean([np.linalg.norm(apply(pred, gamma)[valid] - truth[valid], axis=1).mean()
                              for pred, truth, valid in records]))

    start = np.ones(2, dtype=np.float64)
    opt = minimize(objective, start, method='L-BFGS-B', bounds=GAMMA_BOUNDS,
                   options=OPT_OPTIONS)
    if not np.isfinite(opt.fun):
        raise RuntimeError('Single-model calibration failed: ' + str(opt.message))
    params = {'gamma': opt.x.tolist(), 'global_true': global_true.tolist(),
              'global_prediction': global_pred.tolist(), 'calibration_songs': len(records),
              'starting_med': objective(start), 'selected_med': objective(opt.x),
              'optimizer_success': bool(opt.success), 'optimizer_message': str(opt.message)}
    return params, lambda pred: apply(np.asarray(pred, dtype=np.float64), opt.x)


def load_predictions(folder, ids):
    result = {}
    for sid in ids:
        with np.load(folder / f'{sid}.npz', allow_pickle=False) as z:
            result[sid] = z['pred']
    return result


def pooled_sufficient(pred, truth, mask):
    rows = []
    for axis in range(2):
        valid = mask[:, axis]
        x = truth[valid, axis].astype(np.float64)
        y = pred[valid, axis].astype(np.float64)
        rows.append((len(x), x.sum(), y.sum(), np.square(x).sum(),
                     np.square(y).sum(), np.dot(x, y)))
    return np.asarray(rows, dtype=np.float64)


def ccc_from_sufficient(s):
    n, sx, sy, sxx, syy, sxy = np.moveaxis(s, -1, 0)
    mx, my = sx/n, sy/n
    cov = sxy/n - mx*my
    vx, vy = sxx/n - mx*mx, syy/n - my*my
    denom = vx + vy + (mx-my)**2
    return np.where(denom > 1e-12, 2*cov/denom, np.nan)


def bootstrap(tables, sufficient, folds, repeat, out):
    """Resample paired songs within each fixed outer fold; never pool repeats."""
    ids = sorted(folds)
    rng = np.random.default_rng(2026 + repeat)
    locations = {sid: i for i, sid in enumerate(ids)}
    by_fold = [[locations[sid] for sid in ids if folds[sid] == fold] for fold in range(1, 6)]
    if sorted(map(len, by_fold)) != [360, 360, 360, 360, 362]:
        raise AssertionError('Outer-fold sizes changed')
    draws = np.concatenate([rng.choice(group, size=(10000, len(group)), replace=True)
                            for group in by_fold], axis=1)
    rows = []
    for reference in CONTRASTS:
        for metric in METRICS:
            if metric.endswith('_pooled_ccc'):
                axis = 0 if metric.startswith('valence') else 1
                left = sufficient['B3_level_only'][:, axis]
                right = sufficient[reference][:, axis]
                samples = (ccc_from_sufficient(left[draws].sum(axis=1))
                           - ccc_from_sufficient(right[draws].sum(axis=1)))
                difference = float(ccc_from_sufficient(left.sum(axis=0))
                                   - ccc_from_sufficient(right.sum(axis=0)))
            else:
                delta = (tables['B3_level_only'].loc[ids, metric]
                         - tables[reference].loc[ids, metric]).to_numpy(dtype=float)
                samples = np.nanmean(delta[draws], axis=1)
                difference = float(np.nanmean(delta))
            if not np.isfinite(samples).all():
                raise AssertionError('Nonfinite bootstrap contrast')
            rows.append({'repeat': repeat, 'contrast': 'B3_level_only minus ' + reference,
                         'metric': metric, 'difference': difference,
                         'ci95_lower': float(np.quantile(samples, .025)),
                         'ci95_upper': float(np.quantile(samples, .975)),
                         'bootstrap_draws': 10000, 'bootstrap_unit': 'song',
                         'stratification': 'original outer fold'})
    pd.DataFrame(rows).to_csv(out, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--reference-run', required=True)
    p.add_argument('--plan-root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.plan_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    index, _ = load_index(args.manifest, args.reference_run)
    protocol = {'name': 'B3 calibration-matched single-model controls',
                'repeat_count': 3, 'outer_folds': 5,
                'calibration_songs_per_fold': 144,
                'control': 'global equal-song calibration label mean and prediction mean, per-axis gamma; preserve centered trajectory',
                'gamma_bounds': GAMMA_BOUNDS, 'gamma_start': [1., 1.],
                'b3_alpha_bounds': [[0., 1.], [0., 1.]],
                'objective': 'mean song MED', 'optimizer': 'L-BFGS-B',
                'optimizer_options': OPT_OPTIONS,
                'bootstrap': '10000 paired song draws stratified within original outer folds; separate repeats',
                'outer_labels_used_for_fit_selection_or_calibration': False}
    protocol_file = out / 'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Frozen calibration protocol changed')
    protocol_file.write_text(json.dumps(protocol, indent=2))
    all_summary = []
    all_parameters = []
    for repeat in range(3):
        nested = root / ('b3_nested' if repeat == 0 else f'b3_nested_repeat{repeat}')
        level = root / ('b3_nested_level_only' if repeat == 0 else f'b3_nested_repeat{repeat}_level_only')
        repeat_dir = out / f'repeat_{repeat}'
        repeat_dir.mkdir(exist_ok=True)
        predictions = {name: {} for name in MODELS}
        folds = {}
        for fold in range(1, 6):
            source = nested / f'fold_{fold}'
            split = json.loads((source / 'split.json').read_text())
            cal_ids, outer_ids = split['calibration_ids'], split['outer_ids']
            groups = [set(split[k]) for k in ('fit_ids', 'checkpoint_ids', 'calibration_ids', 'outer_ids')]
            if sum(map(len, groups)) != 1802 or len(set.union(*groups)) != 1802:
                raise AssertionError('Four-way song partition overlap or omission')
            calibration = {}
            b2_cal, mert_cal = {}, {}
            for sid in cal_ids:
                with np.load(source / 'calibration_predictions' / f'{sid}.npz', allow_pickle=False) as z:
                    calibration[sid] = (z['mean'], z['mask'])
                    b2_cal[sid], mert_cal[sid] = z['b2'], z['mert']
            b2_params, b2_apply = fit_single(calibration, b2_cal)
            mert_params, mert_apply = fit_single(calibration, mert_cal)
            b3_params, b3_detail = fit_calibrator(cal_ids, b2_cal, mert_cal, index, scale_fixed=1.)
            saved_b3 = json.loads((level / f'fold_{fold}' / 'validation_metrics.json').read_text())['metadata']
            if not np.allclose(b3_params[:2], [saved_b3['alpha_valence'], saved_b3['alpha_arousal']], atol=1e-8):
                raise AssertionError('Archived B3 calibration mismatch')
            raw_b2 = load_predictions(source / 'B2' / 'evaluation' / 'predictions', outer_ids)
            raw_mert = load_predictions(source / 'MERT' / 'evaluation' / 'predictions', outer_ids)
            archived_b3 = load_predictions(level / f'fold_{fold}' / 'predictions', outer_ids)
            fold_predictions = {'B2_raw': raw_b2, 'MERT_raw': raw_mert,
                                'B2_calibrated': {}, 'MERT_calibrated': {}, 'B3_level_only': archived_b3}
            for sid in outer_ids:
                fold_predictions['B2_calibrated'][sid] = b2_apply(raw_b2[sid]).astype(np.float32)
                fold_predictions['MERT_calibrated'][sid] = mert_apply(raw_mert[sid]).astype(np.float32)
                recalculated = fuse(raw_b2[sid], raw_mert[sid], b3_params).astype(np.float32)
                if not np.allclose(recalculated, archived_b3[sid], atol=2e-7):
                    raise AssertionError('Archived B3 prediction mismatch: ' + sid)
                if sid in folds:
                    raise AssertionError('Repeated outer song')
                folds[sid] = fold
            for name in MODELS:
                predictions[name].update(fold_predictions[name])
            all_parameters.extend([{'repeat': repeat, 'fold': fold, 'model': model, **params}
                                   for model, params in (('B2_calibrated', b2_params),
                                                         ('MERT_calibrated', mert_params))])
            all_parameters.append({'repeat': repeat, 'fold': fold, 'model': 'B3_level_only',
                                   'alpha': b3_params[:2].tolist(), **b3_detail})
            print(f'CALIBRATED repeat={repeat} fold={fold}', flush=True)
        if set(folds) != set(index):
            raise AssertionError('Incomplete held-out coverage')
        tables, sufficient = {}, {name: [] for name in MODELS}
        for name in MODELS:
            model_dir = repeat_dir / name
            if name in ('B2_calibrated', 'MERT_calibrated'):
                result_root(model_dir)
                save_report(model_dir, predictions[name], index, sorted(index),
                            {'model': name, 'repeat': repeat,
                             'outer_labels_used_for_calibration': False})
            table, metrics = evaluate(predictions[name], index, sorted(index))
            tables[name] = table.set_index('song_id')
            all_summary.append({'repeat': repeat, 'model': name, 'songs': len(folds), **metrics})
            for sid in sorted(index):
                z = load_song(index, sid)
                sufficient[name].append(pooled_sufficient(predictions[name][sid], z['mean'], z['mask']))
            sufficient[name] = np.asarray(sufficient[name])
        pd.DataFrame([r for r in all_summary if r['repeat'] == repeat]).to_csv(repeat_dir / 'summary.csv', index=False)
        bootstrap(tables, sufficient, folds, repeat, repeat_dir / 'paired_bootstraps.csv')
    pd.DataFrame(all_summary).to_csv(out / 'summary_by_repeat.csv', index=False)
    pd.DataFrame(all_parameters).to_json(out / 'calibration_parameters.json', orient='records', indent=2)
    grouped = pd.DataFrame(all_summary).groupby('model', sort=False)[list(METRICS)].agg(['mean', 'std'])
    grouped.columns = [f'{metric}_{stat}' for metric, stat in grouped.columns]
    grouped.to_csv(out / 'summary_three_seed.csv')
    print(grouped[['med_mean', 'med_std', 'valence_ccc_mean', 'arousal_ccc_mean']].to_string())


if __name__ == '__main__':
    main()
