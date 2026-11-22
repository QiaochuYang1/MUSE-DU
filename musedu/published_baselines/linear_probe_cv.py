"""Run the frozen MERT linear probe on the existing five DEAM outer folds.

This entry point intentionally needs only NumPy and scikit-learn, avoiding a
PyTorch import for a purely linear model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .linear_probe import run_mert_linear


def _ccc(y, p):
    if len(y) < 2:
        return float('nan')
    denominator = np.var(y) + np.var(p) + (np.mean(y) - np.mean(p)) ** 2
    return float(2 * np.mean((y - np.mean(y)) * (p - np.mean(p))) / denominator) if denominator > 1e-12 else float('nan')


def _pearson(y, p):
    if len(y) < 3:
        return float('nan')
    y = y - np.mean(y)
    p = p - np.mean(p)
    denominator = np.linalg.norm(y) * np.linalg.norm(p)
    return float(np.dot(y, p) / denominator) if denominator > 1e-9 else float('nan')


def _row(sid, pred, song):
    truth = song['mean']
    mask = song['mask'].astype(bool)
    row = {'song_id': sid}
    for axis, name in enumerate(('valence', 'arousal')):
        use = mask[:, axis]
        y = truth[use, axis].astype(np.float64)
        p = pred[use, axis].astype(np.float64)
        row[f'{name}_mae'] = float(np.mean(np.abs(y-p))) if len(y) else float('nan')
        row[f'{name}_rmse'] = float(np.sqrt(np.mean((y-p)**2))) if len(y) else float('nan')
        row[f'{name}_ccc'] = _ccc(y, p)
        row[f'{name}_within_pearson'] = _pearson(y, p)
    both = mask.all(1)
    row['med'] = float(np.linalg.norm(pred[both]-truth[both], axis=1).mean()) if both.any() else float('nan')
    return row


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(index, ids, predictions, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'predictions').mkdir(exist_ok=True)
    rows = []
    truth_pooled = [[], []]
    pred_pooled = [[], []]
    for sid in ids:
        with np.load(index[sid], allow_pickle=False) as archive:
            song = {name: archive[name] for name in ('times', 'mean', 'mask')}
        pred = predictions[sid]
        if pred.shape != song['mean'].shape or not np.isfinite(pred).all():
            raise ValueError(f'Invalid prediction for {sid}')
        rows.append(_row(sid, pred, song))
        np.savez_compressed(directory/'predictions'/f'{sid}.npz', pred=pred,
                            mean=song['mean'], mask=song['mask'], times=song['times'])
        for axis in range(2):
            use = song['mask'][:, axis].astype(bool)
            truth_pooled[axis].append(song['mean'][use, axis])
            pred_pooled[axis].append(pred[use, axis])
    _write_csv(directory/'validation_by_song.csv', rows)
    metrics = {key: float(np.nanmean([r[key] for r in rows])) for key in rows[0] if key != 'song_id'}
    for axis, name in enumerate(('valence', 'arousal')):
        metrics[f'{name}_pooled_ccc'] = _ccc(np.concatenate(truth_pooled[axis]), np.concatenate(pred_pooled[axis]))
    (directory/'validation_metrics.json').write_text(json.dumps({'metrics': metrics, 'metadata': metadata}, indent=2))
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--cv-root', required=True)
    args = parser.parse_args()
    with open(args.manifest, newline='', encoding='utf-8') as handle:
        index = {str(row['song_id']): row['file'] for row in csv.DictReader(handle)}
    root = Path(args.cv_root)
    protocol = json.loads((root/'protocol.json').read_text())
    folds = protocol['folds']
    if len(index) != 1802 or set(index) != set().union(*(set(fold) for fold in folds)):
        raise ValueError('Manifest IDs differ from fixed five-fold protocol')
    out = root/'mert_linear'
    all_rows = []
    for fold_number, outer in enumerate(folds, 1):
        target = out/f'fold_{fold_number}'
        if (target/'validation_metrics.json').is_file():
            print(f'SKIP fold={fold_number}', flush=True)
            continue
        training = [sid for j, fold in enumerate(folds) if j != fold_number-1 for sid in fold]
        shuffled = list(training)
        np.random.default_rng(protocol['split_seed']+fold_number).shuffle(shuffled)
        nstop = int(round(.1*len(shuffled)))
        stop, fit = shuffled[:nstop], shuffled[nstop:]
        if set(outer) & (set(fit) | set(stop)):
            raise AssertionError('Outer fold leakage')
        print(f'START fold={fold_number} fit={len(fit)} checkpoint={len(stop)} outer={len(outer)}', flush=True)
        pred, detail = run_mert_linear(index, fit, stop, outer, args.features_dir)
        metadata = {'method': 'mert_linear', 'fold': fold_number,
                    'fit_song_count': len(fit), 'checkpoint_song_count': len(stop),
                    'outer_song_count': len(outer), 'detail': detail,
                    'outer_labels_used_for_selection': False,
                    'outer_labels_used_for_training': False}
        rows, metrics = _report(index, outer, pred, target, metadata)
        print(f'DONE fold={fold_number} MED={metrics["med"]:.6f} alpha={detail["selected_alpha"]}', flush=True)
    all_pred = {}
    for fold_number in range(1, 6):
        target = out/f'fold_{fold_number}'
        if not (target/'validation_metrics.json').is_file():
            return
        for archive_path in (target/'predictions').glob('*.npz'):
            if archive_path.stem in all_pred:
                raise AssertionError('Song predicted twice')
            with np.load(archive_path, allow_pickle=False) as archive:
                all_pred[archive_path.stem] = archive['pred']
    if set(all_pred) != set(index):
        raise AssertionError('Five-fold predictions incomplete')
    rows, metrics = _report(index, sorted(index), all_pred, out, {'method': 'mert_linear', 'outer_folds': 5})
    (out/'out_of_fold_by_song.csv').write_bytes((out/'validation_by_song.csv').read_bytes())
    (out/'crossval_summary.json').write_text(json.dumps({'method': 'mert_linear', 'songs': 1802,
        'outer_folds': 5, 'all_songs_scored_once': True, 'metrics': metrics}, indent=2))
    print(f'AGGREGATE MED={metrics["med"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
