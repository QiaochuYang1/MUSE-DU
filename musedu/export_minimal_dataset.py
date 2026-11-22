"""Export the paper's archived predictions and summary results without synthetic rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

CV = Path(r'E:\deam_cv5_stage1')
STACK = Path(r'E:\deam_cv5_stage1_adapted40')
ABL = Path(r'E:\deam_cv5_paper_ablations')
PMEMO = Path(r'E:\deam_pmemo_frozen_transfer')
MERP = Path(r'E:\deam_merp_frozen_transfer')
EXTERNAL_BASE = Path(r'E:\deam_frozen_baseline_transfer')
BASELINES = (
    ('conformer', CV / 'conformer'),
    ('librosa_rf', CV / 'librosa_rf'),
    ('librosa_xgboost', CV / 'librosa_xgboost'),
    ('mert_probe', CV / 'mert_probe'),
    ('opensmile_lstm', CV / 'opensmile_lstm'),
    ('opensmile_mlp', CV / 'opensmile_mlp'),
    ('opensmile_librosa_ridge_adapted', STACK / 'opensmile_librosa_ridge_adapted'),
)
ABLATIONS = (
    ('B2', 0.02, 'fusion_temporal', CV / 'B2'),
    ('B2_point', 0.0, 'fusion_temporal', CV / 'B2_point'),
    ('shape_0p005', 0.005, 'fusion_temporal', ABL / 'shape_0p005'),
    ('shape_0p01', 0.01, 'fusion_temporal', ABL / 'shape_0p01'),
    ('shape_0p05', 0.05, 'fusion_temporal', ABL / 'shape_0p05'),
    ('shape_0p1', 0.1, 'fusion_temporal', ABL / 'shape_0p1'),
    ('mert_only', 0.02, 'mert_temporal', ABL / 'mert_only'),
    ('acoustic_only', 0.02, 'acoustic_temporal', ABL / 'acoustic_only'),
    ('framewise_fusion', 0.02, 'fusion_framewise', ABL / 'framewise_fusion'),
)
METRICS = ('valence_mae', 'valence_rmse', 'valence_ccc', 'valence_within_pearson',
           'arousal_mae', 'arousal_rmse', 'arousal_ccc', 'arousal_within_pearson',
           'med', 'valence_pooled_ccc', 'arousal_pooled_ccc')
BOOT_METRICS = ('valence_mae', 'arousal_mae', 'valence_rmse', 'arousal_rmse',
                'valence_ccc', 'arousal_ccc', 'valence_within_pearson',
                'arousal_within_pearson', 'med')


def archive(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ('pred', 'mean', 'mask', 'times')}


def assert_aligned(a, b, where):
    if (not np.array_equal(a['times'], b['times']) or
        not np.array_equal(a['mask'], b['mask']) or
        not np.allclose(a['mean'], b['mean'], rtol=0, atol=1e-6, equal_nan=True)):
        raise AssertionError(f'Prediction/label alignment mismatch: {where}')
    if not np.isfinite(a['mean'][a['mask'].astype(bool)]).all():
        raise AssertionError(f'Nonfinite valid labels: {where}')
    if a['pred'].shape != a['mean'].shape or not np.isfinite(a['pred']).all():
        raise AssertionError(f'Invalid predictions: {where}')


def val(x, valid=True):
    return float(x) if valid else ''


def write_deam(dest):
    protocol = json.loads((CV / 'protocol.json').read_text())
    fields = ['song_id', 'fold', 'timestamp_sec', 'valid_valence', 'valid_arousal',
              'true_valence', 'true_arousal', 'b2_valence', 'b2_arousal',
              'point_valence', 'point_arousal']
    count = 0
    with (dest / 'DEAM_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for fold, ids in enumerate(protocol['folds'], 1):
            for sid in ids:
                b2 = archive(CV / 'B2' / f'fold_{fold}' / 'predictions' / f'{sid}.npz')
                point = archive(CV / 'B2_point' / f'fold_{fold}' / 'predictions' / f'{sid}.npz')
                assert_aligned(b2, point, f'DEAM {sid}')
                for i, t in enumerate(b2['times']):
                    mask = b2['mask'][i].astype(bool)
                    writer.writerow([sid, fold, float(t), int(mask[0]), int(mask[1]),
                                     val(b2['mean'][i, 0], mask[0]),
                                     val(b2['mean'][i, 1], mask[1]),
                                     float(b2['pred'][i, 0]), float(b2['pred'][i, 1]),
                                     float(point['pred'][i, 0]), float(point['pred'][i, 1])])
                    count += 1
    if count != 130002:
        raise AssertionError(f'Unexpected DEAM frame count: {count}')
    return count


def write_external(dest):
    fields = ['dataset', 'song_id', 'timestamp_sec', 'valid_valence', 'valid_arousal',
              'true_valence', 'true_arousal', 'b2_valence', 'b2_arousal',
              'point_valence', 'point_arousal', 'opensmile_lstm_valence',
              'opensmile_lstm_arousal', 'adapted_stack_valence', 'adapted_stack_arousal']
    counts = {}
    with (dest / 'External_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for dataset, root, expected in (('PMEmo', PMEMO, 767), ('MERP', MERP, 50)):
            names = sorted(p.stem for p in (root / 'B2_temporal_shape' / 'predictions').glob('*.npz'))
            if len(names) != expected:
                raise AssertionError(f'{dataset}: expected {expected} songs, got {len(names)}')
            frames = 0
            for sid in names:
                b2 = archive(root / 'B2_temporal_shape' / 'predictions' / f'{sid}.npz')
                point = archive(root / 'B2_point_only' / 'predictions' / f'{sid}.npz')
                lstm = archive(EXTERNAL_BASE / dataset.lower() / 'opensmile_lstm' /
                               'predictions' / f'{dataset.lower()}_{sid}.npz')
                stack = archive(EXTERNAL_BASE / dataset.lower() /
                                'opensmile_librosa_ridge_adapted' / 'predictions' /
                                f'{dataset.lower()}_{sid}.npz')
                for other in (point, lstm, stack):
                    assert_aligned(b2, other, f'{dataset} {sid}')
                for i, t in enumerate(b2['times']):
                    mask = b2['mask'][i].astype(bool)
                    writer.writerow([dataset, sid, float(t), int(mask[0]), int(mask[1]),
                                     val(b2['mean'][i, 0], mask[0]),
                                     val(b2['mean'][i, 1], mask[1]),
                                     *[float(a['pred'][i, j]) for a in
                                       (b2, point, lstm, stack) for j in range(2)]])
                    frames += 1
            counts[dataset] = {'songs': len(names), 'frames': frames}
    return counts


def metric_rows(writer, name, root, extra):
    summary = json.loads((root / 'crossval_summary.json').read_text())
    if summary.get('songs') != 1802 or not summary.get('all_songs_scored_once'):
        raise AssertionError(f'Incomplete cross-validation: {name}')
    for fold in range(1, 6):
        data = json.loads((root / f'fold_{fold}' / 'validation_metrics.json').read_text())
        n = data['metadata']['outer_song_count']
        writer.writerow([name, 'fold', fold, n, *extra,
                         *[data['metrics'][k] for k in METRICS]])
    writer.writerow([name, 'all_folds', '', 1802, *extra,
                     *[summary['metrics'][k] for k in METRICS]])


def write_results(dest):
    with (dest / 'Baseline_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'scope', 'fold', 'songs', *METRICS])
        for name, root in BASELINES:
            metric_rows(writer, name, root, ())
    with (dest / 'Ablation_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'scope', 'fold', 'songs', 'shape_loss_weight',
                         'component_configuration', *METRICS])
        for name, weight, configuration, root in ABLATIONS:
            metric_rows(writer, name, root, (weight, configuration))


def write_bootstrap(dest):
    with (dest / 'Bootstrap_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'reference_model', 'comparator', 'metric',
                         'difference_reference_minus_comparator', 'ci95_lower',
                         'ci95_upper', 'bootstrap_unit', 'draws', 'seed', 'songs'])
        cv = json.loads((CV / 'paired_song_bootstrap.json').read_text())
        for comparator, result in cv['comparisons'].items():
            for metric in BOOT_METRICS:
                r = result[metric]
                writer.writerow(['DEAM', 'B2', comparator, metric, r['difference'],
                                 *r['ci95'], 'song', cv['draws'], cv['seed'], cv['songs']])
        for dataset, root in (('PMEmo', PMEMO), ('MERP', MERP)):
            report = json.loads((root / 'paired_transfer_report.json').read_text())
            for comparator, result in report['paired_differences'].items():
                for metric in BOOT_METRICS:
                    r = result[metric]
                    writer.writerow([dataset, 'B2_temporal_shape', comparator, metric,
                                     r['difference'], *r['ci95'], 'song',
                                     report['bootstrap_draws'], report['bootstrap_seed'],
                                     report['songs']])


README = """MUSE-DU minimal results dataset
================================

These files export archived experimental predictions and results. No row was
created from a rounded manuscript summary value. DEAM and external song IDs
refer to their respective source datasets; audio files are not redistributed.

DEAM_predictions.csv
--------------------
One row per 0.5-second DEAM annotation timestamp for each of 1,802 songs,
covering five song-disjoint outer folds. fold is 1--5. b2_* and point_*
are actual held-out predictions from B2 temporal-shape and identical
point-only architecture. true_* is blank where that axis lacks a valid label.
valid_valence and valid_arousal are binary masks. Metric calculations must
use only valid labels. The five folds contain 360, 360, 360, 360, and 362
songs. Each outer training pool reserved 10% of its songs for internal
checkpoint selection. No outer-fold labels selected checkpoints.

External_predictions.csv
------------------------
One row per aligned 0.5-second timestamp from 767 PMEmo and 50 MERP songs.
Models are B2, point-only B2, OpenSMILE-LSTM, and the adapted two-family
OpenSMILE-Librosa Ridge stack. All were fitted on DEAM only, with no external
labels used for fitting, checkpoint selection, or stack fitting. PMEmo
source labels on 0--1 were converted by the fixed formula 2*x-1. The four
known DEAM anchor recordings in MERP were excluded. External song_id values
are unique within dataset; use (dataset,song_id,timestamp_sec) as the key.
The adapted stack is not the published Mazzola Hybrid Ridge model.

Baseline_results.csv and Ablation_results.csv
---------------------------------------------
Each contains per-fold rows and one all_folds row per model. Seven baseline
configurations are included, excluding B2 and its point-only control.
Ablation_results.csv contains B2, point-only B2, four additional shape-loss
weights, and three component removals. The scope column distinguishes fold
metrics from metrics recomputed over all 1,802 saved outer-fold predictions.
Pointwise and within-song metrics are equal-song averages; pooled CCC is
computed after concatenating valid time steps across songs.

Bootstrap_results.csv
---------------------
Saved paired comparisons of B2 minus comparator for DEAM, PMEmo, and MERP.
The interval is a 95% percentile interval from 10,000 song-level resamples,
using seed 2026. Positive MED difference means B2 has higher error; positive
Pearson difference means B2 has higher within-song correlation.

Metric definitions
------------------
For each song, MED is the mean Euclidean distance between the two-dimensional
prediction and truth at time steps valid for both axes. Reported MED is the
arithmetic mean of song MEDs. MAE, RMSE, and CCC are calculated per axis per
song, then averaged equally across songs. Within-song Pearson is undefined
for fewer than three valid observations or near-zero centered-vector norm;
undefined song values are omitted from the corresponding mean. The CSVs
retain masks so readers can reproduce this handling.

Sources and provenance
----------------------
DEAM audio/annotations: public DEAM dataset; this study used the imsparsh
Kaggle mirror, version 1, because the official download did not complete.
Official DEAM checksums for all 1,802 songs were unavailable. PMEmo:
official PMEmo 2019 dynamic annotation archive. MERP: official released
WAVs and filtered dynamic annotations. See the manuscript for dataset
citations and licensing. Frozen MERT-v1-95M revision
12af15fef9d0ac838c3f475bfbbf26d2060dd4f5 and 18 acoustic descriptors
were prepared at the original annotation timestamps. The DEAM feature
manifest does not independently store the MERT revision for each song.

The external evaluation is cross-dataset evaluation; older PMEmo/MERP
artifacts existed in the study workspace. Earlier uncertainty and
explainability diagnostics belong to a separate checkpoint and are not
included here.
"""


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Refusing nonempty output folder: {out}')
    deam_frames = write_deam(out)
    external = write_external(out)
    write_results(out)
    write_bootstrap(out)
    (out / 'README.txt').write_text(README, encoding='utf-8')
    files = ['DEAM_predictions.csv', 'External_predictions.csv',
             'Baseline_results.csv', 'Ablation_results.csv',
             'Bootstrap_results.csv', 'README.txt']
    zip_path = out.parent / (out.name + '.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for name in files:
            z.write(out / name, arcname=name)
    print(json.dumps({'folder': str(out), 'zip': str(zip_path),
                      'deam_frames': deam_frames, 'external': external,
                      'sha256': {name: sha256(out / name) for name in files}},
                     indent=2))


if __name__ == '__main__':
    main()
