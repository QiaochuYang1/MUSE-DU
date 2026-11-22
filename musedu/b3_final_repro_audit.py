"""Independently verify final B3 control predictions, contrasts, and artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .published_baselines.io import evaluate, load_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--reference-run', required=True)
    p.add_argument('--plan-root', required=True)
    args = p.parse_args()
    root = Path(args.plan_root)
    final = root / 'b3_final_calibration_audit'
    index, _ = load_index(args.manifest, args.reference_run)
    summary = pd.read_csv(final / 'summary_by_repeat.csv')
    protocol = json.loads((final / 'protocol.json').read_text())
    assert protocol['outer_labels_used_for_fit_selection_or_calibration'] is False
    assert len(summary) == 15 and set(summary.model) == {
        'B2_raw', 'MERT_raw', 'B2_calibrated', 'MERT_calibrated', 'B3_level_only'}
    checks = []
    for repeat in range(3):
        nested = root / ('b3_nested' if repeat == 0 else f'b3_nested_repeat{repeat}')
        b3 = root / ('b3_nested_level_only' if repeat == 0 else f'b3_nested_repeat{repeat}_level_only')
        repeat_dir = final / f'repeat_{repeat}'
        boot = pd.read_csv(repeat_dir / 'paired_bootstraps.csv')
        assert len(boot) == 20
        assert set(boot.bootstrap_draws) == {10000}
        assert set(boot.stratification) == {'original outer fold'}
        for reference in ('B2_raw', 'MERT_raw', 'B2_calibrated', 'MERT_calibrated'):
            row = boot[(boot.contrast == 'B3_level_only minus ' + reference) & (boot.metric == 'med')].iloc[0]
            expected = (summary[(summary.repeat == repeat) & (summary.model == 'B3_level_only')].med.iloc[0]
                        - summary[(summary.repeat == repeat) & (summary.model == reference)].med.iloc[0])
            assert np.isclose(row.difference, expected, atol=1e-10)
            assert row.ci95_upper < 0
        for name in ('B2_calibrated', 'MERT_calibrated'):
            folder = repeat_dir / name / 'predictions'
            ids = {path.stem for path in folder.glob('*.npz')}
            assert ids == set(index)
            preds = {}
            for sid in ids:
                with np.load(folder / f'{sid}.npz', allow_pickle=False) as z:
                    preds[sid] = z['pred']
                    assert z['pred'].shape == z['mean'].shape == z['mask'].shape
                    assert np.isfinite(z['pred']).all()
            _, metrics = evaluate(preds, index, sorted(index))
            row = summary[(summary.repeat == repeat) & (summary.model == name)].iloc[0]
            for metric in ('med', 'valence_ccc', 'arousal_ccc', 'valence_pooled_ccc', 'arousal_pooled_ccc'):
                assert np.isclose(metrics[metric], row[metric], atol=1e-10)
            checks.append({'repeat': repeat, 'model': name, 'songs': len(ids),
                           'med_recomputed': metrics['med']})
        outer = set()
        for fold in range(1, 6):
            split = json.loads((nested / f'fold_{fold}' / 'split.json').read_text())
            groups = [set(split[key]) for key in ('fit_ids', 'checkpoint_ids', 'calibration_ids', 'outer_ids')]
            assert sum(map(len, groups)) == 1802 and len(set.union(*groups)) == 1802
            assert not outer.intersection(groups[-1])
            outer.update(groups[-1])
            archived = {p.stem for p in (b3 / f'fold_{fold}' / 'predictions').glob('*.npz')}
            assert archived == groups[-1]
        assert outer == set(index)
    external = pd.read_csv(final / 'external_med_interval_audit.csv')
    assert set(external.dataset) == {'pmemo', 'merp'}
    assert set(external.bootstrap_draws) == {10000}
    assert (external.ci95_upper < 0).all()
    required = ['B3_results.md', 'B3_final_calibration_report.md',
                'B3_manuscript_text.tex', 'b3_three_seed_summary.csv']
    assert all((root / name).is_file() for name in required)
    pd.DataFrame(checks).to_csv(final / 'reproducibility_audit.csv', index=False)
    result = {'status': 'passed', 'repeats': 3, 'outer_folds_per_repeat': 5,
              'held_out_songs_per_repeat': 1802, 'recomputed_calibrated_model_runs': len(checks),
              'deam_contrasts': 12, 'bootstrap_draws_per_contrast': 10000,
              'external_datasets_audited': ['pmemo', 'merp']}
    (final / 'reproducibility_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
