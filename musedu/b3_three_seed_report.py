"""Audit and summarize three full nested B3 refits, including both base models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ('med', 'valence_within_pearson', 'arousal_within_pearson',
           'valence_ccc', 'arousal_ccc', 'valence_pooled_ccc', 'arousal_pooled_ccc')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-root', required=True)
    args = p.parse_args()
    root = Path(args.plan_root)
    rows = []
    expected_splits = None
    checkpoint_hashes = {'B2': set(), 'MERT': set()}
    for repeat in range(3):
        nested = root / ('b3_nested' if repeat == 0 else f'b3_nested_repeat{repeat}')
        level = root / ('b3_nested_level_only' if repeat == 0 else f'b3_nested_repeat{repeat}_level_only')
        protocol = json.loads((nested / 'protocol.json').read_text())
        assert protocol['outer_labels_used_for_fit_selection_or_calibration'] is False
        assert protocol['split_sha256'] == json.loads((root / 'b3_nested' / 'protocol.json').read_text())['split_sha256']
        fold_splits = []
        for fold in range(1, 6):
            folder = nested / f'fold_{fold}'
            split = json.loads((folder / 'split.json').read_text())
            groups = [set(split[k]) for k in ('fit_ids', 'checkpoint_ids', 'calibration_ids', 'outer_ids')]
            assert sum(map(len, groups)) == 1802 and len(set.union(*groups)) == 1802
            fold_splits.append(split)
            cal = json.loads((folder / 'calibration.json').read_text())
            assert cal['seed'] == 2026 + fold + 1000 * repeat
            assert cal['outer_labels_used_for_fit_selection_or_calibration'] is False
            for model in ('B2', 'MERT'):
                checkpoint_hashes[model].add(cal[model.lower() + '_checkpoint_sha256'])
                pred_ids = {f.stem for f in (folder / model / 'evaluation' / 'predictions').glob('*.npz')}
                assert pred_ids == groups[3]
            fused_ids = {f.stem for f in (level / f'fold_{fold}' / 'predictions').glob('*.npz')}
            assert fused_ids == groups[3]
        assert len(set().union(*(set(s['outer_ids']) for s in fold_splits))) == 1802
        if expected_splits is None:
            expected_splits = fold_splits
        else:
            assert fold_splits == expected_splits
        summary = pd.read_csv(level / 'summary.csv').set_index('model')
        assert set(summary.index) == {'B2', 'MERT', 'B3_level_only'}
        for model in summary.index:
            assert int(summary.loc[model, 'songs']) == 1802
            rows.append({'repeat': repeat, 'model': model,
                         **{metric: float(summary.loc[model, metric]) for metric in METRICS}})
    assert all(len(values) == 15 for values in checkpoint_hashes.values())
    out = pd.DataFrame(rows)
    out.to_csv(root / 'b3_three_seed_by_repeat.csv', index=False)
    grouped = out.groupby('model', sort=False)[list(METRICS)].agg(['mean', 'std'])
    grouped.columns = [f'{metric}_{stat}' for metric, stat in grouped.columns]
    grouped.to_csv(root / 'b3_three_seed_summary.csv')
    differences = []
    pivot = out.set_index(['repeat', 'model'])
    for repeat in range(3):
        for reference in ('B2', 'MERT'):
            for metric in METRICS:
                differences.append({'repeat': repeat, 'contrast': f'B3_level_only minus {reference}',
                                    'metric': metric,
                                    'difference': float(pivot.loc[(repeat, 'B3_level_only'), metric]
                                                        - pivot.loc[(repeat, reference), metric])})
    pd.DataFrame(differences).to_csv(root / 'b3_three_seed_differences.csv', index=False)
    print(grouped[['med_mean', 'med_std', 'valence_within_pearson_mean',
                   'arousal_within_pearson_mean', 'valence_pooled_ccc_mean']].to_string())


if __name__ == '__main__':
    main()
