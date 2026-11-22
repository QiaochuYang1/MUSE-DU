"""Recalculate PMEmo and MERP paired B3-minus-B2 MED intervals from saved songs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .published_baselines.io import per_song_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root, out = Path(args.plan_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, count in (('pmemo', 767), ('merp', 50)):
        source = root / 'b3_external_transfer' / dataset
        protocol = json.loads((source / 'protocol.json').read_text())
        if protocol['external_song_count'] != count or protocol['external_labels_used_for_training_checkpoint_selection_or_calibration']:
            raise AssertionError('External protocol changed')
        ids = pd.read_csv(source / 'B3_level_only' / 'validation_by_song.csv',
                          dtype={'song_id': str})['song_id'].tolist()
        saved_ids = {p.stem for p in (source / 'B3_level_only' / 'predictions').glob('*.npz')}
        if len(ids) != count or len(set(ids)) != count or set(ids) != saved_ids:
            raise AssertionError('External song count changed')
        paired = []
        for sid in ids:
            arrays = []
            for model in ('B3_level_only', 'B2_ensemble'):
                with np.load(source / model / 'predictions' / f'{sid}.npz', allow_pickle=False) as z:
                    arrays.append((z['pred'], z['mean'], z['mask'], z['times']))
            if not all(np.array_equal(arrays[0][k], arrays[1][k]) for k in (1, 2, 3)):
                raise AssertionError('External labels/timestamps not paired: ' + sid)
            paired.append(per_song_metrics(*arrays[0][:3])['med']
                          - per_song_metrics(*arrays[1][:3])['med'])
        paired = np.asarray(paired, dtype=np.float64)
        rng = np.random.default_rng(2026)
        draws = rng.integers(0, count, size=(10000, count), dtype=np.int32)
        samples = paired[draws].mean(axis=1)
        previous = pd.read_csv(source / 'paired_bootstraps.csv')
        match = previous[(previous.contrast == 'B3_level_only minus B2_ensemble')
                         & (previous.metric == 'med')].iloc[0]
        difference = float(paired.mean())
        lower, upper = np.quantile(samples, [.025, .975])
        if not np.isclose(difference, match.difference, atol=1e-8):
            raise AssertionError('Previously reported external MED difference changed')
        if not np.allclose([lower, upper], [match.ci95_lower, match.ci95_upper], atol=1e-8):
            raise AssertionError('Previously reported external MED interval changed')
        rows.append({'dataset': dataset, 'songs': count, 'contrast': 'B3_level_only minus B2_ensemble',
                     'metric': 'equal_song_med', 'difference': difference,
                     'ci95_lower': float(lower), 'ci95_upper': float(upper),
                     'bootstrap_draws': 10000, 'bootstrap_unit': 'song',
                     'external_labels_used_for_fitting': False})
    pd.DataFrame(rows).to_csv(out / 'external_med_interval_audit.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
