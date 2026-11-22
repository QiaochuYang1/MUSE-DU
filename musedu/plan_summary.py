"""Write an audit-friendly narrative from the three completed seed summaries."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root)
    seed=pd.read_csv(root/'seed_summary.csv')
    boot=pd.read_csv(root/'paired_bootstraps.csv')
    if len(seed)!=12 or len(boot)!=36:raise AssertionError('Incomplete matrix statistics')
    lines=['# Full-budget 2×2 DEAM diagnostic','',
           'These are new 25-epoch diagnostic fits, distinct from the historical patience-five results. Each cell is one complete five-fold, 1,802-song out-of-fold evaluation repeated under three seed groups. Values below are mean ± sample SD across those three repeats.','',
           '| Training / checkpoint selection | Equal-song MED | Valence within-song r | Arousal within-song r |',
           '|---|---:|---:|---:|']
    for train in ('point','shape'):
        for select in ('point','shape'):
            x=seed[(seed.train==train)&(seed.select==select)]
            values=[f'{x[col].mean():.4f} ± {x[col].std(ddof=1):.4f}' for col in
                    ('med','valence_within_pearson','arousal_within_pearson')]
            lines.append(f'| {train} / {select} | '+' | '.join(values)+' |')
    lines += ['', 'Paired differences below are first minus second and are based on the same songs within each fitted seed group. Negative MED differences and positive correlation differences favor the first condition. Percentile 95% intervals use 10,000 song-resampled draws and condition on the fitted models.','',
              '| Contrast | Metric | Seed A | Seed B | Seed C |',
              '|---|---|---:|---:|---:|']
    for contrast in boot.contrast.unique():
        for metric in ('med','valence_within_pearson','arousal_within_pearson'):
            x=boot[(boot.contrast==contrast)&(boot.metric==metric)].sort_values('repeat')
            values=[f'{r.difference_first_minus_second:+.4f} [{r.ci95_lower:+.4f}, {r.ci95_upper:+.4f}]' for r in x.itertuples()]
            lines.append(f'| {contrast} | {metric} | '+' | '.join(values)+' |')
    lines += ['', 'The historical early-stopped B2 and point-only results remain in the baseline comparison. The new full-budget P/S cell can select later epochs than that historical point-only protocol; direct substitution would conflate checkpoint regime and training objective.','']
    comparison=root/'baseline_comparison.csv'
    if comparison.exists():
        base=pd.read_csv(comparison)
        if 'mert_330m_2026_adapted' in set(base.model):
            lines += ['## Five-fold comparator context','',
                      'The 2026 MERT-330M row is a matched-protocol adaptation with a frozen encoder and trainable weighted-layer MLP probe. It is ranked here using the same equal-song MED implementation as all other rows, not the publisher-reported frame-weighted score. The earlier MERT probe remains in the table.','',
                      '| Model | Equal-song MED | Valence within-song r | Arousal within-song r |',
                      '|---|---:|---:|---:|']
            for name in ('B2','B2_point','mert_probe','mert_330m_2026_adapted'):
                r=base[base.model==name].iloc[0]
                lines.append(f'| {name} | {r.med:.4f} | {r.valence_within_pearson:.4f} | {r.arousal_within_pearson:.4f} |')
            lines += ['', 'All remaining prior baselines and their full metric vectors are retained in `baseline_comparison.csv`.','']
    (root/'results_overview.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote results_overview.md')

if __name__=='__main__':main()
