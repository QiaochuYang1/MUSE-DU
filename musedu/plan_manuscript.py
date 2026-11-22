"""Create a LaTeX results addendum using saved measurements, never typed scores."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def fmt(mean,sd):return f'{mean:.4f} $\\pm$ {sd:.4f}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root)
    seeds=pd.read_csv(root/'seed_summary.csv')
    baselines=pd.read_csv(root/'baseline_comparison.csv')
    if len(seeds)!=12:raise AssertionError('Expected four cells and three full seed repeats')
    if 'mert_330m_2026_adapted' not in set(baselines.model):
        raise AssertionError('New five-fold MERT baseline missing')
    lines=[r'\subsection{Additional full-budget DEAM diagnostics}',
           r'\label{subsec:additional-diagnostics}',
           r'The following results are new diagnostics and are separate from the previously reported patience-five experiments. We retained the original five song-disjoint outer DEAM folds and internal validation song IDs. For each of three initialization-seed groups, point-only and temporal-shape models were trained for all 25 epochs. Both checkpoint rules were then applied to each saved trajectory, so selector comparisons do not alter the optimization path. No outer-fold labels entered model selection.',
           r'\begin{table}[t]',r'\centering',
           r'\caption{Full-budget loss-by-checkpoint-selection diagnostic. Each cell is the mean $\pm$ standard deviation of three complete 1,802-song out-of-fold seed repeats. MED is equal-song aggregated; $r$ is the mean defined within-song Pearson correlation. Lower MED and higher $r$ are favorable.}',
           r'\label{tab:loss-selection-matrix}',
           r'\begin{tabular}{llccc}',r'\hline',
           r'Training loss & Selection loss & MED & Valence $r$ & Arousal $r$ \\',
           r'\hline']
    for train in ('point','shape'):
        for select in ('point','shape'):
            x=seeds[(seeds.train==train)&(seeds.select==select)]
            if len(x)!=3:raise AssertionError('Incomplete seed group')
            values=[fmt(x[m].mean(),x[m].std(ddof=1)) for m in
                    ('med','valence_within_pearson','arousal_within_pearson')]
            lines.append(f'{train} & {select} & '+ ' & '.join(values)+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}',
              r'Song-level paired differences and their percentile 95\% intervals from 10,000 paired bootstrap resamples are reported separately for each fitted seed group. These intervals condition on the trained models; the between-seed standard deviations in Table~\ref{tab:loss-selection-matrix} reflect refitting variation.',
              r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figure_A_matrix.pdf}\caption{Four-cell diagnostic across three seed groups. Points show individual complete-seed results; error bars indicate between-seed standard deviation.}\label{fig:matrix}\end{figure}',
              r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figure_B_paired_differences.pdf}\caption{Paired per-song differences for each seed group with 95\% song-bootstrap intervals. The sign is first condition minus second; negative MED differences favor the first condition.}\label{fig:paired-new}\end{figure}',
              r'\begin{table}[t]',r'\centering',
              r'\caption{Five-fold DEAM comparison under the shared equal-song evaluation code. The original results are retained. The 2026 MERT-330M result is a new matched-protocol adaptation, not the publisher-reported frame-weighted MED.}',
              r'\label{tab:baseline-2026}',
              r'\begin{tabular}{lccc}',r'\hline',
              r'Model & MED & Valence $r$ & Arousal $r$ \\',r'\hline']
    for row in baselines.itertuples():
        name=str(row.model).replace('_',r'\_')
        lines.append(f'{name} & {row.med:.4f} & {row.valence_within_pearson:.4f} & {row.arousal_within_pearson:.4f} '+r'\\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}',
              r'The adapted comparator follows the frozen MERT layer-weighted MLP design reported by Mazzola, Seidita, and La Cascia (2026; DOI 10.1007/s00521-026-12370-y). It uses frozen, 24-layer MERT-v1-330M features and a trainable softmax-weighted layer mixture with a 264,986-parameter MLP probe; the B2 prediction head has 336,772 trainable parameters and uses a different frozen encoder. These are not capacity-matched feature extractors. The exact comparator encoder revision cannot be independently verified from the archived per-song feature files. MLP width, activation, training-loss aggregation, and the shared checkpoint rule are documented adaptations. The baseline uses the same outer evaluation songs and metrics as MUSE-DU.',
              r'The previously completed frozen PMEmo and MERP transfer experiments remain distinct from these DEAM cross-validation results.']
    (root/'manuscript_addendum.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote manuscript_addendum.tex')

if __name__=='__main__':main()
