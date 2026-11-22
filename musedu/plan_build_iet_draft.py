"""Create an isolated IET manuscript draft integrating the frozen B3 analysis."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ABSTRACT = r"""\begin{abstract}
Continuous music emotion recognition must predict both the level and the
within-song evolution of valence and arousal. We first evaluate shape-aware
MUSE-DU B2, which combines frozen music representations, acoustic descriptors,
and convolutional--Transformer temporal modeling with song-balanced point and
within-song correlation losses. Across five song-disjoint DEAM folds, B2
improves within-song Pearson correlation over an architecture-matched
point-only model while its pointwise gains are smaller. We then test a
level-only fusion that adjusts each B2 song mean toward a frozen MERT probe's
song mean while retaining B2's centered temporal trajectory. In three
complete, strictly partitioned five-fold refits, fusion achieves equal-song
mean Euclidean distance (MED) $0.2073\pm0.0005$ (mean and standard deviation
across seeds), compared with $0.2322\pm0.0018$ for paired B2 and
$0.2231\pm0.0005$ for MERT. Calibration-matched B2-only and MERT-only
controls achieve $0.2307\pm0.0005$ and $0.2221\pm0.0006$, respectively.
Fold-stratified paired song-bootstrap intervals favor fusion on MED in every
seed. Its within-song Pearson correlation is unchanged from B2 by
construction, and its pooled valence concordance is worse than both
calibrated controls. Frozen DEAM-only ensemble evaluation on PMEmo and MERP
shows lower MED than the paired B2 ensemble, with wider uncertainty on the
50-song MERP set. All three datasets informed earlier development, so these
results are development-informed evaluations rather than untouched
independent confirmation. The findings distinguish temporal fidelity from
song-level calibration and show where combining them improves error without
improving every concordance measure.
\end{abstract}"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one anchor, found {text.count(old)}: {old[:80]}')
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--b3-text', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    original = (source / 'main.tex').read_text(encoding='utf-8')
    start = original.index(r'\begin{abstract}')
    end = original.index(r'\end{abstract}', start) + len(r'\end{abstract}')
    draft = original[:start] + ABSTRACT + original[end:]
    old = ('The outstanding problem, therefore, is not simply to reduce average emotion-prediction error, '
           'but to determine when explicit trajectory-shape supervision yields better within-song temporal fidelity')
    new = ('The remaining question is how to improve absolute prediction error while retaining the temporal '
           'fidelity learned by shape supervision. A calibrated B2--MERT level fusion tests whether '
           'complementary song-level information helps beyond granting each single model the same '
           'calibration songs. We therefore evaluate both B2 trajectory learning and calibration-matched '
           'fusion under fixed, song-disjoint protocols. The outstanding problem is to determine when '
           'explicit trajectory-shape supervision yields better within-song temporal fidelity')
    draft = replace_once(draft, old, new)
    old = r'\end{enumerate}' + '\n' + r'\section{Proposed Model: Shape-Aware MUSE-DU (B2)}'
    new = (r'  \item \textbf{A calibration-matched level-fusion analysis.} We refit B2 and a MERT probe '
           'across three seeds and five fixed outer folds with separate calibration songs, compare '
           'the resulting level-only fusion against both raw and calibrated single-model controls, '
           'and audit paired fold-stratified song-bootstrap intervals. The fusion preserves B2\'s '
           'centered trajectory and is not presented as an independently better temporal tracker.' + '\n'
           r'\end{enumerate}' + '\n' + r'\section{Proposed Model: Shape-Aware MUSE-DU (B2)}')
    draft = replace_once(draft, old, new)
    draft = replace_once(draft, r'\section{Discussion}',
                         r'\input{B3_manuscript_text.tex}' + '\n\n' + r'\section{Discussion}')
    discussion = (r'\subsection{Interpreting the calibration-matched fusion}' + '\n'
                  'The final level-only fusion improved equal-song MED over both calibrated '
                  'single-model controls in all three refits. Because the fusion alters only '
                  'song-level means, its within-song Pearson correlation is inherited exactly '
                  'from B2. The pooled valence CCC decline relative to both controls limits '
                  'any blanket superiority claim. This comparison supports complementary '
                  'song-level information under the specified calibration procedures, while '
                  'leaving other possible calibration families and independent-corpus '
                  'confirmation as limitations.\n\n')
    draft = replace_once(draft, r'\subsection{Statistical evidence, reliability and study limitations}',
                         discussion + r'\subsection{Statistical evidence, reliability and study limitations}')
    conclusion = (r'\printbibliography')
    addition = ('The separate three-seed nested level-fusion analysis achieved equal-song '
                'MED $0.2073\\pm0.0005$, below both raw and calibration-matched B2 and '
                'MERT controls. Its temporal correlation is B2\'s by construction; pooled '
                'valence CCC remains unfavorable to both calibrated controls. The external '
                'ensembles and the source-corpus experiments were development-informed, '
                'so the fusion result supports a bounded accuracy--trajectory trade-off '
                'rather than an untouched independent superiority claim.\n\n'
                + conclusion)
    draft = replace_once(draft, conclusion, addition)
    (out / 'main.tex').write_text(draft, encoding='utf-8')
    shutil.copy2(args.b3_text, out / 'B3_manuscript_text.tex')
    shutil.copy2(source / 'references.bib', out / 'references.bib')
    shutil.copytree(source / 'figures', out / 'figures', dirs_exist_ok=True)
    print(out / 'main.tex')


if __name__ == '__main__':
    main()
