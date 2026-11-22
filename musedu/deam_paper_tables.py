"""Assemble manuscript-ready ablation tables from complete outer-fold outputs."""
from __future__ import annotations
import json
from pathlib import Path

BASE=Path(r'E:\deam_cv5_stage1')
ABL=Path(r'E:\deam_cv5_paper_ablations')
OUT=Path('outputs/paper_figures')

def row(name,root):
    path=root/name/'crossval_summary.json'
    if not path.exists():return None
    d=json.loads(path.read_text())
    if d.get('songs')!=1802 or not d.get('all_songs_scored_once'):
        raise ValueError(f'Incomplete five-fold data: {name}')
    m=d['metrics']
    return (m['med'],m['valence_mae'],m['arousal_mae'],
            m['valence_within_pearson'],m['arousal_within_pearson'],
            m['valence_ccc'],m['arousal_ccc'])

def main():
    lines=['# DEAM five-fold paper additions','',
           'All values are from one held-out prediction per each of 1,802 DEAM songs. '
           'The inner validation songs select checkpoints with the fixed equal-song Huber '
           '+ 0.02 temporal-correlation rule. Each run uses a 25-epoch cap, patience 5, '
           '0.5-second predictions, and the same five song-disjoint folds and seeds. '
           'None of the 35 new fits selected a checkpoint at the epoch cap. '
           'Higher Pearson/CCC and lower MED/MAE are better.','',
           '## Table 8. Shape-loss weight sensitivity','',
           '| Training weight λs | MED ↓ | Valence MAE ↓ | Arousal MAE ↓ | Valence within-song Pearson ↑ | Arousal within-song Pearson ↑ |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    spec=[('0 (point only)','B2_point',BASE),('0.005','shape_0p005',ABL),
          ('0.01','shape_0p01',ABL),('0.02 (B2)','B2',BASE),
          ('0.05','shape_0p05',ABL),('0.1','shape_0p1',ABL)]
    missing=[]
    for label,name,root in spec:
        x=row(name,root)
        if x is None:missing.append(name);continue
        lines.append(f'| {label} | {x[0]:.4f} | {x[1]:.4f} | {x[2]:.4f} | {x[3]:.4f} | {x[4]:.4f} |')
    lines+=['','At 0.01, MED is lowest; at 0.02, within-song Pearson is higher. '
            'Larger weights (0.05–0.1) raise pointwise error. '
            'The weight sweep is exploratory; selecting a new weight from these outer-fold results would require fresh evaluation.','',
            '## Table 9. Component ablation','',
            '| Configuration | MERT | Acoustic descriptors | Cross-frame model | Parameters | MED ↓ | Valence within-song Pearson ↑ | Arousal within-song Pearson ↑ |',
            '| --- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |']
    spec=[('Full B2','B2',BASE,'Yes','Yes','Yes',336772),
          ('Remove acoustic descriptors','mert_only',ABL,'Yes','No','Yes',335044),
          ('Remove MERT','acoustic_only',ABL,'No','Yes','Yes',263044),
          ('Remove cross-frame modeling','framewise_fusion',ABL,'Yes','Yes','No',132100)]
    for label,name,root,mert,acoustic,temporal,parameters in spec:
        x=row(name,root)
        if x is None:missing.append(name);continue
        lines.append(f'| {label} | {mert} | {acoustic} | {temporal} | {parameters:,} | {x[0]:.4f} | {x[3]:.4f} | {x[4]:.4f} |')
    lines+=['','Both feature removals raise MED. Acoustic-only raises arousal within-song Pearson, '
            'so that component effect is mixed across metrics. The framewise model replaces '
            'both convolution and self-attention with per-frame feed-forward blocks; it has fewer '
            'parameters and is an architectural ablation, not a parameter-matched control.','',
            '## Figure captions','',
            '**Figure 2.** Held-out valence and arousal trajectories for the DEAM song at the 95th percentile of annotation variability, chosen without consulting predictions. B2 and its point-only control use the matching outer-fold models.','',
            '**Figure 3.** Paired held-out change in MAE versus within-song Pearson for all 1,802 songs. Positive axes favor B2 over point-only B2.','',
            '**Figure 4.** B2 improvement relative to the point-only control, adapted OpenSMILE–Librosa Ridge stack, and OpenSMILE–LSTM. Error bars are paired 95% percentile intervals from 10,000 song-level bootstrap draws; positive values favor B2.','']
    if missing:lines+=['**Pending complete five-fold runs:** '+', '.join(missing)+'.','']
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'paper_additions.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Missing:',missing)

if __name__=='__main__':main()
