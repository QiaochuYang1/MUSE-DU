"""Export comparison tables and publication figures from saved results only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .published_baselines.io import load_index,evaluate

ORIGINAL=Path(r'E:\deam_cv5_stage1')
ADAPTED=Path(r'E:\deam_cv5_stage1_adapted40')

def baseline_table(root,index,original_run=ORIGINAL,adapted_run=ADAPTED):
    names=('B2','B2_point','mert_probe','opensmile_lstm','conformer',
           'mert_linear','opensmile_mlp','librosa_xgboost','librosa_rf',
           'opensmile_librosa_ridge_adapted')
    rows=[]
    for name in names:
        base=Path(adapted_run) if name=='opensmile_librosa_ridge_adapted' else Path(original_run)
        path=base/name/'crossval_summary.json'
        result=json.loads(path.read_text())
        rows.append({'model':name,'experiment':'historical five-fold',**result['metrics']})
    new=root/'mert2026'
    if all((new/f'fold_{fold}'/'validation_metrics.json').is_file() for fold in range(1,6)):
        pp={}
        for fold in range(1,6):
            for f in (new/f'fold_{fold}'/'predictions').glob('*.npz'):
                if f.stem in pp:raise AssertionError('Duplicate MERT baseline prediction')
                with np.load(f,allow_pickle=False) as z:pp[f.stem]=z['pred']
        if set(pp)!=set(index):raise AssertionError('MERT baseline incomplete')
        _,metrics=evaluate(pp,index,sorted(index))
        rows.append({'model':'mert_330m_2026_adapted','experiment':'new matched-protocol adaptation',**metrics})
    result=pd.DataFrame(rows)
    result.to_csv(root/'baseline_comparison.csv',index=False)
    return result

def figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed=pd.read_csv(root/'seed_summary.csv')
    boot=pd.read_csv(root/'paired_bootstraps.csv')
    colors={'point':'#4C78A8','shape':'#E45756'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for i,(metric,label) in enumerate((('med','Equal-song MED (lower is better)'),
                                       ('valence_within_pearson','Valence within-song r'),
                                       ('arousal_within_pearson','Arousal within-song r'))):
        ax=axes[i]
        for train in ('point','shape'):
            for j,select in enumerate(('point','shape')):
                values=seed[(seed.train==train)&(seed.select==select)][metric].to_numpy()
                x=j+(-.1 if train=='point' else .1)
                ax.scatter(np.full(len(values),x),values,color=colors[train],alpha=.45,s=22)
                ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt='o',
                            capsize=3,color=colors[train],markersize=6,
                            label=train+' training' if i==0 and j==0 else None)
        ax.set_xticks([0,1],['Point selector','Shape selector'])
        ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False)
    fig.savefig(root/'figure_A_matrix.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    nice={'training_at_point':'Training: point selector',
          'training_at_shape':'Training: shape selector',
          'selection_given_point':'Selection: point training',
          'selection_given_shape':'Selection: shape training'}
    for panel,(ax,(metric,label)) in enumerate(zip(axes,(('med','MED difference'),
                                         ('valence_within_pearson','Valence r difference'),
                                         ('arousal_within_pearson','Arousal r difference')))):
        subset=boot[boot.metric==metric]
        names=list(subset.contrast.unique())
        for j,name in enumerate(names):
            for repeat in range(3):
                r=subset[(subset.contrast==name)&(subset.repeat==repeat)].iloc[0]
                y=j+(repeat-1)*.16
                ax.plot([r.ci95_lower,r.ci95_upper],[y,y],color=colors['shape' if j<2 else 'point'])
                ax.scatter(r.difference_first_minus_second,y,s=16,
                           color=colors['shape' if j<2 else 'point'])
        ax.axvline(0,color='black',lw=.8)
        ax.set_yticks(range(len(names)),[nice[n] for n in names] if panel==0 else ['']*len(names))
        ax.set_xlabel(label)
        ax.grid(axis='x',alpha=.2)
    fig.savefig(root/'figure_B_paired_differences.pdf',bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True);p.add_argument('--root',required=True)
    p.add_argument('--original-run',default=str(ORIGINAL))
    p.add_argument('--adapted-run',default=str(ADAPTED))
    a=p.parse_args();root=Path(a.root)
    index,_=load_index(a.manifest,a.reference_run)
    baseline_table(root,index,a.original_run,a.adapted_run)
    if (root/'seed_summary.csv').exists() and (root/'paired_bootstraps.csv').exists():
        figures(root)
    print('Exported baseline comparison and available figures')

if __name__=='__main__':main()
