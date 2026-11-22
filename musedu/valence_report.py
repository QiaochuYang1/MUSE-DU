"""Full, unfiltered report of staged valence development experiments."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import load_manifest,read_npz
from .metrics import transition_metrics,event_f1_from_counts
from .matched_temporal import thresholds

METRICS=('valence_within_pearson','arousal_within_pearson','valence_mae','arousal_mae',
         'valence_ccc','arousal_ccc','valence_rmse','arousal_rmse')


def song_table(paths):
    frames=[pd.read_csv(p/'validation_by_song.csv',dtype={'song_id':str}).set_index('song_id').sort_index() for p in paths]
    if any(not frame.index.equals(frames[0].index) for frame in frames):raise ValueError('Different evaluated songs')
    return sum(frame.loc[:,METRICS] for frame in frames)/len(frames)


def summarize(root):
    root=Path(root);selection=json.loads((root/'screening_selection.json').read_text())
    runs={};rows=[]
    for base in (root/'shape_stable',root/'experiments'):
        for group in sorted(base.iterdir()):
            if not group.is_dir():continue
            paths=sorted(p.parent for p in group.glob('seed*/validation_summary.json'))
            if not paths:continue
            name=('stable_' if base.name=='shape_stable' else '')+group.name
            runs[name]=paths
            seed_means=pd.DataFrame([pd.read_csv(p/'validation_by_song.csv').loc[:,METRICS].mean() for p in paths])
            row={'experiment':name,'n_seeds':len(paths),'n_songs':len(pd.read_csv(paths[0]/'validation_by_song.csv'))}
            for metric in METRICS:row[metric]=float(seed_means[metric].mean());row[metric+'_seed_sd']=float(seed_means[metric].std())
            rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(root/'all_results.csv',index=False)
    selected={label:f'{label}_{item["chosen"]}' for label,item in selection.items()}
    comparisons=[(selected['B2'],'stable_B2'),(selected['E6'],'stable_E6'),(selected['E6'],'stable_B2')]
    rng=np.random.default_rng(20261028);paired=[]
    for left,right in comparisons:
        a=song_table(runs[left]);b=song_table(runs[right])
        if not a.index.equals(b.index):raise ValueError('Unpaired songs')
        for metric in METRICS:
            diff=(a[metric]-b[metric]).dropna().to_numpy()
            boot=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(1)
            paired.append({'comparison':f'{left}-{right}','metric':metric,'difference':float(diff.mean()),
                           'low':float(np.quantile(boot,.025)),'high':float(np.quantile(boot,.975))})
    pairs=pd.DataFrame(paired);pairs.to_csv(root/'paired_results.csv',index=False)
    old=load_manifest('outputs/deam_mert/deam_manifest.csv')
    split=json.loads(Path('outputs/deam_primary_musedu/splits.json').read_text());lookup=dict(zip(old.song_id,old.file))
    train_sd=[];sd={}
    for sid in split['train']+split['val']:
        z=read_npz(lookup[sid]);value=float(z['mean'][z['mask'][:,0],0].std());sd[sid]=value
        if sid in split['train']:train_sd.append(value)
    cuts=np.quantile(train_sd,[.25,.5,.75]);groups=[]
    for name in ('stable_B2',selected['B2'],selected['E6']):
        songs=song_table(runs[name]);songs['target_sd']=[sd[sid] for sid in songs.index]
        songs['variability_group']=np.searchsorted(cuts,songs.target_sd)+1
        for group,t in songs.groupby('variability_group'):
            groups.append({'experiment':name,'variability_group':group,'n_songs':len(t),
                           'within_pearson':float(t.valence_within_pearson.mean()),
                           'n_above_05':int((t.valence_within_pearson>=.5).sum()),
                           'n_negative':int((t.valence_within_pearson<0).sum())})
    pd.DataFrame(groups).to_csv(root/'variability_groups.csv',index=False)
    reference_thresholds=thresholds(old,split['train']);temporal=[]
    for name in selected.values():
        for path in runs[name]:
            trans=[];pooled_p=[[],[]];pooled_y=[[],[]]
            for file in (path/'vp').glob('*.npz'):
                with np.load(file) as z:
                    trans.append(transition_metrics(z['pred'],z['truth'],z['mask'],z['times'],.5,reference_thresholds,reference_thresholds))
                    for axis in range(2):
                        keep=z['mask'][:,axis];pooled_p[axis].append(z['pred'][keep,axis]);pooled_y[axis].append(z['truth'][keep,axis])
            t=pd.DataFrame(trans);row={'experiment':name,'seed':path.name}
            for axis,aname in enumerate(('valence','arousal')):
                row[f'{aname}_first_difference_mae']=float(t[f'{aname}_first_difference_mae'].mean())
                row[f'{aname}_transition_f1']=event_f1_from_counts(*(int(t[f'{aname}_event_{c}'].sum()) for c in ('tp','fp','fn')))
                p=np.concatenate(pooled_p[axis]);y=np.concatenate(pooled_y[axis]);cov=np.mean((p-p.mean())*(y-y.mean()))
                row[f'{aname}_pooled_ccc']=float(2*cov/(p.var()+y.var()+(p.mean()-y.mean())**2))
            temporal.append(row)
    temporal_table=pd.DataFrame(temporal)
    baseline_temporal=root/'shape_stable/temporal_summary_by_seed.csv'
    if baseline_temporal.is_file():
        base=pd.read_csv(baseline_temporal).rename(columns={'variant':'experiment'})
        base['experiment']='stable_'+base['experiment']
        temporal_table=pd.concat((base,temporal_table),ignore_index=True)
    temporal_table.to_csv(root/'temporal_results.csv',index=False)
    folds=[]
    for f in (root/'development_folds').glob('*/validation_summary.json'):
        result=json.loads(f.read_text());folds.append({'fold_run':f.parent.name,**result['validation_metrics']})
    fold_table=pd.DataFrame(folds);fold_table.to_csv(root/'fold_results.csv',index=False)
    reliability=pd.read_csv(root/'audit/inter_rater_reliability.csv')
    delay=json.loads((root/'audit/delay_selection.json').read_text())
    best_five=table.loc[table.n_seeds==5].sort_values('valence_within_pearson',ascending=False).iloc[0]
    lines=['# Valence improvement experiments','',
        f'Best five-seed valence Pearson: {best_five["valence_within_pearson"]:.3f} ({best_five["experiment"]}). '
        'The .50 full-corpus research target has not been established.', '',
        'Exploratory DEAM development. All 270 eligible validation songs remain in the primary comparison. '
        'Original scale/calibration/test labels were excluded. Five seeds: 2026–2030. '
        'Every model used equal-song Huber + .02 stable shape loss, AdamW, a 15-epoch cap, patience five, and the same validation selector.',
        '', '| Experiment | Seeds | Valence Pearson | Arousal Pearson | Valence MAE | Arousal MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["experiment"]} | {row["n_seeds"]} | {row["valence_within_pearson"]:.3f} | {row["arousal_within_pearson"]:.3f} | {row["valence_mae"]:.3f} | {row["arousal_mae"]:.3f} |')
    lines+=['','The single-seed rows screen individual changes; the selected candidates were expanded to five seeds using the rule in experiment_plan.json. '
            'Selection and reporting reuse a development validation set, so these results cannot establish independent confirmation.',
            '', 'The loss-only correction changed the five-seed mean by less than .0001 Pearson for both B2 and ECHO. '
            'It fixes the numerical behavior but does not explain the practical gain here.',
            '', '## Paired song comparisons','']
    for row in paired:
        if row['metric'].endswith('within_pearson'):
            lines.append(f'- {row["comparison"]}, {row["metric"]}: {row["difference"]:+.3f} [{row["low"]:+.3f}, {row["high"]:+.3f}] (95% paired-song bootstrap interval).')
    temporal_means=temporal_table.groupby('experiment').mean(numeric_only=True)
    old_event=temporal_means.loc['stable_E6','valence_transition_f1']
    new_event=temporal_means.loc[selected['E6'],'valence_transition_f1']
    lines+=['','## Temporal tradeoff','',
        f'ECHO valence transition-event F1 fell from {old_event:.3f} to {new_event:.3f}, despite improved within-song Pearson and lower first-difference MAE. '
        'This is consistent with smoother trajectories missing abrupt changes. The upgraded ECHO variant should remain experimental; '
        'its improved Pearson does not establish better event detection. Enhanced B2 remains the strongest overall candidate in these comparisons.']
    lines+=['','## Alignment, reliability, and feature checks','',
        f'The training-only alignment probe selected {delay["selected_delay_seconds"]:.1f} seconds: label at time t is paired with audio at t minus that delay. '
        f'The probe gain was {delay["mean_gain_over_zero"]:.3f} and improved in {delay["fold_wins_over_zero"]}/3 training folds. '
        'This is an alignment choice for prediction; it does not identify psychological response latency.',
        '',f'Mean validation split-half correlation across raters: valence '
        f'{reliability.query("split == \'val\' and axis == \'valence\'").split_half_pearson.mean():.3f}, arousal '
        f'{reliability.query("split == \'val\' and axis == \'arousal\'").split_half_pearson.mean():.3f}. '
        'Weak agreement complicates a .50 target; these split-half values are not a hard performance ceiling.',
        '', 'The new cache uses overlapping five-second windows with a 2.5-second hop, exact convolutional frame centers, '
        'pooling within annotation bins, and four frozen layers (3/6/9/12). Harmonic experiments add 57 transparent chord/key template and tonal-change descriptors. '
        'The original caches and checkpoints remain available. See audit/identity_audit.json for the 1,172-song source/label identity check.',
        '', '## Development-fold robustness','',
        'These fixed-configuration comparisons evaluate the original and enhanced B2 inputs on three folds within the original 902 training songs. '
        'The delay/configuration choices were already frozen for these development checks; this is not a nested independent evaluation.', '']
    for row in folds:
        lines.append(f'- {row["fold_run"]}: valence Pearson {row["valence_within_pearson"]:.3f}, arousal {row["arousal_within_pearson"]:.3f}.')
    lines+=['','The fold comparisons use only original training songs. A genuinely independent labeled corpus is still needed before a confirmatory claim.',
            '', 'Detailed outputs: all_results.csv, paired_results.csv, variability_groups.csv, temporal_results.csv, fold_results.csv, and audit/.']
    (root/'improvement_report.md').write_text('\n'.join(lines),encoding='utf-8')
    os.environ['MPLCONFIGDIR']=str((root/'mpl').resolve())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen_names=['stable_B2',selected['B2'],'stable_E6',selected['E6']]
    display=table.set_index('experiment').loc[chosen_names]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,axis in zip(axes,('valence','arousal')):
        ax.bar(np.arange(4),display[f'{axis}_within_pearson'],yerr=display[f'{axis}_within_pearson_seed_sd'],capsize=4)
        ax.set_xticks(np.arange(4),chosen_names,rotation=20,ha='right');ax.set_ylim(0,.65)
        ax.axhline(.5,color='gray',linestyle='--',label='.50 research target')
        ax.set_title(axis.title());ax.set_ylabel('Equal-song within-song Pearson');ax.legend(fontsize=8)
    fig.suptitle('Exploratory DEAM validation; error bars show seed SD');fig.tight_layout()
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    return table,pairs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();summarize(a.root)
