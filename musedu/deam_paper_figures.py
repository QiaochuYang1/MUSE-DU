"""Publication figures from saved DEAM five-fold held-out predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CV=Path(r'E:\deam_cv5_stage1')
STACK=Path(r'E:\deam_cv5_stage1_adapted40\opensmile_librosa_ridge_adapted')
COLORS={'truth':'#202124','B2':'#087e8b','B2_point':'#e09f3e'}

def table(method):
    path=(STACK if method=='opensmile_librosa_ridge_adapted' else CV/method)/'out_of_fold_by_song.csv'
    return pd.read_csv(path,dtype={'song_id':str}).set_index('song_id').sort_index()

def save(fig,folder,stem):
    fig.savefig(folder/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(a.manifest,dtype={'song_id':str}).set_index('song_id')
    b2=table('B2');point=table('B2_point')
    assert b2.index.equals(point.index) and len(b2)==1802

    # Figure 2: select by annotation dynamics only, without consulting model errors.
    candidates=[]
    for sid,row in manifest.iterrows():
        with np.load(row['file'],allow_pickle=False) as z:
            y=z['mean'];mask=z['mask'].astype(bool)
            variability=np.mean([np.std(y[mask[:,j],j]) for j in range(2)])
            candidates.append((variability,sid))
    candidates.sort(key=lambda q:(q[0],q[1]))
    sid=candidates[(95*len(candidates))//100][1]
    fold=int(b2.loc[sid,'fold'])
    with np.load(manifest.loc[sid,'file'],allow_pickle=False) as z:
        times=z['times'];truth=z['mean'];mask=z['mask'].astype(bool)
    with np.load(CV/'B2'/f'fold_{fold}'/'predictions'/f'{sid}.npz',allow_pickle=False) as z:pred=z['pred']
    with np.load(CV/'B2_point'/f'fold_{fold}'/'predictions'/f'{sid}.npz',allow_pickle=False) as z:pred_point=z['pred']
    fig,axes=plt.subplots(2,1,figsize=(9,5.8),sharex=True,layout='constrained')
    for j,(ax,axis) in enumerate(zip(axes,('Valence','Arousal'))):
        ax.plot(times,np.where(mask[:,j],truth[:,j],np.nan),color=COLORS['truth'],lw=2,label='Annotation')
        ax.plot(times,pred[:,j],color=COLORS['B2'],lw=1.6,label='B2 temporal shape')
        ax.plot(times,pred_point[:,j],color=COLORS['B2_point'],lw=1.4,label='B2 point only')
        ax.set(ylabel=axis,ylim=(-1,1));ax.grid(alpha=.18)
    axes[0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.31),frameon=False)
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'Held-out DEAM song {sid} (fold {fold})',fontsize=12)
    save(fig,out,'figure_2_trajectories')

    # Figure 3: each mark is one paired held-out song; x positive means lower error.
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.2),layout='constrained')
    shape_summary={}
    for ax,axis in zip(axes,('valence','arousal')):
        dx=(point[f'{axis}_mae']-b2[f'{axis}_mae']).to_numpy()
        dy=(b2[f'{axis}_within_pearson']-point[f'{axis}_within_pearson']).to_numpy()
        finite=np.isfinite(dx)&np.isfinite(dy)
        ax.hexbin(dx[finite],dy[finite],gridsize=36,mincnt=1,cmap='Blues',linewidths=0)
        ax.axhline(0,color='#555',lw=.8);ax.axvline(0,color='#555',lw=.8)
        ax.set(xlabel='MAE improvement (point only − B2)',ylabel='Within-song Pearson gain (B2 − point only)',title=axis.title())
        ax.grid(alpha=.12)
        shape_summary[axis]={'songs':int(finite.sum()),'mae_improved':int((dx[finite]>0).sum()),
                             'pearson_improved':int((dy[finite]>0).sum()),
                             'both_improved':int(((dx[finite]>0)&(dy[finite]>0)).sum())}
    fig.suptitle('Temporal correlation versus pointwise accuracy',fontsize=12)
    save(fig,out,'figure_3_shape_vs_accuracy')

    # Figure 4: paired 10,000-resample song bootstrap, positive is favorable to B2.
    boot=json.loads((CV/'paired_song_bootstrap.json').read_text())
    names=['B2_point','opensmile_librosa_ridge_adapted','opensmile_lstm']
    labels=['Point-only B2','Adapted Ridge stack','OpenSMILE–LSTM']
    metrics=[('med','MED reduction'),('valence_within_pearson','Valence Pearson gain'),
             ('arousal_within_pearson','Arousal Pearson gain')]
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,(key,title) in zip(axes,metrics):
        for i,(name,label) in enumerate(zip(names,labels)):
            entry=boot['comparisons'][name][key]
            sign=-1 if key=='med' else 1
            estimate=sign*entry['difference']
            lo,hi=sorted(sign*np.asarray(entry['ci95']))
            ax.errorbar(estimate,i,xerr=[[estimate-lo],[hi-estimate]],fmt='o',color='#087e8b',capsize=3)
        ax.axvline(0,color='#555',lw=.9)
        ax.set(yticks=range(3),yticklabels=labels if ax is axes[0] else ['']*3,
               ylim=(2.5,-.5),xlabel='B2 improvement',title=title)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Paired held-out improvements with 95% song-bootstrap intervals',fontsize=12)
    save(fig,out,'figure_4_paired_confidence_intervals')
    (out/'figure_metadata.json').write_text(json.dumps({'figure_2_song_id':sid,
        'figure_2_selection':'95th percentile annotation variability across all 1802 songs; model predictions not consulted',
        'figure_2_fold':fold,'figure_3':shape_summary,
        'figure_4_bootstrap_draws':boot['draws'],'figure_4_unit':boot['unit']},indent=2))

if __name__=='__main__':main()
