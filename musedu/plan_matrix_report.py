"""Regenerate all 2x2/three-repeat tables, bootstraps, and frame shards."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .published_baselines.io import load_index,evaluate
from .plan_matrix_2x2 import REPEATS,sha

METRICS=('med','valence_mae','arousal_mae','valence_rmse','arousal_rmse',
         'valence_ccc','arousal_ccc','valence_within_pearson',
         'arousal_within_pearson','valence_pooled_ccc','arousal_pooled_ccc')
BOOT_METRICS=('med','valence_within_pearson','arousal_within_pearson')
CONTRASTS=(('training_at_point','shape','point','point','point'),
           ('training_at_shape','shape','shape','point','shape'),
           ('selection_given_point','point','shape','point','point'),
           ('selection_given_shape','shape','shape','shape','point'))

def collect(root,index,ids):
    fold_rows=[];per_song=[];repeat_rows=[];manifest=[]
    frozen=json.loads((root/'frozen_splits.json').read_text())
    environment=json.loads((root/'environment.json').read_text())
    for repeat in range(3):
        for train in ('point','shape'):
            for select in ('point','shape'):
                predictions={};selected=[]
                for fold in range(1,6):
                    folder=root/f'repeat_{repeat}'/f'train_{train}'/f'fold_{fold}'/f'select_{select}'
                    metrics_file=folder/'validation_metrics.json'
                    if not metrics_file.is_file():raise FileNotFoundError(metrics_file)
                    report=json.loads(metrics_file.read_text())
                    meta=report['metadata'];selected.append(meta['selected_epoch'])
                    history=json.loads((folder.parent/'history.json').read_text())
                    if len(history)!=25:raise AssertionError('Incomplete training trajectory')
                    best=min(history,key=lambda row:(row['validation_'+select],row['epoch']))
                    if meta['selected_epoch']!=best['epoch']:
                        raise AssertionError('Checkpoint selection differs from saved validation history')
                    checkpoint=folder/'selected_checkpoint.pt'
                    epoch_file=folder.parent/'epoch_checkpoints'/f'epoch_{best["epoch"]:02d}.pt'
                    if not checkpoint.is_file() or not epoch_file.is_file():
                        raise AssertionError('Selected or source checkpoint missing')
                    if sha(checkpoint)!=meta['checkpoint_sha256'] or sha(epoch_file)!=meta['checkpoint_sha256']:
                        raise AssertionError('Selected checkpoint hash mismatch')
                    fold_rows.append({'repeat':repeat,'fold':fold,'seed':meta['seed'],
                                      'train':train,'select':select,
                                      'selected_epoch':meta['selected_epoch'],
                                      'songs':meta['outer_song_count'],**report['metrics']})
                    manifest.append({'repeat':repeat,'fold':fold,'seed':meta['seed'],
                                     'train':train,'select':select,
                                     'dataset':'DEAM',
                                     'dataset_manifest_sha256':frozen['deam_manifest_sha256'],
                                     'split_protocol_sha256':frozen['source_protocol_sha256'],
                                     'b2_code_sha256':environment['code_sha256']['model.py'],
                                     'mert_repository':environment['mert_repository'],
                                     'mert_revision':environment['mert_revision'],
                                     'python_version':environment['python'],
                                     'torch_version':environment['torch'],
                                     'cuda_device':environment['cuda_device'],
                                     'fit_songs':meta['fit_song_count'],
                                     'checkpoint_songs':meta['checkpoint_song_count'],
                                     'outer_songs':meta['outer_song_count'],
                                     'checkpoint_epoch':meta['selected_epoch'],
                                     'checkpoint_sha256':meta['checkpoint_sha256'],
                                     'normalizer_sha256':sha(folder.parent/'normalizer.npz'),
                                     'train_weight':0. if train=='point' else .02,
                                     'select_weight':0. if select=='point' else .02})
                    table=pd.read_csv(folder/'validation_by_song.csv',dtype={'song_id':str})
                    if len(table)!=meta['outer_song_count']:
                        raise AssertionError('Missing per-song metric rows')
                    counts={}
                    for file in (folder/'predictions').glob('*.npz'):
                        if file.stem in predictions:raise AssertionError('Duplicate outer prediction')
                        with np.load(file,allow_pickle=False) as z:
                            predictions[file.stem]=z['pred']
                            counts[file.stem]=int(z['mask'].all(axis=1).sum())
                    for row in table.to_dict('records'):
                        if row['song_id'] not in counts:raise AssertionError('Missing song prediction')
                        per_song.append({'dataset':'DEAM','model':'B2',
                                         'repeat':repeat,'fold':fold,'seed':meta['seed'],
                                         'train':train,'select':select,
                                         'checkpoint_epoch':meta['selected_epoch'],
                                         'evaluation_checkpoint_id':meta['checkpoint_sha256'],
                                         'valid_count':counts[row['song_id']],**row})
                if set(predictions)!=set(ids) or len(predictions)!=1802:
                    raise AssertionError('Incomplete repeated OOF predictions')
                _,metrics=evaluate(predictions,index,ids)
                repeat_rows.append({'repeat':repeat,'train':train,'select':select,
                                    'songs':1802,'mean_selected_epoch':float(np.mean(selected)),
                                    'sd_selected_epoch':float(np.std(selected,ddof=1)),**metrics})
    return pd.DataFrame(fold_rows),pd.DataFrame(per_song),pd.DataFrame(repeat_rows),pd.DataFrame(manifest)

def bootstrap(per_song):
    rng=np.random.default_rng(2026)
    ids=sorted(per_song.song_id.unique())
    rows=[]
    for repeat in range(3):
        subset=per_song[per_song.repeat==repeat].set_index(['train','select','song_id'])
        draws=rng.integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
        for name,ta,sa,tb,sb in CONTRASTS:
            for metric in BOOT_METRICS:
                a=np.array([subset.loc[(ta,sa,sid),metric] for sid in ids],dtype=float)
                b=np.array([subset.loc[(tb,sb,sid),metric] for sid in ids],dtype=float)
                difference=a-b
                sample=np.nanmean(difference[draws],axis=1)
                rows.append({'repeat':repeat,'contrast':name,'metric':metric,
                             'difference_first_minus_second':float(np.nanmean(difference)),
                             'ci95_lower':float(np.quantile(sample,.025)),
                             'ci95_upper':float(np.quantile(sample,.975)),
                             'bootstrap_draws':10000,'bootstrap_unit':'song',
                             'first':ta+'/'+sa,'second':tb+'/'+sb})
    return pd.DataFrame(rows)

def frame_shards(root,split):
    output=root/'per_frame_predictions';output.mkdir(exist_ok=True)
    for repeat in range(3):
        for train in ('point','shape'):
            for select in ('point','shape'):
                path=output/f'repeat_{repeat}_{train}_{select}.parquet'
                if path.exists():continue
                rows=[]
                for fold in range(1,6):
                    folder=root/f'repeat_{repeat}'/f'train_{train}'/f'fold_{fold}'/f'select_{select}'
                    meta=json.loads((folder/'validation_metrics.json').read_text())['metadata']
                    for file in (folder/'predictions').glob('*.npz'):
                        with np.load(file,allow_pickle=False) as z:
                            pred=z['pred'];truth=z['mean'];mask=z['mask'];times=z['times']
                        for i,t in enumerate(times):
                            rows.append({'dataset':'DEAM','song_id':file.stem,'fold':fold,
                                         'seed':meta['seed'],'repeat':repeat,'model':'B2',
                                         'training_loss':train,'selection_loss':select,
                                         'checkpoint_epoch':meta['selected_epoch'],
                                         'timestamp_sec':float(t),
                                         'valid_valence':bool(mask[i,0]),
                                         'valid_arousal':bool(mask[i,1]),
                                         'y_valence':float(truth[i,0]) if mask[i,0] else np.nan,
                                         'y_arousal':float(truth[i,1]) if mask[i,1] else np.nan,
                                         'pred_valence':float(pred[i,0]),
                                         'pred_arousal':float(pred[i,1])})
                frame=pd.DataFrame(rows)
                if len(frame)!=130002 or frame.duplicated(['song_id','timestamp_sec']).any():
                    raise AssertionError('Frame key/count mismatch')
                frame.to_parquet(path,index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root)
    index,_=load_index(a.manifest,a.reference_run);ids=sorted(index)
    fold,per_song,repeat,manifest=collect(root,index,ids)
    fold.to_csv(root/'fold_metrics.csv',index=False)
    per_song.to_csv(root/'per_song_metrics.csv',index=False)
    repeat.to_csv(root/'seed_summary.csv',index=False)
    manifest.to_csv(root/'experiment_manifest.csv',index=False)
    bootstrap(per_song).to_csv(root/'paired_bootstraps.csv',index=False)
    frame_shards(root,None)
    summary=repeat.groupby(['train','select'])[list(METRICS)].agg(['mean','std'])
    summary.to_csv(root/'matrix_summary.csv')
    print(summary[['med','valence_within_pearson','arousal_within_pearson']].to_string())

if __name__=='__main__':main()
