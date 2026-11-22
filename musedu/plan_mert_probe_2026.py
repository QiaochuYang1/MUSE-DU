"""Matched DEAM folds for an adapted 2026 MERT-330M weighted-layer probe.

The archived 24-layer 1024-wide feature cache is reused. The publication did not
pin an exact model revision or disclose all MLP details, so this is an adaptation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .plan_matrix_2x2 import prepare, sha, ORIGINAL
from .published_baselines.io import get_features, result_root, save_report
from .published_baselines.run import run_neural


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True)
    p.add_argument('--original-run',default=str(ORIGINAL))
    p.add_argument('--features',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs',type=int,default=50)
    a=p.parse_args()
    root=Path(a.out).resolve()
    _,index,splits=prepare(a.manifest,a.reference_run,root.parent,a.original_run)
    if a.epochs<16:raise ValueError('Epoch cap must allow patience 15')
    feature_dir=Path(a.features).resolve()
    if {p.stem for p in (feature_dir/'mert').glob('*.npz')}!=set(index):
        raise ValueError('MERT feature cache is incomplete')
    x,z=get_features(index,splits[0]['outer_ids'][0],'mert',feature_dir)
    if x.ndim!=3 or x.shape[1:]!=(24,1024) or len(x)!=len(z['times']):
        raise ValueError('Expected timestamp-aligned 24 x 1024 MERT-330M features')
    root.mkdir(parents=True,exist_ok=True)
    protocol={'name':'Mazzola et al. (2026) inspired MERT-330M weighted-layer MLP (adapted)',
              'source_doi':'10.1007/s00521-026-12370-y',
              'encoder_repository':'m-a-p/MERT-v1-330M',
              'encoder_revision':'5240c2708a5acaee1007f43fb9735c7dcd0b78c9 (local Hugging Face snapshot; per-feature revision not archived)',
              'local_snapshot_pytorch_model_sha256':'0c03774d821a6ad972cb48220b128a10ebea2c790116bd0fd35e5a013f8017f6',
              'encoder_frozen':True,'feature_shape':[24,1024],
              'cached_feature_extraction':'5-second segments, aligned to original 0.5-second DEAM timestamps',
              'probe':'softmax layer weights, LayerNorm, Linear(1024,256), GELU, dropout 0.25, Linear(256,2), tanh',
              'optimizer':'Adam','learning_rate':.001,'weight_decay':0.,'batch_songs':64,
              'gpu_microbatch_songs':16,
              'patience':15,'epoch_cap':a.epochs,'training_loss':'equal-song MSE',
              'checkpoint_selector':'equal-song Huber + 0.02 temporal correlation loss',
              'adaptations':['MLP width and activation unspecified in paper',
                             'shared DEAM internal checkpoint IDs instead of newly randomized 10% validation',
                             'shared B2 checkpoint selection rule',
                             'cached feature extraction revision not verifiable per song'],
              'frozen_splits_sha256':sha(root.parent/'frozen_splits.json')}
    protocol_file=root/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text())!=protocol:
        if any((root/f'fold_{fold}'/'validation_metrics.json').exists() for fold in range(1,6)):
            raise ValueError('Baseline protocol changed after completed fold')
    protocol_file.write_text(json.dumps(protocol,indent=2))
    for split in splits:
        fold=split['fold']
        if fold not in a.folds:continue
        target=root/f'fold_{fold}'
        if (target/'validation_metrics.json').exists():
            print(f'SKIP fold={fold}',flush=True);continue
        if target.exists():
            if {p.name for p in target.iterdir()}!={'predictions'} or any((target/'predictions').iterdir()):
                raise RuntimeError(f'Incomplete baseline output requires inspection: {target}')
        else:
            result_root(target)
        seed=2026+fold
        predictions,history,norm=run_neural(
            index,split['fit_ids'],split['checkpoint_ids'],split['outer_ids'],
            'mert','probe',feature_dir,seed=seed,epochs=a.epochs,device=a.device,
            batch_size=64,lr=.001,patience=15,
            checkpoint_path=target/'selected_checkpoint.pt',optimizer_kind='Adam',
            cache_in_memory=True,microbatch_size=16)
        metadata={'name':protocol['name'],'seed':seed,'fold':fold,
                  'fit_song_count':len(split['fit_ids']),
                  'checkpoint_song_count':len(split['checkpoint_ids']),
                  'outer_song_count':len(split['outer_ids']),
                  'selected_epoch':history['best_epoch'],
                  'selected_validation_score':history['best_selector'],
                  'training_history':history['history'],
                  'outer_labels_used_for_selection':False,
                  'checkpoint_sha256':sha(target/'selected_checkpoint.pt')}
        metrics=save_report(target,predictions,index,split['outer_ids'],metadata)
        print(f'MERT-2026 fold={fold} epoch={history["best_epoch"]} MED={metrics["med"]:.6f}',flush=True)


if __name__=='__main__':main()
