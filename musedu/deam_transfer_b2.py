"""Frozen DEAM-only B2 training and zero-shot external prediction."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from .experiment import load_manifest
from .deam_stage1_cv import internal_split,run_b2
from .published_baselines.io import save_report,result_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deam-manifest',required=True)
    p.add_argument('--external-manifest',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    df=load_manifest(a.deam_manifest)
    if df.dataset.nunique()!=1 or set(df.dataset)!={'deam'} or len(df)!=1802:
        raise ValueError('Expected full 1,802-song DEAM training manifest')
    with open(a.external_manifest,newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    if not rows or len(set(r['song_id'] for r in rows))!=len(rows):
        raise ValueError('Empty or duplicate external manifest')
    external={r['song_id']:r['file'] for r in rows}
    datasets=set(r['dataset'] for r in rows)
    if len(datasets)!=1 or 'deam' in datasets:
        raise ValueError('Expected one external dataset')
    dataset=datasets.pop()
    ids=list(external)
    for sid,path in external.items():
        with np.load(path,allow_pickle=False) as archive:
            times=archive['times']
            if len(times)>1 and not np.allclose(np.diff(times),.5,atol=1e-3):
                raise ValueError(f'{sid}: external timestamps are not 0.5-second spaced')
            if dataset=='pmemo' and times[0]<15:
                raise ValueError(f'{sid}: PMEmo timestamp starts before 15 seconds')
    fit,stop=internal_split(sorted(df.song_id.astype(str)),a.seed)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    protocol={'source_dataset':'DEAM','source_songs':1802,'fit_songs':len(fit),
              'internal_checkpoint_songs':len(stop),'external_dataset':dataset,
              'external_songs':len(ids),'seed':a.seed,'epochs':a.epochs,
              'architecture':'B2 v1 fusion hidden=96',
              'checkpoint_selector':'equal-song Huber + 0.02 centered correlation',
              'external_labels_used_for_training_or_selection':False,
              'external_preprocessing':'dataset adapter fixed conversion; exact archived 0.5-second timestamps'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name,objective in [('B2_temporal_shape','point_shape'),('B2_point_only','point')]:
        dest=out/name
        if (dest/'validation_metrics.json').is_file():
            print(f'SKIP {name}',flush=True);continue
        result_root(dest)
        print(f'START {name} DEAM fit={len(fit)} checkpoint={len(stop)} {dataset}={len(ids)}',flush=True)
        predictions,detail=run_b2(df,None,fit,stop,ids,a.seed,a.epochs,a.device,8,
                                  objective,predict_lookup=external,
                                  checkpoint_path=dest/'model.pt')
        metadata={'method':name,'training_dataset':'DEAM','evaluation_dataset':dataset,
                  'objective':objective,'detail':detail,
                  'external_labels_used_for_training_or_selection':False}
        metrics=save_report(dest,predictions,external,ids,metadata)
        print(f'DONE {name} {dataset} {metrics}',flush=True)


if __name__=='__main__':main()
