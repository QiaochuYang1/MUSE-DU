"""DEAM-only OpenSMILE-LSTM and adapted-stack zero-shot transfer."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from .deam_stage1_cv import internal_split
from .published_baselines.io import result_root,save_report
from .published_baselines.run import run_neural,adapted_stack


def read(path):
    with open(path,newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deam-manifest',required=True)
    p.add_argument('--external-manifest',action='append',required=True)
    p.add_argument('--features-dir',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--methods',nargs='+',choices=('opensmile_lstm','opensmile_librosa_ridge_adapted'),
                   default=['opensmile_lstm','opensmile_librosa_ridge_adapted'])
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    source=read(a.deam_manifest)
    if len(source)!=1802 or {r['dataset'] for r in source}!={'deam'}:
        raise ValueError('Expected all 1,802 DEAM songs')
    index={r['song_id']:r['file'] for r in source}
    external={}
    for path in a.external_manifest:
        rows=read(path)
        names={r['dataset'] for r in rows}
        if len(names)!=1 or 'deam' in names:raise ValueError('Expected one external dataset per manifest')
        name=names.pop()
        ids=[name+'_'+r['song_id'] for r in rows if not (name=='merp' and r['song_id'].startswith('deam_'))]
        external[name]=ids
        for r in rows:index[name+'_'+r['song_id']]=r['file']
    root=Path(a.features_dir)
    views=('opensmile','librosa') if 'opensmile_librosa_ridge_adapted' in a.methods else ('opensmile',)
    for name,ids in external.items():
        for sid in ids:
            for view in views:
                archive=root/view/(sid+'.npz')
                if not archive.is_file():raise FileNotFoundError(archive)
                with np.load(archive,allow_pickle=False) as f,np.load(index[sid],allow_pickle=False) as z:
                    if not np.allclose(f['times'],z['times'],atol=1e-3):
                        raise ValueError(f'{sid}: {view} timestamps mismatch')
    training=sorted(r['song_id'] for r in source)
    fit,stop=internal_split(training,2026)
    external_ids=[sid for ids in external.values() for sid in ids]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    protocol={'source_dataset':'DEAM','source_songs':1802,'fit_songs':len(fit),
              'checkpoint_songs':len(stop),'seed':2026,'external_songs':{k:len(v) for k,v in external.items()},
              'external_labels_used_for_training_or_selection':False,
              'opensmile_lstm_epochs':25,'adapted_stack_epochs':40,
              'adapted_stack_OOF_folds':3,'adapted_stack_ridge_alpha':1.0}
    (out/'baseline_transfer_protocol.json').write_text(json.dumps(protocol,indent=2))
    for method in a.methods:
        if all((out/name/method/'validation_metrics.json').is_file() for name in external):
            print(f'SKIP {method}',flush=True);continue
        print(f'START {method} source={len(training)} external={len(external_ids)}',flush=True)
        if method=='opensmile_lstm':
            checkpoint=out/'opensmile_lstm_model.pt'
            predictions,detail,_=run_neural(index,fit,stop,external_ids,'opensmile','lstm',
                a.features_dir,seed=2026,epochs=25,device=a.device,batch_size=8,
                checkpoint_path=checkpoint)
        else:
            predictions,detail,_=adapted_stack(index,training,external_ids,a.features_dir,
                seed=2026,epochs=40,device=a.device,batch_size=8,folds=3)
        for name,ids in external.items():
            dest=out/name/method
            result_root(dest)
            metadata={'method':method,'source_dataset':'DEAM','external_dataset':name,
                      'detail':detail,'external_labels_used_for_training_or_selection':False,
                      'excluded_known_deam_merp_anchors':True if name=='merp' else None}
            metrics=save_report(dest,{sid:predictions[sid] for sid in ids},index,ids,metadata)
            print(f'DONE {method} {name} {metrics}',flush=True)


if __name__=='__main__':main()
