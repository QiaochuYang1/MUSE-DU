"""Prespecified staged valence experiments; every trial remains in the report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .valence_experiment import run
from .valence_audit import delay_folds,annotations_and_boundaries


def main(manifest,reference,root,annotation_root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    audit=root/'audit'
    if (audit/'delay_selection.json').is_file():
        delay=json.loads((audit/'delay_selection.json').read_text())['selected_delay_seconds']
    else:delay=delay_folds(manifest,reference,audit)
    if not (audit/'identity_audit.json').is_file():
        annotations_and_boundaries('outputs/deam_mert/deam_manifest.csv',manifest,reference,audit,annotation_root)
    configs={
        'continuity':dict(representation='last',delay=0.,harmonic=False),
        'layers':dict(representation='mix',delay=0.,harmonic=False),
        'harmony':dict(representation='mix',delay=delay,harmonic=True),
    }
    if delay:configs['aligned']=dict(representation='mix',delay=delay,harmonic=False)
    plan={'configs':configs,'seeds':list(range(2026,2031)),
          'screening_seed':2026,'selection':'highest valence within-song Pearson with per-axis MAE <= stable baseline + .01 and arousal Pearson >= baseline - .03; if none eligible retain best experimental candidate but flag',
          'expansion':'best candidate per architecture repeated on five seeds',
          'long_valence':'additional ECHO context branch tested after harmonic input',
          'independent_confirmation':False}
    (root/'experiment_plan.json').write_text(json.dumps(plan,indent=2))
    summaries={}
    for architecture,label in (('v1','B2'),('echo_event','E6')):
        candidates={}
        for name,config in configs.items():
            result=run(manifest,reference,root/'experiments'/f'{label}_{name}'/'seed2026',architecture,seed=2026,**config)
            candidates[name]=result
        if architecture=='echo_event':
            config={**configs['harmony'],'long_valence':True}
            candidates['long_valence']=run(manifest,reference,root/'experiments'/'E6_long_valence'/'seed2026',architecture,seed=2026,**config)
        baseline=json.loads((root/'shape_stable'/label/'seed2026'/'validation_summary.json').read_text())['validation_metrics']
        eligible=[]
        for name,result in candidates.items():
            m=result['validation_metrics']
            if all(m[f'{a}_mae']<=baseline[f'{a}_mae']+.01 for a in ('valence','arousal')) and m['arousal_within_pearson']>=baseline['arousal_within_pearson']-.03:
                eligible.append(name)
        chosen=max(eligible or candidates,key=lambda name:candidates[name]['validation_metrics']['valence_within_pearson'])
        summaries[label]={'chosen':chosen,'passed_guard':chosen in eligible,'screening_results':candidates}
        (root/'screening_selection.json').write_text(json.dumps(summaries,indent=2))
        config=({**configs['harmony'],'long_valence':True} if chosen=='long_valence' else configs[chosen])
        for seed in range(2027,2031):
            run(manifest,reference,root/'experiments'/f'{label}_{chosen}'/f'seed{seed}',architecture,seed=seed,**config)
    # Development-fold robustness: baseline and chosen v1 model on inner folds of original training songs.
    folds=json.loads((audit/'delay_selection.json').read_text())['folds']
    chosen=summaries['B2']['chosen'];config=configs[chosen]
    for i,valid in enumerate(folds):
        train=[sid for j,fold in enumerate(folds) if j!=i for sid in fold]
        for name,source,settings in (
            ('baseline','outputs/deam_mert/deam_manifest.csv',dict(representation='last',delay=0.,harmonic=False)),
            ('enhanced',manifest,config)):
            run(source,reference,root/'development_folds'/f'fold{i}_{name}','v1',seed=2026,
                split_override={'train':train,'val':valid},**settings)
    (root/'pipeline_complete.json').write_text(json.dumps({'status':'complete','test_set_used':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--root',required=True);p.add_argument('--annotation-root',required=True)
    a=p.parse_args();main(a.manifest,a.reference,a.root,a.annotation_root)
