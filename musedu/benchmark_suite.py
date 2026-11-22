"""Batch-run independently refitted, protocol-matched baselines and create CSV reports."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .baselines import METHODS
from .baseline_experiment import run_baseline_training,run_baseline_external


def run_suite(manifest,out,*,methods=None,reference_run=None,external_manifests=None,
              device='cpu',epochs=25,seed=2026,hidden=96,window=120,lr=3e-4,batch_size=8):
    methods=list(methods or METHODS)
    if not methods or len(set(methods))!=len(methods) or set(methods)-set(METHODS):
        raise ValueError(f'Unknown or duplicate baseline methods: {methods}')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method in methods:
        trained=out/method
        rep=run_baseline_training(manifest,trained,method,reference_run=reference_run,
                   device=device,epochs=epochs,seed=seed,hidden=hidden,
                   window=window,lr=lr,batch_size=batch_size)
        row={'method':method,'evaluation':'in_domain_test','dataset':'source',
             'input_view':rep['input_view'],**rep['point_metrics'],
             **{f'uniform_cp_{k}':v for k,v in rep['uniform_conformal'].items()}}
        rows.append(row)
        for external in (external_manifests or []):
            ext=run_baseline_external(external,trained,device=device,window=window)
            rows.append({'method':method,'evaluation':'zero_shot_external',
                 'dataset':ext['external_dataset'],'input_view':rep['input_view'],
                 **ext['point_metrics'],
                 **{f'uniform_cp_{k}':v for k,v in ext['uniform_conformal'].items()}})
    pd.DataFrame(rows).to_csv(out/'baseline_comparison.csv',index=False)
    (out/'baseline_comparison.json').write_text(json.dumps(rows,indent=2))
    return {'n_methods':len(methods),'n_evaluations':len(rows),
            'csv':str((out/'baseline_comparison.csv').resolve())}
