"""Verify frozen five-fold predictions and original published anchor metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .published_baselines.io import load_index,evaluate
from .plan_matrix_2x2 import prepare,ORIGINAL

ANCHORS={'B2':{'med':.2274,'valence_within_pearson':.2165,
               'arousal_within_pearson':.3485},
         'B2_point':{'med':.2292,'valence_within_pearson':.1275,
                     'arousal_within_pearson':.2366}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--reference-run',required=True)
    p.add_argument('--original-run',default=str(ORIGINAL))
    p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.out)
    _,index,splits=prepare(a.manifest,a.reference_run,root,a.original_run)
    result={'distinct_songs':len(index),'outer_fold_sizes':[len(s['outer_ids']) for s in splits],
            'internal_fit_sizes':[len(s['fit_ids']) for s in splits],
            'internal_checkpoint_sizes':[len(s['checkpoint_ids']) for s in splits],
            'anchors':{}}
    original=Path(a.original_run)
    for method,expected in ANCHORS.items():
        predictions={}
        for split in splits:
            folder=original/method/f'fold_{split["fold"]}'/'predictions'
            for file in folder.glob('*.npz'):
                if file.stem in predictions:raise AssertionError('Duplicate prediction '+file.stem)
                with np.load(file,allow_pickle=False) as z:
                    predictions[file.stem]=z['pred']
        if set(predictions)!=set(index):raise AssertionError(method+' missing OOF songs')
        metrics=evaluate(predictions,index,sorted(index))[1]
        discrepancies={name:metrics[name]-value for name,value in expected.items()}
        if any(abs(delta)>5e-5 for delta in discrepancies.values()):
            raise AssertionError(f'Anchor discrepancy {method}: {discrepancies}')
        result['anchors'][method]={'recomputed':metrics,'manuscript_rounded':expected,
                                   'difference':discrepancies,'prediction_songs':len(predictions)}
    (root/'phase0_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
