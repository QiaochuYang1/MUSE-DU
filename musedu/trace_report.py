"""Aggregate independently trained TRACE-MER checkpoints and their ensemble."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import song_metrics,within_song_pearson


def report(root,seeds=range(2026,2031)):
    root=Path(root);seeds=list(seeds)
    summaries=[json.loads((root/f'seed{s}'/'validation_summary.json').read_text()) for s in seeds]
    ids=pd.read_csv(root/f'seed{seeds[0]}'/'validation_by_song.csv',dtype={'song_id':str}).song_id
    rows=[];pooled_pred=[[],[]];pooled_truth=[[],[]]
    for sid in ids:
        files=[np.load(root/f'seed{s}'/'vp'/f'{sid}.npz') for s in seeds]
        prediction=np.mean([z['pred'] for z in files],axis=0)
        rows.append({'song_id':sid,**song_metrics(prediction,files[0]['truth'],files[0]['mask']),
                     **within_song_pearson(prediction,files[0]['truth'],files[0]['mask'])})
        for axis in range(2):
            keep=files[0]['mask'][:,axis].astype(bool)
            pooled_pred[axis].append(prediction[keep,axis]);pooled_truth[axis].append(files[0]['truth'][keep,axis])
        for z in files:z.close()
    table=pd.DataFrame(rows);table.to_csv(root/'ensemble_validation_by_song.csv',index=False)
    keys=list(summaries[0]['validation_metrics'])
    def ccc(parts_y,parts_p):
        y=np.concatenate(parts_y);p=np.concatenate(parts_p)
        cov=np.mean((y-y.mean())*(p-p.mean()))
        denominator=np.var(y)+np.var(p)+(y.mean()-p.mean())**2
        return float(2*cov/denominator) if denominator>0 else float('nan')
    pooled={f'{name}_pooled_ccc':ccc(pooled_truth[i],pooled_pred[i])
            for i,name in enumerate(('valence','arousal'))}
    result={'seeds':seeds,'songs':len(table),
            'selected_epochs':[x['selected_epoch'] for x in summaries],
            'single_model_mean':{k:float(np.mean([x['validation_metrics'][k] for x in summaries])) for k in keys},
            'single_model_sd':{k:float(np.std([x['validation_metrics'][k] for x in summaries],ddof=1)) for k in keys},
            'ensemble_metrics':{k:float(table[k].mean()) for k in keys},
            'pooled_metrics':pooled,
            'split':'902 train / 270 development validation','test_set_used':False,
            'counterfactual_training':False}
    (root/'ensemble_summary.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--seeds',nargs='+',type=int,default=list(range(2026,2031)))
    a=p.parse_args();print(json.dumps(report(a.root,a.seeds),indent=2))
