"""Build the matched T0-T3 five-model ensemble comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .trace_report import report


def main(t0,t1,t2,t3,out):
    roots={'T0_B2':Path(t0),'T1_no_cue_sufficiency':Path(t1),
           'T2_no_cues':Path(t2),'T3_cue_sufficiency':Path(t3)}
    results={name:report(root) for name,root in roots.items()}
    rows=[]
    for name,result in results.items():
        rows.append({'model':name,**result['ensemble_metrics'],**result['pooled_metrics'],
                     'selected_epochs':','.join(map(str,result['selected_epochs']))})
    destination=Path(out);destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(destination/'matched_ensemble_results.csv',index=False)
    rng=np.random.default_rng(20261026);paired=[]
    metrics=('valence_mae','arousal_mae','valence_within_pearson','arousal_within_pearson')
    tables={name:pd.read_csv(root/'ensemble_validation_by_song.csv',dtype={'song_id':str}).set_index('song_id')
            for name,root in roots.items()}
    for candidate,reference in (('T3_cue_sufficiency','T0_B2'),
                                ('T3_cue_sufficiency','T1_no_cue_sufficiency'),
                                ('T3_cue_sufficiency','T2_no_cues')):
        ids=tables[candidate].index.intersection(tables[reference].index)
        for metric in metrics:
            difference=(tables[candidate].loc[ids,metric]-tables[reference].loc[ids,metric]).to_numpy()
            samples=np.nanmean(difference[rng.integers(0,len(difference),(10000,len(difference)))],axis=1)
            paired.append({'candidate':candidate,'reference':reference,'metric':metric,
                           'mean_difference':float(np.nanmean(difference)),
                           'ci95_low':float(np.nanquantile(samples,.025)),
                           'ci95_high':float(np.nanquantile(samples,.975))})
    pd.DataFrame(paired).to_csv(destination/'paired_bootstrap.csv',index=False)
    payload={'protocol':{'train_songs':902,'validation_songs':270,'seeds':list(range(2026,2031)),
                         'selector':'equal-song Huber + .02 stable Pearson loss',
                         'ccc_note':'*_ccc is equal-song mean; *_pooled_ccc concatenates all valid frames',
                         'bootstrap':'10,000 song-paired resamples; exploratory reused validation set',
                         'test_set_used':False},'results':results,'paired_comparisons':paired}
    (destination/'matched_ensemble_results.json').write_text(json.dumps(payload,indent=2))
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--t0',required=True);p.add_argument('--t1',required=True)
    p.add_argument('--t2',required=True);p.add_argument('--t3',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(main(a.t0,a.t1,a.t2,a.t3,a.out),indent=2))
