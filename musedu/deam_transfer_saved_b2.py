"""Apply frozen DEAM-trained B2 checkpoints to an external dataset."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from .matched_development import make_model,output_mean
from .published_baselines.io import save_report,result_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external-manifest',required=True)
    p.add_argument('--trained-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    with open(a.external_manifest,newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    names={r['dataset'] for r in rows}
    if len(names)!=1 or 'deam' in names:raise ValueError('Expected one external dataset')
    dataset=names.pop()
    excluded=[r['song_id'] for r in rows if dataset=='merp' and r['song_id'].startswith('deam_')]
    rows=[r for r in rows if r['song_id'] not in excluded]
    index={r['song_id']:r['file'] for r in rows}
    ids=list(index)
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    for name in ('B2_temporal_shape','B2_point_only'):
        dest=root/name
        if (dest/'validation_metrics.json').is_file():
            print(f'SKIP {name}',flush=True);continue
        # These checkpoints were written by deam_transfer_b2 in this workspace.
        state=torch.load(Path(a.trained_root)/name/'model.pt',map_location='cpu',weights_only=False)
        mean=state['normalizer_mean'];std=state['normalizer_std']
        model=make_model('v1',len(mean),96).to(a.device)
        model.load_state_dict(state['state_dict']);model.eval()
        predictions={}
        with torch.inference_mode():
            for sid in ids:
                with np.load(index[sid],allow_pickle=False) as z:
                    times=z['times'];x=np.concatenate((z['acoustic'],z['embedding']),axis=-1)
                if len(times)>1 and not np.allclose(np.diff(times),.5,atol=1e-3):
                    raise ValueError(f'{sid}: bad timestamps')
                x=(x-mean)/std
                xt=torch.as_tensor(x[None],dtype=torch.float32,device=a.device)
                tt=torch.as_tensor(times[None],dtype=torch.float32,device=a.device)
                valid=torch.ones((1,len(times)),dtype=torch.bool,device=a.device)
                predictions[sid]=output_mean(model(xt,tt,valid))[0].cpu().numpy().astype(np.float32)
        result_root(dest)
        metadata={'method':name,'source_dataset':'DEAM','external_dataset':dataset,
                  'source_checkpoint':str(Path(a.trained_root)/name/'model.pt'),
                  'excluded_known_deam_merp_anchor_ids':excluded,
                  'external_labels_used_for_training_or_selection':False}
        metrics=save_report(dest,predictions,index,ids,metadata)
        print(f'DONE {name} {dataset} songs={len(ids)} MED={metrics["med"]:.6f}',flush=True)
    (root/'protocol.json').write_text(json.dumps({'source':'frozen DEAM checkpoints',
        'external_dataset':dataset,'evaluated_songs':len(ids),
        'excluded_known_deam_merp_anchor_ids':excluded},indent=2))


if __name__=='__main__':main()
