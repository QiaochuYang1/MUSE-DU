"""Incrementally extract frozen MERT-330M layers at external annotation times."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .published_baselines.features import decode_audio,pretrained_layer_features
from .plan_matrix_2x2 import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model',required=True,help='Pinned local MERT-330M snapshot directory')
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    with open(a.manifest,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len({r['song_id'] for r in rows})!=len(rows):raise AssertionError('Duplicate external song ID')
    protocol={'manifest_sha256':sha(a.manifest),'dataset':rows[0]['dataset'],
              'songs':len(rows),'model_snapshot':str(Path(a.model).resolve()),
              'model_weights_sha256':'0c03774d821a6ad972cb48220b128a10ebea2c790116bd0fd35e5a013f8017f6',
              'encoder_sample_rate':24000,'chunk_seconds':5.,
              'alignment':'interpolate frozen layer frames to original external annotation timestamps',
              'external_labels_used_for_extraction':False}
    file=out/'protocol.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('Feature protocol changed')
    file.write_text(json.dumps(protocol,indent=2))
    if a.limit:rows=rows[:a.limit]
    for i,row in enumerate(rows,1):
        sid=row['song_id'];dest=out/f'{sid}.npz'
        with np.load(row['file'],allow_pickle=False) as z:times=z['times']
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                if np.allclose(z['times'],times,atol=1e-3) and z['features'].shape==(len(times),24,1024):
                    continue
            raise ValueError(f'Existing feature archive invalid: {dest}')
        audio,sr=decode_audio(row['audio'],target_sr=44100,decoder='soundfile')
        feature=pretrained_layer_features(audio,sr,times,'mert',a.model,device=a.device,chunk_s=5.)
        if feature.shape!=(len(times),24,1024) or not np.isfinite(feature).all():
            raise ValueError(f'{sid}: invalid MERT features')
        np.savez_compressed(dest,times=times,features=feature)
        if i%10==0 or i==len(rows):print(f'EXTRACT {protocol["dataset"]} {i}/{len(rows)}',flush=True)

if __name__=='__main__':main()
