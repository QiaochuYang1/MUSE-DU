"""Extract frozen external OpenSMILE/Librosa descriptors at annotation timestamps."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from .published_baselines.features import decode_audio,frame_features_librosa,opensmile_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--features-dir',required=True)
    p.add_argument('--views',nargs='+',choices=('librosa','opensmile'),default=['librosa','opensmile'])
    a=p.parse_args()
    with open(a.manifest,newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    dataset={r['dataset'] for r in rows}
    if len(dataset)!=1 or 'deam' in dataset:raise ValueError('Expected one external dataset')
    prefix=dataset.pop()+'_'
    root=Path(a.features_dir)
    for i,row in enumerate(rows,1):
        sid=prefix+row['song_id']
        needed=[v for v in a.views if not (root/v/(sid+'.npz')).is_file()]
        if not needed:continue
        with np.load(row['file'],allow_pickle=False) as z:times=z['times']
        audio,sr=decode_audio(row['audio'])
        for view in needed:
            features=(frame_features_librosa(audio,sr,times) if view=='librosa'
                      else opensmile_features(audio,sr,times))
            if len(features)!=len(times) or not np.isfinite(features).all():
                raise ValueError(f'{sid}: invalid {view} features')
            dest=root/view/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,times=times,features=features)
        print(f'DONE {i}/{len(rows)} {sid} {needed}',flush=True)


if __name__=='__main__':main()
