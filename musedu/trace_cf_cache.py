"""Recompute MERT and acoustic features for accepted TRACE interventions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import load_manifest,read_npz
from .features import _audio,_bin_features,MERT_REVISION
from .trace_interventions import intervene
from .valence_features import overlap_layers,harmonic_descriptors,LAYERS


def build(manifest,measurements,out,device='cuda'):
    df=load_manifest(manifest);lookup=df.set_index('song_id');audit=pd.read_csv(measurements,dtype={'song_id':str})
    audit=audit[audit.accepted.astype(bool)].reset_index(drop=True)
    root=Path(out);root.mkdir(parents=True,exist_ok=True);records=[]
    for number,row in audit.iterrows():
        sid=str(row.song_id);source=lookup.loc[sid];original=read_npz(source.file)
        # Keep the evaluation component short enough for legacy Windows MAX_PATH.
        folder='ev' if row.partition=='evaluation' else 'train'
        target=root/'features'/folder/f'{sid}_{row.target_group}.npz';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.is_file():
            waveform=_audio(source.audio,24000)
            altered=intervene(waveform,24000,float(row.start_seconds),float(row.end_seconds),row.target_group)
            times=original['context_times']
            layers=overlap_layers(altered,24000,times,device)
            acoustic=_bin_features(altered,24000,times,2.)
            harmonic=harmonic_descriptors(acoustic)
            index=np.searchsorted(times,original['times'])
            if not np.allclose(times[index],original['times'],atol=1e-4):raise ValueError('Timestamp mismatch')
            region=(original['times']>=row.start_seconds)&(original['times']<row.end_seconds)
            data={**original,'acoustic':acoustic[index],'embedding':layers[index,-1],
                  'embedding_layers':layers[index],'harmonic':harmonic[index],
                  'context_layers':layers,'context_acoustic':acoustic,'context_harmonic':harmonic,
                  'intervention_region':region,'intervention_group':np.array(row.target_group),
                  'source_audio_sha256':original['audio_sha256'],
                  'audio_sha256':np.array(hashlib.sha256(np.asarray(altered,dtype='<f4').tobytes()).hexdigest()),
                  'feature_version':np.array('trace_cf_overlap_bin_v1')}
            temporary=target.with_suffix('.partial.npz');np.savez_compressed(temporary,**data);temporary.replace(target)
        records.append({'partition':row.partition,'song_id':sid,'target_group':row.target_group,
                        'start_seconds':row.start_seconds,'end_seconds':row.end_seconds,
                        'original_file':str(Path(source.file).resolve()),'modified_file':str(target.resolve())})
        print(f'counterfactual cache {number+1}/{len(audit)} {row.partition} {sid} {row.target_group}',flush=True)
    pd.DataFrame(records).to_csv(root/'intervention_manifest.csv',index=False)
    protocol={'accepted_interventions':len(records),'train':sum(r['partition']=='train' for r in records),
              'evaluation':sum(r['partition']=='evaluation' for r in records),
              'mert_layers':LAYERS,'mert_revision':MERT_REVISION,'mert_recomputed':True,
              'acoustic_recomputed':True,'waveform_saved':False,
              'source_measurements':str(Path(measurements).resolve())}
    (root/'cache_protocol.json').write_text(json.dumps(protocol,indent=2));return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--measurements',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();print(json.dumps(build(a.manifest,a.measurements,a.out,a.device),indent=2))
