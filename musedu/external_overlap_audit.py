"""Audit exact decoded-audio SHA256 overlap across prepared datasets."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def hashes(path):
    with open(path,newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    result={}
    for row in rows:
        with np.load(row['file'],allow_pickle=False) as archive:
            result[row['song_id']]=str(archive['audio_sha256'].item())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deam',required=True)
    p.add_argument('--pmemo',required=True)
    p.add_argument('--merp',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    data={name:hashes(getattr(a,name)) for name in ('deam','pmemo','merp')}
    source=set(data['deam'].values())
    report={'counts':{name:len(rows) for name,rows in data.items()},
            'DEAM_PMemo_exact_audio_overlap':len(source&set(data['pmemo'].values())),
            'DEAM_MERP_exact_audio_overlap':len(source&set(data['merp'].values())),
            'MERP_known_anchor_ids':[sid for sid in data['merp'] if sid.startswith('deam_')]}
    Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
