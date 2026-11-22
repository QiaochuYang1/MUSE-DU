"""Add source data and prepared features to the paper reproducibility bundle."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import zipfile

PROJECT=Path(__file__).resolve().parents[1]
DEAM_SOURCE=Path(r'C:\Users\Asad khan\.cache\kagglehub\datasets\imsparsh\deam-mediaeval-dataset-emotional-analysis-in-music\versions\1')
DATASETS=(
    ('source/deam_mirror',DEAM_SOURCE),
    ('source/pmemo',PROJECT/'data'/'pmemo'),
    ('source/merp',PROJECT/'data'/'merp'),
    ('prepared/deam_mert',PROJECT/'outputs'/'deam_mert'),
    ('prepared/pmemo_mert',PROJECT/'outputs'/'pmemo_mert'),
    ('prepared/merp_mert',PROJECT/'outputs'/'merp_mert'),
    ('prepared/deam_published_feature_cache',PROJECT/'outputs'/'deam_published_feature_cache'),
)

def extended(path):
    """Permit deep dataset paths on Windows without shortening source names."""
    return Path('\\\\?\\'+str(path)) if os.name=='nt' else path

def copy_resumable(source,dest):
    if not source.is_dir():raise FileNotFoundError(source)
    n=0;size=0
    for item in source.rglob('*'):
        if not item.is_file():continue
        relative=item.relative_to(source)
        target=dest/relative
        long_target=extended(target)
        long_target.parent.mkdir(parents=True,exist_ok=True)
        if not long_target.is_file() or long_target.stat().st_size!=item.stat().st_size:
            shutil.copy2(extended(item),long_target)
        n+=1;size+=item.stat().st_size
    return {'files':n,'bytes':size}

def manifest(bundle,dataset,source):
    original=PROJECT/'outputs'/f'{dataset}_mert'/f'{dataset}_manifest.csv'
    if not original.is_file():raise FileNotFoundError(original)
    target=bundle/'data'/'manifests'/f'{dataset}_manifest_portable.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    with original.open(newline='',encoding='utf-8') as inp,target.open('w',newline='',encoding='utf-8') as out:
        reader=csv.DictReader(inp);writer=csv.DictWriter(out,fieldnames=reader.fieldnames)
        writer.writeheader();count=0
        for row in reader:
            sid=row['song_id']
            row['file']=f'../data/prepared/{dataset}_mert/features/{dataset}/{sid}.npz'
            audio=Path(row['audio'])
            relative=audio.relative_to(source)
            source_name='deam_mirror' if dataset=='deam' else dataset
            row['audio']=(Path('../data/source')/source_name/relative).as_posix()
            writer.writerow(row);count+=1
    return count

def stats(folder):
    files=[x for x in extended(folder).rglob('*') if x.is_file()]
    return {'files':len(files),'bytes':sum(x.stat().st_size for x in files)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    a=parser.parse_args();bundle=a.bundle.resolve()
    if not (bundle/'code'/'musedu').is_dir() or not (bundle/'results'/'MUSE_DU_Minimal_Dataset').is_dir():
        raise ValueError('Expected completed paper bundle with code and results')
    dest=bundle/'data';dest.mkdir(exist_ok=True)
    copied={}
    for name,source in DATASETS:
        print('COPY',name,flush=True)
        copied[name]=copy_resumable(source,dest/name)
        print('DONE',name,copied[name],flush=True)
    counts={dataset:manifest(bundle,dataset,source) for dataset,source in
            (('deam',DEAM_SOURCE),('pmemo',PROJECT/'data'/'pmemo'),
             ('merp',PROJECT/'data'/'merp'))}
    shutil.copy2(Path(__file__),bundle/'code'/'musedu'/Path(__file__).name)
    readme=(bundle/'README.txt').read_text(encoding='utf-8')
    before=readme.index('Exclusions\n----------')
    after=readme.index('The adapted OpenSMILE-Librosa Ridge stack',before)
    readme=readme[:before]+'''Included data
-------------
data/source/deam_mirror contains the DEAM mirror audio and annotations used
in this study. data/source/pmemo and data/source/merp contain the locally
prepared official PMEmo and MERP source directories, including audio and
annotations. data/prepared contains the aligned MERT/acoustic archives for
all three datasets and the OpenSMILE, Librosa, mel, and MERT baseline caches.
The data/manifests/*_portable.csv files use paths relative to code/ so that
training commands can use them from the bundle's code directory. The
original manifests are preserved in provenance/ and prepared/.

Redistribution of source datasets remains subject to their original terms.
The DEAM source was obtained from the imsparsh Kaggle mirror, version 1;
official checksums for all 1,802 DEAM recordings were unavailable. The
provided copies allow local retraining from the archived prepared features.

'''+readme[after:]
    (bundle/'README.txt').write_text(readme,encoding='utf-8')
    summary={name:stats(bundle/name) for name in ('code','results','provenance','data')}
    (bundle/'bundle_manifest.json').write_text(json.dumps(
        {'sections':summary,'data_sources':copied,'portable_manifest_songs':counts},indent=2))
    zip_path=bundle.parent/(bundle.name+'.zip')
    temp=zip_path.with_suffix('.tmp.zip')
    print('ZIP START',flush=True)
    long_bundle=extended(bundle)
    with zipfile.ZipFile(temp,'w',allowZip64=True) as z:
        for file in sorted(x for x in long_bundle.rglob('*') if x.is_file()):
            suffix=file.suffix.lower()
            compression=(zipfile.ZIP_STORED if suffix in
                         {'.npz','.mp3','.zip','.pt','.parquet','.pdf','.png'}
                         else zipfile.ZIP_DEFLATED)
            z.write(file,arcname=file.relative_to(long_bundle).as_posix(),
                    compress_type=compression,compresslevel=1)
    with zipfile.ZipFile(temp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Corrupt ZIP member: {bad}')
        members=len(z.namelist())
    os.replace(temp,zip_path)
    print(json.dumps({'bundle':str(bundle),'zip':str(zip_path),
                      'zip_bytes':zip_path.stat().st_size,'zip_members':members,
                      'sections':summary,'portable_manifest_songs':counts},indent=2))

if __name__=='__main__':main()
