"""Collect paper code and saved results into one portable review folder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / 'outputs'
EXTERNAL_RESULTS = {
    'deam_cv5_stage1': Path(r'E:\deam_cv5_stage1'),
    'deam_cv5_stage1_adapted40': Path(r'E:\deam_cv5_stage1_adapted40'),
    'deam_cv5_paper_ablations': Path(r'E:\deam_cv5_paper_ablations'),
    'deam_pmemo_frozen_transfer': Path(r'E:\deam_pmemo_frozen_transfer'),
    'deam_merp_frozen_transfer': Path(r'E:\deam_merp_frozen_transfer'),
    'deam_frozen_baseline_transfer': Path(r'E:\deam_frozen_baseline_transfer'),
}
LOCAL_RESULTS = (
    'paper_figures',
    'MUSE_DU_Minimal_Dataset',
    'deam_explanation_validation',
    'deam_uncertainty_efficiency',
    'deam_primary_uncertainty_ablations',
    'deam_primary_musedu',
    'deam_temporal_diagnostics',
)
PROVENANCE = {
    'outputs/deam_mert': ('deam_manifest.csv', 'deam_audit.json',
                           'full_feature_audit.json', 'source_identity_audit.json'),
    'outputs/pmemo_mert': ('pmemo_manifest.csv', 'pmemo_audit.json',
                            'feature_provenance.json'),
    'outputs/merp_mert': ('merp_manifest.csv', 'feature_provenance.json'),
}


def copy_code(source, dest):
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '.pytest_cache', '*.pyo'))


def file_stats(folder):
    files=[p for p in folder.rglob('*') if p.is_file()]
    return {'files':len(files),'bytes':sum(p.stat().st_size for p in files)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    if out.exists():raise FileExistsError(f'Refusing existing package: {out}')
    out.mkdir(parents=True)
    code=out/'code';code.mkdir()
    copy_code(PROJECT/'musedu',code/'musedu')
    copy_code(PROJECT/'tests',code/'tests')
    patch=PROJECT/'MUSE_DU_Published_Baselines_Diagnostic_Patch'
    if patch.is_dir():copy_code(patch,code/patch.name)
    for name in ('pyproject.toml','README.md','VERIFICATION.md'):
        shutil.copy2(PROJECT/name,code/name)
    results=out/'results';results.mkdir()
    for name,source in EXTERNAL_RESULTS.items():
        if not source.is_dir():raise FileNotFoundError(source)
        shutil.copytree(source,results/name)
    for name in LOCAL_RESULTS:
        source=OUTPUTS/name
        if source.is_dir():shutil.copytree(source,results/name)
    for name in ('deam_frozen_transfer_report.md','evaluation_summary.md',
                 'MUSE_DU_Minimal_Dataset.zip'):
        source=OUTPUTS/name
        if source.is_file():shutil.copy2(source,results/name)
    provenance=out/'provenance';provenance.mkdir()
    for source_dir,names in PROVENANCE.items():
        source=PROJECT/source_dir
        target=provenance/Path(source_dir).name;target.mkdir()
        for name in names:
            file=source/name
            if file.is_file():shutil.copy2(file,target/name)
    readme='''MUSE-DU paper reproducibility bundle
====================================

Contents
--------
code/ contains the project Python modules, tests, baseline diagnostic patch,
pyproject.toml, and project documentation.

results/ contains the full saved five-fold outer predictions and metrics,
adapted ridge stack, shape-weight and component ablations, frozen DEAM-to-
PMEmo/MERP transfers, paired bootstrap reports, paper figures, and the six
shareable minimal-dataset files. The older explainability and uncertainty
diagnostics are kept in separately named directories because they used a
different checkpoint. The deam_primary_musedu directory holds that older
checkpoint and its split/provenance metadata.

provenance/ contains DEAM, PMEmo, and MERP feature manifests and audits.
Some original manifests contain absolute paths from the study computer;
update these paths before attempting to retrain on another computer.

Fast verification
-----------------
Run from this bundle's code directory after installing the dependencies in
pyproject.toml and the optional MERT/baseline dependencies:

  python -m musedu.verify_minimal_dataset --folder ../results/MUSE_DU_Minimal_Dataset

The verification script automatically uses the copied results/ directories
when run from this bundle. The CSV metric definitions are documented in
results/MUSE_DU_Minimal_Dataset/README.txt. Saved predictions and labels can
be analyzed without rerunning audio feature extraction or neural training.

Exclusions
----------
Raw DEAM, PMEmo, and MERP audio/annotation source files and the large cached
MERT/OpenSMILE/Librosa feature arrays are not copied. The source datasets
remain subject to their original distribution terms. The DEAM feature cache
alone is several hundred MB, and the broader baseline feature cache is over
12 GB. This bundle supports independent recalculation of the main reported
metrics from saved predictions. Full retraining additionally requires source
audio, annotations, and regenerated or original feature caches.

The adapted OpenSMILE-Librosa Ridge stack is not the original published
Mazzola Hybrid Ridge, which requires the unavailable PAA feature family.
PMEmo/MERP had earlier workspace analyses; describe the frozen transfer as
cross-dataset evaluation, not a wholly untouched independent test.
'''
    (out/'README.txt').write_text(readme,encoding='utf-8')
    stats={name:file_stats(out/name) for name in ('code','results','provenance')}
    (out/'bundle_manifest.json').write_text(json.dumps(stats,indent=2))
    zip_path=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6,allowZip64=True) as z:
        for file in sorted(p for p in out.rglob('*') if p.is_file()):
            z.write(file,arcname=file.relative_to(out).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Corrupt ZIP member: {bad}')
    print(json.dumps({'folder':str(out),'zip':str(zip_path),
                      'zip_bytes':zip_path.stat().st_size,'sections':stats},indent=2))

if __name__=='__main__':main()
