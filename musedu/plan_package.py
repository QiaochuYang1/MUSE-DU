"""Add the completed new experiment to the existing code/results/data bundle."""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

def totals(folder):
    absolute=folder.resolve()
    if os.name=='nt':absolute=Path('\\\\?\\'+str(absolute))
    files=[p for p in absolute.rglob('*') if p.is_file()]
    return {'files':len(files),'bytes':sum(p.stat().st_size for p in files)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True)
    p.add_argument('--bundle',required=True)
    p.add_argument('--source-plan',required=True)
    a=p.parse_args();results=Path(a.results).resolve();bundle=Path(a.bundle).resolve()
    repo=Path(__file__).resolve().parents[1]
    if not (bundle/'data'/'prepared'/'deam_mert').is_dir() or not (bundle/'data'/'prepared'/'deam_published_feature_cache'/'mert').is_dir():
        raise FileNotFoundError('Existing bundle is missing prepared DEAM and MERT data')
    if len(list((results/'per_frame_predictions').glob('*.parquet')))!=12:
        raise AssertionError('Matrix frame-level predictions incomplete')
    if any(not (results/'mert2026'/f'fold_{fold}'/'validation_metrics.json').is_file() for fold in range(1,6)):
        raise AssertionError('MERT-2026 five-fold baseline incomplete')
    if not (results/'manuscript_addendum.tex').is_file():
        raise AssertionError('Manuscript addendum not generated')
    shutil.copytree(results,bundle/'results'/'plan2026',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('figure_A_preview.png','figure_B_preview.png',
                                                  'b3_level_fusion'))
    for source in list((repo/'musedu').glob('plan_*.py'))+list((repo/'musedu').glob('b3_*.py')):
        shutil.copy2(source,bundle/'code'/'musedu'/source.name)
    shutil.copy2(repo/'musedu'/'published_baselines'/'run.py',
                 bundle/'code'/'musedu'/'published_baselines'/'run.py')
    shutil.copy2(repo/'tests'/'test_plan_matrix_protocol.py',
                 bundle/'code'/'tests'/'test_plan_matrix_protocol.py')
    for source in (repo/'tests').glob('test_b3_*.py'):
        shutil.copy2(source,bundle/'code'/'tests'/source.name)
    (bundle/'code'/'docs').mkdir(exist_ok=True)
    for source in (repo/'docs').glob('plan2026_*.md'):
        shutil.copy2(source,bundle/'code'/'docs'/source.name)
    shutil.copy2(a.source_plan,bundle/'provenance'/'MUSE_DU_Experimental_Implementation_Plan.md')
    readme=bundle/'README.txt'
    text=readme.read_text()
    addition=('\nAdditional 2026 experiments\n---------------------------\n'
              'results/plan2026 contains the completed full-budget 2x2 checkpoint\n'
              'diagnostic, three seed repeats, 2026 adapted MERT-330M comparison,\n'
              'genuine held-out predictions, confidence intervals, figures, and a\n'
              'LaTeX addendum. The corresponding commands and new code are under\n'
              'code/docs/plan2026_run_instructions.md and code/musedu/plan_*.py.\n')
    if 'Additional 2026 experiments' not in text:
        readme.write_text(text+addition)
        text=readme.read_text()
    b3_note=('\nB3-C fusion pilot\n-----------------\n'
             'results/plan2026/b3_level_fusion_v2 contains an exploratory\n'
             'level-trajectory calibration with genuine five-fold predictions\n'
             'for three B2 seed groups and paired bootstrap intervals. It reuses\n'
             'the internal checkpoint songs for calibration and a fixed MERT\n'
             'teacher, so it needs a strictly separated confirmation study.\n')
    if 'B3-C fusion pilot' not in text:
        readme.write_text(text+b3_note)
        text=readme.read_text()
    nested_note=('\nNested B3 confirmation\n----------------------\n'
                 'results/plan2026/B3_results.md summarizes disjoint fitting,\n'
                 'checkpoint, calibration, and outer-fold evaluation. The\n'
                 'level-only variant preserves B2 trajectory amplitude and\n'
                 'achieves the best balance of MED and within-song CCC.\n'
                 'Shape-weight-0.05 variants are explicitly exploratory.\n')
    if 'Nested B3 confirmation' not in text:
        readme.write_text(text+nested_note)
        text=readme.read_text()
    external_note=('\nB3 external transfer\n--------------------\n'
                   'results/plan2026/b3_external_features contains the aligned\n'
                   'frozen MERT-330M layer caches for PMEmo and MERP. The\n'
                   'b3_external_transfer directories contain five-model\n'
                   'DEAM-only ensemble predictions, metrics, and paired\n'
                   'bootstrap intervals. Four DEAM/MERP overlaps are excluded.\n'
                   'These corpora were seen during earlier development, so this\n'
                   'is cross-dataset evaluation, not a pristine independent test.\n')
    if 'B3 external transfer' not in text:
        readme.write_text(text+external_note)
        text=readme.read_text()
    three_seed_note=('\nB3 three-seed nested refit\n--------------------------\n'
                     'results/plan2026/b3_nested, b3_nested_repeat1, and\n'
                     'b3_nested_repeat2 contain three independent five-fold\n'
                     'downstream-model refits with disjoint fit, checkpoint,\n'
                     'calibration, and outer song groups. The paired level-only\n'
                     'outputs, three-seed summary CSVs, and manuscript-ready\n'
                     'B3_manuscript_text.tex are also in results/plan2026.\n'
                     'B3_results.md records the pooled-CCC tradeoff and the\n'
                     'development-informed evaluation limitation.\n')
    if 'B3 three-seed nested refit' not in text:
        readme.write_text(text+three_seed_note)
        text=readme.read_text()
    final_note=('\nB3 final calibration and interval audit\n'
                '---------------------------------------\n'
                'results/plan2026/B3_final_calibration_report.md documents\n'
                'B2-only and MERT-only calibration controls fitted on the\n'
                'same 144 songs per fold and seed as B3. The associated\n'
                'b3_final_calibration_audit directory contains predictions,\n'
                'fold-stratified paired intervals, and a passed independent\n'
                'reproducibility audit. B3_manuscript_text.tex includes the\n'
                'calibration analysis and unfavorable CCC results.\n')
    if 'B3 final calibration and interval audit' not in text:
        readme.write_text(text+final_note)
    manifest_file=bundle/'bundle_manifest.json'
    manifest=json.loads(manifest_file.read_text())
    for section in ('code','results','provenance','data'):
        manifest['sections'][section]=totals(bundle/section)
    manifest['additional_experiment']='results/plan2026'
    manifest_file.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['sections'],indent=2))

if __name__=='__main__':main()
