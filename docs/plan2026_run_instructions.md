# 2026 additional DEAM experiments

The source plan is `E:\Understanding Emotional Dynamics\MUSE_DU_Experimental_Implementation_Plan.md`. The original code and genuine DEAM predictions were recovered locally, so the plan's missing-code prerequisite does not apply to this workspace.

Run from this repository in the `pytorch` Conda environment. On this Windows host, set `KMP_DUPLICATE_LIB_OK=TRUE` before Python. Source data paths are supplied through the manifest; output paths below are examples.

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$python='C:\Users\Asad khan\anaconda3\envs\pytorch\python.exe'
& $python -m musedu.plan_audit --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --out ..\plan2026
& $python -m pytest tests/test_plan_matrix_protocol.py -q
& $python -m musedu.plan_matrix_2x2 --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --out ..\plan2026 --device cuda
& $python -m musedu.plan_matrix_report --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --root ..\plan2026
& $python -m musedu.plan_summary --root ..\plan2026
& $python -m musedu.plan_mert_probe_2026 --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --features outputs/deam_published_feature_cache --out ..\plan2026\mert2026 --device cuda
& $python -m musedu.plan_export --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --root ..\plan2026
& $python -m musedu.plan_manuscript --root ..\plan2026
& $python -m musedu.b3_level_fusion --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --matrix-root ..\plan2026 --mert-root ..\plan2026\mert2026 --features outputs/deam_published_feature_cache --out ..\plan2026\b3_level_fusion_v2 --device cuda
& $python -m musedu.b3_level_report --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --matrix-root ..\plan2026 --mert-root ..\plan2026\mert2026 --root ..\plan2026\b3_level_fusion_v2
```

The matrix runner saves all 25 epoch checkpoints for each of 30 optimizer fits, then selects both point and shape checkpoints from the *same* trajectory. It uses the original five outer folds and internal validation song IDs. A rerun skips fully scored fits. Results are kept separate from the original patience-five runs under `E:\deam_cv5_stage1`.

The 2026 MERT baseline is a **matched-protocol adaptation**, not an exact reproduction. It reuses timestamp-aligned 24-layer MERT-330M feature files. Published settings used where specified are Adam, learning rate 0.001, dropout 0.25, an effective batch size of 64 songs (four 16-song GPU microbatches), and patience 15. Its MLP width and activation, fit loss, and shared checkpoint selector are documented in `mert2026/protocol.json` because the paper did not specify those details. The original cached MERT per-song feature files do not record an encoder commit; the local Hugging Face snapshot is identified, but an exact per-file encoder revision cannot be guaranteed.

The `plan_export` script preserves every prior five-fold baseline result, including the earlier MERT probe and the separately completed adapted OpenSMILE–Librosa stack. It adds the 2026 MERT result only after all five new folds complete. The publisher's frame-weighted MED is excluded from this equal-song ranking.

The B3-C commands perform an **exploratory post-processing pilot**, not the full B3 architecture. They fit alpha and trajectory scale only on internal DEAM validation songs and score held-out outer songs. The same internal songs had already selected the base checkpoints; the strictly separated follow-up is documented below. The MERT teacher is fixed across the three B2 seed repeats. Only `b3_level_fusion_v2` is valid; the earlier `b3_level_fusion` directory is marked invalid because of a numerical-optimizer issue.

The stricter follow-up is reproducible with these commands from the project
repository. The first command retrains both base models using separate fit,
checkpoint, and calibration songs within each outer fold. The second command
checks all 1,802 outer predictions and computes paired intervals. The third
uses the same saved base models with trajectory scale fixed to one, preserving
B2's centered dynamics. The later shape-0.05 commands are exploratory because
that coefficient was chosen after inspecting previous DEAM results.

```powershell
& $python -m musedu.b3_nested_confirm --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --matrix-root ..\plan2026 --features outputs/deam_published_feature_cache --out ..\plan2026\b3_nested --device cuda
& $python -m musedu.b3_nested_report --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --root ..\plan2026\b3_nested
& $python -m musedu.b3_nested_level_only --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --nested-root ..\plan2026\b3_nested --out ..\plan2026\b3_nested_level_only
& $python -m musedu.b3_nested_shape05 --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --nested-root ..\plan2026\b3_nested --out ..\plan2026\b3_nested_shape05 --device cuda
& $python -m musedu.b3_nested_shape05_report --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --nested-root ..\plan2026\b3_nested --root ..\plan2026\b3_nested_shape05
& $python -m musedu.b3_nested_shape05_level_only --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --nested-root ..\plan2026\b3_nested --shape05-root ..\plan2026\b3_nested_shape05 --out ..\plan2026\b3_nested_shape05_level_only --device cuda
```

For the complete three-seed nested refit, the archived run above is repeat 0.
Repeat 1 and 2 use the same frozen outer folds and four-way song partitions,
but independently initialize and train **both** B2 and MERT. The model seed is
`2026 + fold + 1000 * repeat`. Each repeat has its own fitted level-only
calibrator on songs disjoint from checkpoint selection. Run each command after
the previous GPU training command has finished:

```powershell
foreach ($repeat in 1,2) {
  $nested="..\plan2026\b3_nested_repeat$repeat"
  $level="..\plan2026\b3_nested_repeat${repeat}_level_only"
  & $python -m musedu.b3_nested_confirm --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --matrix-root ..\plan2026 --features outputs/deam_published_feature_cache --out $nested --repeat $repeat --device cuda
  & $python -m musedu.b3_nested_report --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --root $nested
  & $python -m musedu.b3_nested_level_only --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --nested-root $nested --out $level
}
& $python -m musedu.b3_three_seed_report --plan-root ..\plan2026
```

The final calibration-matched controls use only archived predictions. They fit
B2-only and MERT-only song-level calibrators on the same 144 reserved songs
per fold and seed as B3, reproduce the archived B3 outputs, and calculate
10,000 paired song-bootstrap intervals stratified by original outer fold.
External PMEmo/MERP B3-versus-B2 MED intervals are independently recalculated
from saved prediction files. No additional neural training is performed.

```powershell
& $python -m musedu.b3_final_calibration_audit --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --plan-root ..\plan2026 --out ..\plan2026\b3_final_calibration_audit
& $python -m musedu.b3_external_interval_audit --plan-root ..\plan2026 --out ..\plan2026\b3_final_calibration_audit
& $python -m musedu.b3_final_repro_audit --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --plan-root ..\plan2026
& $python -m pytest tests/test_b3_final_calibration_audit.py tests/test_b3_level_fusion.py -q
```

For frozen external transfer, extract MERT-330M layers at the existing PMEmo
and MERP annotation timestamps, then run the saved five-model DEAM ensemble.
The `--model` argument is the local Hugging Face snapshot directory recorded
in `mert2026/protocol.json`. The output reports exclude four known DEAM/MERP
overlaps and use only DEAM-fitted calibration parameters.

```powershell
$mert330='C:\Users\Asad khan\.cache\huggingface\hub\models--m-a-p--MERT-v1-330M\snapshots\5240c2708a5acaee1007f43fb9735c7dcd0b78c9'
& $python -m musedu.b3_external_features --manifest outputs/pmemo_mert/pmemo_manifest.csv --out ..\plan2026\b3_external_features\pmemo --model $mert330 --device cuda
& $python -m musedu.b3_external_features --manifest outputs/merp_mert/merp_manifest.csv --out ..\plan2026\b3_external_features\merp --model $mert330 --device cuda
& $python -m musedu.b3_external_transfer --external-manifest outputs/pmemo_mert/pmemo_manifest.csv --features ..\plan2026\b3_external_features\pmemo --nested-root ..\plan2026\b3_nested --calibration-root ..\plan2026\b3_nested_level_only --out ..\plan2026\b3_external_transfer\pmemo --device cuda
& $python -m musedu.b3_external_transfer --external-manifest outputs/merp_mert/merp_manifest.csv --features ..\plan2026\b3_external_features\merp --nested-root ..\plan2026\b3_nested --calibration-root ..\plan2026\b3_nested_level_only --out ..\plan2026\b3_external_transfer\merp --device cuda
```

For a rerun from `MUSE_DU_Paper_Reproducibility_Bundle/code`, use `../data/manifests/deam_manifest_portable.csv` as `--manifest`, `../results/deam_primary_musedu` as `--reference-run`, and `../results/deam_cv5_stage1` as `--original-run`. The MERT command takes `../data/prepared/deam_published_feature_cache` as `--features`; the export command also takes `../results/deam_cv5_stage1_adapted40` as `--adapted-run`. Select a **new** output directory for reruns so the archived split and manifest checksums remain immutable. The source data and cached features are already in the bundle's `data/` directory.
