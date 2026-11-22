# Understanding Emotional Dynamics in Music: MUSE-DU

Code for the song-disjoint DEAM experiments, published-method baseline
adaptations, temporal-shape ablations, cross-dataset PMEmo/MERP evaluation,
and calibrated level–trajectory fusion. This folder is intended as the
**code-only GitHub repository**. It contains no recordings, extracted feature
caches, model checkpoints, or prediction results.

## Scientific names used in the manuscript

| Final manuscript name | Historical code or result alias | Meaning |
|---|---|---|
| **MUSE-DU with point-only supervision** | `B2_point`, `point_only`, `train_point` | B2 architecture trained without temporal-shape loss |
| **MUSE-DU with temporal-shape supervision** | `B2`, `train_shape` | B2 architecture trained with temporal-shape loss |
| **MUSE-DU with calibrated level–trajectory fusion** | `B3-C level-only`, `B3_level_only` | MERT-informed song-level prediction mean with the centered B2 trajectory preserved |
| **MERT-v1-330M probe** | `MERT-330M`, `mert_330m_2026_adapted` | Frozen 330M encoder with a trained downstream probe |
| **MERT-v1-95M probe** | `MERT probe`, `mert_probe` in the original baseline table | Earlier frozen 95M representation probe |

The historical aliases remain in filenames, CLI arguments, and saved
manifests so archived experiments can be reproduced. Use the **final
manuscript names** in scientific prose, tables, and figure labels. The two
MERT probes are distinct experiments; do not combine or relabel their scores.
See `docs/SCIENTIFIC_NAMING.md` for the caption and reporting convention.

## Code map

| Location | Contents |
|---|---|
| `musedu/` | Core preprocessing, models, training, metrics, uncertainty, and explainability code from the original work |
| `musedu/published_baselines/` | OpenSMILE, Librosa, Conformer, MERT, and adapted stacking baseline code |
| `musedu/deam_stage1_cv.py` and related CV modules | Original five-fold DEAM comparisons and temporal-shape ablation |
| `musedu/plan_*.py` | Full-budget 2 × 2 objective/checkpoint study, 2026 MERT-v1-330M probe, reporting, and packaging |
| `musedu/b3_*.py` | Three-seed nested fusion, calibration-matched controls, external transfer, interval audits, and reproducibility checks |
| `tests/` | Synthetic and protocol tests for the original and new experiments |
| `docs/plan2026_run_instructions.md` | Commands for the archived additional experiments |
| `MUSE_DU_Published_Baselines_Diagnostic_Patch/` | Baseline dependency and feature-extraction diagnostic notes |

The final fusion is

\[
\hat y_{it}^{\mathrm{fusion}} =
(1-\alpha)\bar y_i^{\mathrm{shape}}+
\alpha\bar y_i^{\mathrm{MERT330}}+
(\hat y_{it}^{\mathrm{shape}}-\bar y_i^{\mathrm{shape}}).
\]

The fusion changes a song's predicted level and keeps the centered
MUSE-DU trajectory. Its within-song Pearson correlation therefore equals
that of **MUSE-DU with temporal-shape supervision** by construction. The
calibration-matched analysis compares it with B2-only and MERT-only level
calibrators fitted on the same internal songs.

## Setup

Python 3.10 or newer is required. Install a PyTorch build suitable for the
machine, then install this package and the optional experiment dependencies:

```bash
python -m pip install -e ".[test,baselines,merp,mert]"
python -m pip install -r MUSE_DU_Published_Baselines_Diagnostic_Patch/requirements-extra.txt
python -m pytest -q
```

The baseline diagnostics can be checked with:

```bash
python -m musedu.published_baselines.diagnose --help
```

Download DEAM, PMEmo, and MERP from their respective dataset maintainers
and prepare their manifests and 0.5-second aligned feature archives before
running empirical experiments. Raw audio and pretrained weights are not
included here. The baseline feature cache and frozen MERT embeddings are
also external inputs. See `docs/plan2026_run_instructions.md` for the
commands and `musedu/published_baselines/` for feature requirements. The
local reproducibility bundle contains the archived processed data, fold
assignments, predictions, and results; it is **separate from this GitHub
code folder** because dataset redistribution terms must be respected.

## Experiment scope and interpretation

- The original comparison uses five song-disjoint outer DEAM folds over
  1,802 recordings. Its baseline adaptations share the outer folds.
- The final calibrated fusion has three complete five-fold downstream-model
  refits. For each fold, fitting, checkpoint selection, fusion calibration,
  and outer evaluation use disjoint songs.
- The final calibration controls use the same 144 internal calibration songs
  per fold and seed as the fusion. Fold-stratified paired song bootstraps
  use 10,000 draws and keep the three repeats separate.
- PMEmo and MERP transfer uses frozen DEAM-only models. Four known DEAM
  overlaps are excluded from MERP. The datasets informed earlier
  development, so these are development-informed evaluations rather than
  untouched independent tests.
- Shape-loss weight 0.05 and the earlier B3 calibration pilot are
  exploratory. The adapted OpenSMILE–Librosa Ridge stack is not the
  published three-feature-family Hybrid Ridge because the required PAA
  inputs were unavailable.

The public README uses manuscript terminology; historical run names in
scripts and archived output remain unchanged for reproducibility.
