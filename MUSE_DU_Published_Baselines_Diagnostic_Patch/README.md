# MUSE-DU: Published-Method Baseline Add-on

**Status:** Source code, synthetic smoke tests only. No DEAM training has been run by this package. Do **not** copy numbers from Mazzola et al. (2026) into your new results table as your own reproduced baseline measurements.

This add-on integrates with the existing repository `Asadkhan401/Understanding-Emotional-Dynamics-in-Music`, branch `muse-du-implementation`. It introduces **new files only** under `musedu/published_baselines/`; it does not replace B2, existing preprocessing, or any existing baselines. Copy the `musedu/published_baselines/` directory into the corresponding location in your repository. The package does not use the original paper's weights; it is a transparent independent implementation of *paper-inspired* models.

**Reference:** Mazzola, G., Seidita, V., & La Cascia, M. (2026). *Benchmarking multi-output models for continuous music emotion prediction*. Neural Computing and Applications 38:664. https://doi.org/10.1007/s00521-026-12370-y

## Models provided

| CLI method | Inputs | Implementation |
|---|---|---|
| `opensmile_lstm` | OpenSMILE emobase 988 | 2-layer, 64-hidden LSTM with 0.2 dropout, MSE fit |
| `librosa_xgboost` | Librosa 77 | XGBoost 200 estimators, depth 6, lr 0.1 |
| `librosa_rf` | Librosa 77 | Random Forest 300 estimators, depth 20 |
| `opensmile_mlp` | OpenSMILE 988 | 128/64/32 ReLU MLP, 0.3 dropout |
| `hybrid_ridge` | OpenSMILE, Librosa, PAA | 12 late models + 4 early models; train-only three-fold out-of-fold Ridge stacking |
| `mert_probe` | Frozen **multi-layer MERT** | Learned softmax layer mixture + MLP probe. Default encoder MERT-v1-330M |
| `conformer` | 128-D 2 Hz log-Mel summary | 2-block compact Conformer-style temporal network |
| `hubert_probe` | Frozen multi-layer HuBERT | Learned softmax layer mixture + MLP probe |

All outputs predict valence and arousal in [-1,1]. Models are trained on the `train` split and evaluated on the `val` split from the existing B2 reference run; `scale`, `cal`, and `test` are **never used**. The old test was examined during model development and is not an untouched confirmatory set.

### Important differences from the publication

- Mazzola et al. used **5-fold** file-wise cross-validation (about 1,442 train songs per fold), whereas this code uses **902 train / 270 repeatedly used validation** songs to match the user's existing B2 experiments.
- This implementation uses song-balanced training and a **shared equal-song Huber + 0.02 stable-correlation checkpoint selector**. Published models typically use MSE, and they did not necessarily use this selection objective. This is a *controlled adaptation*, not an exact result reproduction.
- The provided Librosa feature implementation yields 77 dimensions from MFCC mean/std/range, chroma mean/std, spectral contrast and spectral/energy terms. It does not certify bitwise equivalence to Mazzola's 77-D extraction details.
- OpenSMILE uses Audeering `FeatureSet.emobase`, `FeatureLevel.Functionals` on 0.5-s segments and **fails fast** unless dimension=988. Different installations may yield different feature definitions; record your package version.
- The 20-D PAA acoustic input **cannot be fabricated**. To run hybrid Ridge, prepare MoSQITo/Zwicker-Fastl PAA features from the paper's documented recipe, verified independently; store one file `features/paa/<song_id>.npz` for every train/val song. Files must contain `times` (identical to manifest) and `features` [T,20]. PAA is optional for all other baselines.
- Conformer uses a compact 2-Hz log-Mel adaptation rather than a guaranteed exact replica of the publication's mel-spectrogram model. State this in the manuscript.
- Training MERT-330M on a frozen feature cache with all hidden layers is much more costly than the original 95M last-layer B2 cache. **Do not mislabel the 95M last-layer baseline as the published MERT-330M probe.**
- Hybrid Ridge uses three *training-only* OOF folds by default (change with `--stack-folds`), and 16 base learners (12 late+4 early). Checkpoint selection within OOF fits is on a separate subset of that fold's training songs. Neither the OOF training targets nor the validation targets are exposed to Ridge during its fitting.

## Installation

```bash
# From your existing repository's root:
pip install -e '.[baselines,mert]'
pip install opensmile>=2.4
# librosa, scikit-learn, torch, transformers are already project dependencies.
```

Use your existing 902/270 reference run, with `splits.json` and `source_manifest.json`, and a prepared DEAM MERT manifest containing cached `mean`, `mask`, `times`, `acoustic`, and `embedding` arrays.

```bash
export MANIFEST=outputs/deam_mert/deam_manifest.csv
export REF=outputs/deam_primary_musedu
export AUDIO=/absolute/path/to/DEAM/audio
export FEATURES=outputs/deam_published_feature_cache
```

### Step 1 — prepare independent feature families

```bash
python -m musedu.published_baselines.features \
 --manifest "$MANIFEST" --reference-run "$REF" --audio-root "$AUDIO" \
 --features-dir "$FEATURES" --views librosa opensmile mel --device cpu
```

For multi-layer pretrained probing (requires network access for pretrained models, and GPU strongly recommended):

```bash
python -m musedu.published_baselines.features \
 --manifest "$MANIFEST" --reference-run "$REF" --audio-root "$AUDIO" \
 --features-dir "$FEATURES" --views mert hubert --device cuda \
 --mert-model m-a-p/MERT-v1-330M --hubert-model facebook/hubert-base-ls960
```

**PAA for hybrid Ridge:** provide validated `[T,20]` MoSQITo features in `outputs/deam_published_feature_cache/paa/<song_id>.npz` with keys `times` and `features`. The program refuses to guess or substitute PAA descriptors. Then run hybrid Ridge in Step 2.

### Step 2 — train independent models on the existing B2 splits

```bash
python -m musedu.published_baselines.run \
 --manifest "$MANIFEST" --reference-run "$REF" \
 --features-dir "$FEATURES" --out outputs/published_b2_comparison \
 --methods opensmile_lstm librosa_xgboost librosa_rf opensmile_mlp mert_probe conformer hubert_probe \
 --seeds 2026 2027 2028 2029 2030 --epochs 15 --batch-size 8 --device cuda
```

Run Hybrid Ridge **after** adding PAA:

```bash
python -m musedu.published_baselines.run \
 --manifest "$MANIFEST" --reference-run "$REF" \
 --features-dir "$FEATURES" --out outputs/published_b2_hybrid \
 --methods hybrid_ridge --seeds 2026 --stack-folds 3 \
 --epochs 15 --batch-size 8 --device cuda
```

On Windows PowerShell, replace the `$MANIFEST`/`$REF`/`$AUDIO`/`$FEATURES` shell references with actual paths (or use `$env:MANIFEST`, etc.) and write each command on one line.

### Step 3 — outputs

Each method has individual-seed `validation_metrics.json`, `validation_by_song.csv`, and per-song predictions. Five-seed mean predictions are evaluated separately under `<method>/ensemble/`.

The consolidated `all_baselines_development.csv` contains: per-axis song-equal MAE/RMSE/CCC/within-song Pearson, pooled CCC, and joint Mean Euclidean Distance (MED). No external published results are copied into this CSV.

The reference paper's overall CCC is **not** numerically comparable without matching definitions and data splits. If baselines look weaker or stronger, report the actual fitted values rather than filtering out unfavorable comparisons.

## Tests

```bash
python -m pytest -q tests/test_published_baselines.py
```

This add-on is software for running the baseline study, not a validated faithful reproduction of the authors' exact model implementations or published performance scores.

**Training-budget note:** `--epochs 15` matches the prior B2 development cap. This is a preliminary matched-budget comparison, not proof that every baseline was adequately tuned. If their learning curves are still improving, run a prespecified expanded-budget experiment (e.g. 50 epochs) for all trainable neural models, including B2. Report both if available.

## Troubleshooting a first-song Librosa stall (October 2026 patch)

The original add-on called `import librosa` and `librosa.load` even for the `opensmile` view. The patched `features.py` now uses SoundFile+SoXR to decode by default, **without modifying the 77-dimensional Librosa feature formulas**. You may use `--decoder librosa` to compare the old decoder. If SoundFile cannot decode your audio, install a supported libsndfile version or use the Librosa decoder; do not silently skip audio files.

Before any full-batch extraction, run the self-contained diagnostic (PowerShell, from the repository root):

```powershell
$env:NUMBA_CACHE_DIR = "$env:USERPROFILE\numba_cache_musedu"
New-Item -ItemType Directory -Force $env:NUMBA_CACHE_DIR | Out-Null
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -u -m musedu.published_baselines.diagnose --cache-dir "$env:NUMBA_CACHE_DIR" --trace-after 45
```

Optionally include `--audio "C:\\path\\to\\first_song.mp3" --seconds 4` to test real audio. It prints `START` and `DONE` around individual Librosa calls; the last unfinished stage localizes the stall. The recurring faulthandler trace is diagnostic, **not a Python exception**.

Then test just the **first training song** before processing all 1172 train/validation songs:

```powershell
python -u -m musedu.published_baselines.features --manifest outputs/deam_mert/deam_manifest.csv --reference-run outputs/deam_primary_musedu --audio-root "C:\\path\\to\\DEAM\\audio" --features-dir outputs/deam_published_feature_cache --views librosa --limit-songs 1 --trace-after 45 --numba-cache-dir "$env:NUMBA_CACHE_DIR" --decoder soundfile --device cpu
```

If successful, rerun the same command **without `--limit-songs 1`**, optionally adding `opensmile mel` to `--views`. It skips all completed `.npz` feature files unless `--overwrite` is set. Keep the same decoder and feature-extraction versions across all songs. Do **not** set `NUMBA_DISABLE_JIT=1` for the full dataset: this is a debugging switch and can make some computations dramatically slower. Make sure the installed versions are compatible by running `python -m pip check`; do not repeatedly upgrade NumPy/SciPy/Numba/Librosa without first isolating the stalled operation.

If the doctor passes but the feature command stalls before `START kind=librosa`, examine manifest/provenance loading and audio decoding. If it stalls at `START chroma` or another feature step, the problem is feature-specific. If it reports `START import scipy` but no `DONE`, the environment's SciPy import needs attention separately. **This patch cannot guarantee a fix on the user's machine; it makes the failure reproducible and localizable.**
