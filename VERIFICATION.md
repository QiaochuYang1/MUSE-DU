# MUSE-DU implementation verification (October 2026)

The repaired branch contains the following test gates:

| Gate | Automated tests | Limitations |
|---|---|---|
| Annotation adapters | Synthetic DEAM/PMEmo CSV and mocked MERP Parquet | Actual downloaded archives still require audit |
| Time resolution | 1 Hz, 2 Hz and missing-label-gap adjacency | Nonstandard real annotation releases must be checked |
| Model | Forward/backprop, masking, residual-scale optimizer | Actual five-seed DEAM performance not yet measured |
| Music embeddings | MERT-alignment interface with mocked processor/model | Pretrained weights, GPU device, network and license not verified |
| Data independence | Rejects reused corpus and identical decoded-audio SHA-256 | Remaster/codec/cropping/perceptual duplicates need independent audit |
| Conformal | Finite-sample quantile and interval calculations | 90% coverage under dataset shift is not guaranteed |
| Pipeline | Independent synthetic audio from training to external inference | No performance claims for DEAM/PMEmo/MERP |

Run `python -m pytest -q` and `python -m compileall -q musedu tests` after installing the project. GitHub Actions runs the synthetic/unit suite on every branch push.

**Code vs paper:** The actual train/validation/scale/calibration/test division is 50/15/10/10/15 (five disjoint partitions), not the older paper's four-way preliminary allocation. All evaluation windows have finite context (`--window`, default 120 bins). Explainability's quantitative stability/fidelity/attribution baselines, five-seed real-data experiments, and external corpus and audio audits are future experimental work.
