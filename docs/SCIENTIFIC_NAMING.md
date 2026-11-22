# Scientific naming for manuscripts and result tables

Use these display names in manuscript prose, tables, captions, and plots:

| Display name | Archived alias |
|---|---|
| MUSE-DU with point-only supervision | `B2_point` |
| MUSE-DU with temporal-shape supervision | `B2` |
| MUSE-DU with calibrated level–trajectory fusion | `B3-C level-only` |
| MERT-v1-330M probe | `MERT-330M` |
| MERT-v1-95M probe | `MERT probe` in the original baseline comparison |

The 330M and 95M probes use different pretrained encoders and have separate
results. Keep the archived aliases in code paths, CLI flags, manifests, and
saved files so reruns remain traceable. A result table should identify the
exact model configuration and evaluation protocol whenever it compares
historical and new experiments.

The calibrated level–trajectory fusion changes the predicted song mean while
keeping the centered trajectory from MUSE-DU with temporal-shape supervision.
It therefore inherits that model's within-song Pearson correlation; an
unchanged value is not an additional temporal-tracking improvement.
