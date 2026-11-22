"""Song-disjoint cross-validation for the frozen valence experiment configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .v2_experiment import _reference_split
from .experiment import load_manifest
from .valence_experiment import run


def make_folds(song_ids, n_folds=5, seed=2026):
    """Return deterministic, exhaustive, song-disjoint validation folds."""
    ids = np.asarray([str(s) for s in song_ids], dtype=object)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(ids) < n_folds:
        raise ValueError("n_folds cannot exceed the number of songs")
    if len(set(ids)) != len(ids):
        raise ValueError("song_ids contains duplicates")
    order = np.random.default_rng(seed).permutation(ids)
    return [list(map(str, fold)) for fold in np.array_split(order, n_folds)]


def aggregate(root, expected_ids):
    """Pool out-of-fold song metrics and summarize variation across folds."""
    root = Path(root)
    tables = []
    fold_rows = []
    for fold_dir in sorted(root.glob("fold*")):
        path = fold_dir / "validation_by_song.csv"
        if not path.is_file():
            continue
        table = pd.read_csv(path, dtype={"song_id": str})
        table.insert(0, "fold", int(fold_dir.name.removeprefix("fold")))
        tables.append(table)
        fold_rows.append({"fold": int(table.fold.iloc[0]), "songs": len(table),
                          **{c: float(table[c].mean()) for c in table.columns
                             if c not in ("fold", "song_id")}})
    if not tables:
        raise FileNotFoundError(f"No completed folds under {root}")
    oof = pd.concat(tables, ignore_index=True).sort_values(["fold", "song_id"])
    expected = set(map(str, expected_ids))
    observed = set(oof.song_id)
    duplicates = sorted(oof.loc[oof.song_id.duplicated(), "song_id"].unique())
    if duplicates or observed != expected or len(oof) != len(expected):
        raise ValueError({"duplicate_ids": duplicates,
                          "missing_ids": sorted(expected - observed),
                          "unexpected_ids": sorted(observed - expected)})
    folds = pd.DataFrame(fold_rows).sort_values("fold")
    metric_cols = [c for c in oof.columns if c not in ("fold", "song_id")]
    summary = {
        "songs": len(oof),
        "folds": len(folds),
        "out_of_fold_metrics": {c: float(oof[c].mean()) for c in metric_cols},
        "fold_mean_metrics": {c: float(folds[c].mean()) for c in metric_cols},
        "fold_sd_metrics": {c: float(folds[c].std(ddof=1)) for c in metric_cols},
        "all_songs_scored_once": True,
    }
    oof.to_csv(root / "out_of_fold_by_song.csv", index=False)
    folds.to_csv(root / "fold_metrics.csv", index=False)
    (root / "crossval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def cross_validate(manifest, reference, out, architecture="v1", representation="mix",
                   delay=1.0, harmonic=True, long_valence=False, n_folds=5,
                   seed=2026, epochs=15, hidden=96, device="cuda"):
    """Train one model per fold and produce strict out-of-fold predictions."""
    df = load_manifest(manifest)
    reference_split = _reference_split(df, reference)
    eligible = list(map(str, reference_split["train"]))
    folds = make_folds(eligible, n_folds=n_folds, seed=seed)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": "DEAM",
        "scope": "original reference training partition only",
        "eligible_songs": len(eligible),
        "n_folds": n_folds,
        "split_seed": seed,
        "model_seed_per_fold": [seed + i for i in range(n_folds)],
        "architecture": architecture,
        "representation": representation,
        "delay": delay,
        "harmonic": harmonic,
        "long_valence": long_valence,
        "epochs": epochs,
        "hidden": hidden,
        "folds": folds,
        "excluded_reference_partitions": ["val", "scale", "cal", "test"],
        "test_set_used": False,
    }
    protocol_path = root / "crossval_protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing cross-validation protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    for i, valid in enumerate(folds):
        train = [sid for j, fold in enumerate(folds) if j != i for sid in fold]
        run(manifest, reference, root / f"fold{i}", architecture=architecture,
            representation=representation, delay=delay, harmonic=harmonic,
            long_valence=long_valence, seed=seed + i, epochs=epochs,
            hidden=hidden, device=device,
            split_override={"train": train, "val": valid})
    return aggregate(root, eligible)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--architecture", choices=("v1", "echo_event", "trace",
                   "trace_no_cue_loss", "trace_no_cues"), default="v1")
    p.add_argument("--representation", choices=("last", "mix"), default="mix")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--harmonic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--long-valence", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    result = cross_validate(a.manifest, a.reference, a.out, a.architecture,
                            a.representation, a.delay, a.harmonic, a.long_valence,
                            a.folds, a.seed, a.epochs, a.hidden, a.device)
    print(json.dumps(result, indent=2))
