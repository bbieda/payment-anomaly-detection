"""Trening i ewaluacja modeli. Zapisuje <model>.joblib i <model>.json (metryki, próg, krzywe).

python -m anomaly.train --data data/raw/creditcard.csv --models all --seed 42
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .data import FEATURES, TARGET, load_raw, temporal_split
from .metrics import best_f1_threshold, curves, evaluate
from .models import REGISTRY, anomaly_score


def fit_pipeline(name, train, params=None, seed=42, normal_only=False, weighted=True, sampler=None):
    """Buduje i uczy model. weighted=False wyłącza wagi klas; sampler (imblearn) działa tylko na treningu."""
    spec = REGISTRY[name]
    p = {**spec.defaults, **(params or {})}
    Xtr, ytr = train[FEATURES], train[TARGET]
    if spec.kind == "unsupervised" and normal_only:      # wariant „novelty”: uczenie tylko na normalnych
        mask = ytr == 0
        Xtr, ytr = Xtr[mask], ytr[mask]
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if weighted else None
    if spec.max_train_rows and len(Xtr) > spec.max_train_rows:
        Xtr = Xtr.sample(spec.max_train_rows, random_state=seed)
    pipe = spec.builder(p, seed, pos_weight)
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps = list(pipe.steps)
        steps.insert(1, ("sampler", sampler))
        pipe = ImbPipeline(steps)
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr.loc[Xtr.index]) if spec.kind == "supervised" else pipe.fit(Xtr)
    return pipe, p, time.perf_counter() - t0


def train_one(name, splits, params=None, seed=42, normal_only=False, out_dir="models") -> dict:
    spec = REGISTRY[name]
    train, val, test = splits
    pipe, p, fit_seconds = fit_pipeline(name, train, params, seed, normal_only)

    thr = best_f1_threshold(val[TARGET], anomaly_score(pipe, val[FEATURES], spec.kind))
    s_test = anomaly_score(pipe, test[FEATURES], spec.kind)

    # opóźnienie predykcji pojedynczej transakcji (do rozdziału o wydajności)
    sample = test[FEATURES].head(100)
    t0 = time.perf_counter()
    for i in range(len(sample)):
        anomaly_score(pipe, sample.iloc[[i]], spec.kind)
    latency_ms = (time.perf_counter() - t0) / max(len(sample), 1) * 1000

    meta = {
        "name": name, "kind": spec.kind, "schema": "kaggle_pca", "params": p, "seed": seed,
        "normal_only": normal_only,
        "threshold": thr, "trained_at": datetime.now(timezone.utc).isoformat(),
        "fit_seconds": fit_seconds, "predict_latency_ms": latency_ms,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "metrics": evaluate(test[TARGET], s_test, thr),
        "curves": curves(test[TARGET], s_test),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / f"{name}.joblib")
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/creditcard.csv")
    ap.add_argument("--models", default="all", help="lista po przecinku lub 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normal-only", action="store_true")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    splits = temporal_split(load_raw(a.data))
    names = list(REGISTRY) if a.models == "all" else a.models.split(",")
    for n in names:
        m = train_one(n, splits, seed=a.seed, normal_only=a.normal_only, out_dir=a.out)
        k = m["metrics"]
        print(f"{n:18s} PR-AUC={k['pr_auc']:.3f} ROC-AUC={k['roc_auc']:.3f} "
              f"P={k['precision']:.3f} R={k['recall']:.3f} F1={k['f1']:.3f} fit={m['fit_seconds']:.1f}s")


if __name__ == "__main__":
    main()
