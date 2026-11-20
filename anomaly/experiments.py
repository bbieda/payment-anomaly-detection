"""Eksperymenty do rozdziału „Badania własne”: wyniki główne, replikacja ustawienia z [2], wagi vs SMOTE,
okna czasowe, przedziały ufności (bootstrap). Zapisuje CSV, tabele LaTeX i wykresy.

python -m anomaly.experiments --data data/raw/creditcard.csv --out results --what all
Szybki test:  python -m anomaly.experiments --synthetic 20000 --fast --seeds 2 --bootstrap 100 --out /tmp/res
"""
import sys
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import report
from .data import FEATURES, TARGET, TIME, load_raw, temporal_split
from .metrics import best_f1_threshold, evaluate
from .models import REGISTRY, anomaly_score
from .train import fit_pipeline

METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
SUPERVISED = ["logreg", "random_forest", "xgboost"]
FAST = {"random_forest": {"n_estimators": 50}, "xgboost": {"n_estimators": 100},
        "isolation_forest": {"n_estimators": 50}, "autoencoder": {"max_iter": 20}}
# Wartości AUROC podane w tekście pracy [2] (tylko modele, które mamy również u siebie).
# OCSVM: wartość zaokrąglona z tekstu ("OC-SVM ... 0.90"); AE: w tekście tylko "powyżej 0,95".
NIU_REPORTED = {"xgboost": 0.989, "random_forest": 0.988, "ocsvm": 0.90}


def _params(name, fast):
    return FAST.get(name) if fast else None


def _agg(rows, keys=METRICS):
    d = pd.DataFrame(rows)
    out = {}
    for k in keys:
        out[f"{k}_mean"], out[f"{k}_std"] = float(d[k].mean()), float(d[k].std(ddof=0))
    return out


def split_summary(splits):
    rows = [{"split": n, "n": len(d), "n_fraud": int(d[TARGET].sum()), "fraud_pct": 100 * d[TARGET].mean(),
             "t_from": d[TIME].min(), "t_to": d[TIME].max()} for n, d in zip(("train", "walidacja", "test"), splits)]
    return pd.DataFrame(rows)


def bootstrap_ci(y, s, thr, n_boot=500, seed=0, alpha=0.05):
    """Percentylowe przedziały ufności metryk na zbiorze testowym (próbkowanie ze zwracaniem)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pred, n = (s >= thr).astype(int), len(y)
    vals = {k: [] for k in ("pr_auc", "roc_auc", "f1", "precision", "recall")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        sb, pb = s[idx], pred[idx]
        vals["pr_auc"].append(average_precision_score(yb, sb))
        vals["roc_auc"].append(roc_auc_score(yb, sb))
        vals["f1"].append(f1_score(yb, pb, zero_division=0))
        vals["precision"].append(precision_score(yb, pb, zero_division=0))
        vals["recall"].append(recall_score(yb, pb, zero_division=0))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {k: tuple(np.percentile(v, q)) if v else (np.nan, np.nan) for k, v in vals.items()}


def paired_bootstrap(y, scores, n_boot=500, seed=0, alpha=0.05):
    """Bootstrap parami różnic PR-AUC między modelami na tym samym zbiorze testowym.

    W każdej próbie bootstrapowej losujemy te same indeksy dla wszystkich modeli, więc różnica
    PR-AUC(A) - PR-AUC(B) uwzględnia korelację wyników na tych samych transakcjach.
    Zwraca DataFrame: model_a, model_b, delta (na pełnym teście), lo/hi (percentylowy przedział),
    p_boot (dwustronny, orientacyjny) i significant (przedział nie zawiera 0).
    Uwzględnia tylko zmienność zbioru testowego, nie zmienność treningu (seedy).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    names = list(scores)
    S = {k: np.asarray(v) for k, v in scores.items()}
    n = len(y)
    boot = {k: [] for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        for k in names:
            boot[k].append(average_precision_score(yb, S[k][idx]))
    boot = {k: np.asarray(v) for k, v in boot.items()}
    point = {k: average_precision_score(y, S[k]) for k in names}
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, q) if len(d) else (np.nan, np.nan)
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan
            rows.append({"model_a": a, "model_b": b, "delta": point[a] - point[b], "lo": lo, "hi": hi,
                         "p_boot": p, "significant": bool(lo > 0 or hi < 0), "n_boot": int(len(d))})
    return pd.DataFrame(rows)


def run_pairs(y, scores, out, n_boot, seed=0):
    """Liczy i zapisuje bootstrap parami (CSV, tabela LaTeX, wykres). Wymaga co najmniej 2 modeli."""
    if len(scores) < 2:
        print("[pary] pominięto: potrzeba co najmniej 2 modeli")
        return None
    out = Path(out)
    d = paired_bootstrap(y, scores, n_boot, seed)
    d.to_csv(out / "csv" / "pary.csv", index=False)
    report.write(out / "tables" / "tab_pary.tex", report.tab_pairs(d))
    report.plot_pairs(d, out / "figures" / "pary_roznice")
    print(f"[pary] {len(d)} par, istotnych (95% CI bez zera): {int(d.significant.sum())}")
    return d


def run_main(splits, models, seeds, n_boot, fast=False):
    train, val, test = splits
    rows, scores, cms = [], {}, {}
    for name in models:
        spec, per_seed = REGISTRY[name], []
        for sd in seeds:
            pipe, _, fit_s = fit_pipeline(name, train, _params(name, fast), seed=sd)
            thr = best_f1_threshold(val[TARGET], anomaly_score(pipe, val[FEATURES], spec.kind))
            st = anomaly_score(pipe, test[FEATURES], spec.kind)
            m = evaluate(test[TARGET], st, thr)
            per_seed.append({**{k: m[k] for k in METRICS}, "fit_s": fit_s})
            if sd == seeds[0]:
                t0 = time.perf_counter()
                one = test[FEATURES].head(100)
                for i in range(len(one)):
                    anomaly_score(pipe, one.iloc[[i]], spec.kind)
                lat = (time.perf_counter() - t0) / len(one) * 1000
                scores[name] = (test[TARGET].to_numpy(), st, thr)
                cms[name] = m["confusion_matrix"]
                seed0 = m
        row = {"model": name, "kind": spec.kind, **_agg(per_seed, METRICS + ["fit_s"]), "latency_ms": lat}
        b = bootstrap_ci(test[TARGET], scores[name][1], scores[name][2], n_boot)
        for k in ("pr_auc", "roc_auc", "f1", "recall"):
            row[f"{k}_seed0"], (row[f"{k}_lo"], row[f"{k}_hi"]) = seed0[k], b[k]
        rows.append(row)
        print(f"[main] {name}: PR-AUC={row['pr_auc_mean']:.3f}±{row['pr_auc_std']:.3f}")
    return pd.DataFrame(rows), scores, cms


def run_niu(df, models, n_folds=5, undersample=1.0, seed=42, fast=False):
    """Ustawienie zbliżone do [2]: losowa walidacja krzyżowa, AUROC, niedopróbkowanie dla modeli nadzorowanych.
    Stosunek klas po niedopróbkowaniu (undersample = n_norma/n_oszustwa) to założenie, nie dane z [2]."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    rows = []
    for name in models:
        spec, res = REGISTRY[name], []
        for tr_i, te_i in skf.split(df[FEATURES], df[TARGET]):
            tr, te = df.iloc[tr_i], df.iloc[te_i]
            if spec.kind == "supervised" and undersample:
                pos, neg = tr[tr[TARGET] == 1], tr[tr[TARGET] == 0]
                neg = neg.sample(min(len(neg), int(len(pos) * undersample)), random_state=seed)
                tr = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
            pipe, _, _ = fit_pipeline(name, tr, _params(name, fast), seed=seed)
            s = anomaly_score(pipe, te[FEATURES], spec.kind)
            res.append({"auroc": roc_auc_score(te[TARGET], s), "pr_auc": average_precision_score(te[TARGET], s)})
        a = _agg(res, ["auroc", "pr_auc"])
        rows.append({"model": name, "kind": spec.kind, **a, "auroc_niu": NIU_REPORTED.get(name, np.nan)})
        print(f"[niu] {name}: AUROC={a['auroc_mean']:.3f}±{a['auroc_std']:.3f}")
    return pd.DataFrame(rows)


def _variants(seed):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    return {"brak": dict(weighted=False), "wagi klas": dict(weighted=True),
            "SMOTE": dict(weighted=False, sampler=SMOTE(sampling_strategy=0.1, random_state=seed)),
            "undersampling": dict(weighted=False, sampler=RandomUnderSampler(sampling_strategy=0.1, random_state=seed))}


def run_imbalance(splits, models, seeds, fast=False):
    train, val, test = splits
    rows = []
    for name in models:
        spec = REGISTRY[name]
        for vname in ("brak", "wagi klas", "SMOTE", "undersampling"):
            res = []
            for sd in seeds:
                pipe, _, _ = fit_pipeline(name, train, _params(name, fast), seed=sd, **_variants(sd)[vname])
                thr = best_f1_threshold(val[TARGET], anomaly_score(pipe, val[FEATURES], spec.kind))
                res.append(evaluate(test[TARGET], anomaly_score(pipe, test[FEATURES], spec.kind), thr))
            rows.append({"model": name, "variant": vname, **_agg(res)})
            print(f"[imb] {name}/{vname}: PR-AUC={rows[-1]['pr_auc_mean']:.3f}")
    return pd.DataFrame(rows)


def rolling_splits(df, n_folds=4, frac=0.15):
    """Okna z przesuwanym początkiem: test = ostatnie 15%, potem poprzednie okna; walidacja tuż przed testem."""
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    n = len(df)
    out = []
    for i in range(n_folds):
        end, te0, va0 = int(n * (1 - i * frac)), int(n * (1 - (i + 1) * frac)), int(n * (1 - (i + 2) * frac))
        out.append((df.iloc[:va0], df.iloc[va0:te0], df.iloc[te0:end]))
    return out[::-1]


def run_rolling(df, models, n_folds=4, seed=42, fast=False):
    rows, counts = [], []
    for w, (tr, va, te) in enumerate(rolling_splits(df, n_folds), 1):
        counts.append(int(te[TARGET].sum()))
        for name in models:
            spec = REGISTRY[name]
            pipe, _, _ = fit_pipeline(name, tr, _params(name, fast), seed=seed)
            thr = best_f1_threshold(va[TARGET], anomaly_score(pipe, va[FEATURES], spec.kind))
            m = evaluate(te[TARGET], anomaly_score(pipe, te[FEATURES], spec.kind), thr)
            rows.append({"window": w, "model": name, "n_fraud_test": counts[-1], **{k: m[k] for k in METRICS}})
        print(f"[okna] okno {w}: oszustw w teście = {counts[-1]}")
    long = pd.DataFrame(rows)
    summ = long.groupby("model", sort=False).agg(**{f"{k}_{f}": (k, f) for k in METRICS for f in ("mean", "std")}).reset_index()
    summ = summ.fillna(0.0)
    return long, summ, counts


# Siatki do diagnozy skali i strojenia modeli nienadzorowanych (LOF, OCSVM, autoenkoder), wybór wg PR-AUC na walidacji.
# log_amount=True: log1p(Amount) + standaryzacja zamiast RobustScaler (diagnoza wrażliwości na skalę kwoty).
SWEEPS = {
    "lof": [{"n_neighbors": k, "log_amount": la} for la in (False, True) for k in (20, 50, 100)],
    "ocsvm": [{"nu": nu, "gamma": g, "log_amount": la}
              for la in (False, True) for nu in (0.01, 0.1) for g in (0.001, 0.01, 0.1)],
    "autoencoder": [{"hidden_layer_sizes": h, "log_amount": la}
                    for la in (False, True) for h in ([16, 32, 32, 16], [14, 7, 14])],
}


def run_sweep(splits, models, seed=42, fast=False):
    """Dla każdego modelu z SWEEPS uczy wszystkie konfiguracje i wybiera najlepszą wg PR-AUC na WALIDACJI.
    Test jest tylko raportowany (nigdy nie służy do wyboru). Jedna konfiguracja = pierwszy seed."""
    train, val, test = splits
    rows = []
    for name in [m for m in models if m in SWEEPS]:
        spec = REGISTRY[name]
        res = []
        for cfg in SWEEPS[name]:
            pipe, p, sec = fit_pipeline(name, train, {**(_params(name, fast) or {}), **cfg}, seed)
            sv = anomaly_score(pipe, val[FEATURES], spec.kind)
            st = anomaly_score(pipe, test[FEATURES], spec.kind)
            res.append({"model": name, "config": json.dumps(cfg), "log_amount": bool(cfg.get("log_amount", False)),
                        "val_pr_auc": average_precision_score(val[TARGET], sv),
                        "val_roc_auc": roc_auc_score(val[TARGET], sv),
                        "test_pr_auc": average_precision_score(test[TARGET], st),
                        "test_roc_auc": roc_auc_score(test[TARGET], st), "fit_s": sec})
        best = int(np.argmax([r["val_pr_auc"] for r in res]))
        for i, r in enumerate(res):
            r["domyslna"] = i == _default_index(name)
            r["wybrana"] = i == best
        rows += res
    return pd.DataFrame(rows)


def _default_index(name):
    d = REGISTRY[name].defaults
    for i, cfg in enumerate(SWEEPS[name]):
        if not cfg.get("log_amount", False) and all(cfg[k] == d[k] for k in cfg if k != "log_amount"):
            return i
    return -1


def run_all(df, out, models, what, n_seeds=5, n_boot=500, folds=5, undersample=1.0, fast=False):
    out = Path(out)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    seeds = [42 + i for i in range(n_seeds)]
    splits = temporal_split(df)
    t = out / "tables"
    f = out / "figures"
    sp = split_summary(splits)
    sp.to_csv(out / "csv" / "podzial.csv", index=False)
    report.write(t / "tab_podzial.tex", report.tab_split(sp))
    todo = {"main", "niu", "imbalance", "rolling", "sweep"} if "all" in what else set(what)
    if "main" in todo:
        summ, scores, cms = run_main(splits, models, seeds, n_boot, fast)
        summ.to_csv(out / "csv" / "wyniki_glowne.csv", index=False)
        np.savez_compressed(out / "csv" / "scores_test.npz", **{n: v[1] for n, v in scores.items()},
                            y_test=next(iter(scores.values()))[0])
        report.write(t / "tab_wyniki_glowne.tex", report.tab_main(summ))
        report.write(t / "tab_przedzialy.tex", report.tab_ci(summ))
        report.plot_curves({n: (v[0], v[1]) for n, v in scores.items()}, "pr", f / "krzywe_pr")
        report.plot_curves({n: (v[0], v[1]) for n, v in scores.items()}, "roc", f / "krzywe_roc")
        report.plot_ci(summ, f / "przedzialy_pr_auc")
        report.plot_confusion(cms, f / "macierze_pomylek")
        run_pairs(next(iter(scores.values()))[0], {n: v[1] for n, v in scores.items()}, out, n_boot, seeds[0])
    if "pairs" in todo and "main" not in todo:      # z zapisanych wyników (bez ponownego uczenia)
        npz = np.load(out / "csv" / "scores_test.npz")
        sc = {k: npz[k] for k in npz.files if k != "y_test"}
        sc = {k: v for k, v in sc.items() if k in models}
        run_pairs(npz["y_test"], sc, out, n_boot, seeds[0])
    if "niu" in todo:
        d = run_niu(df, models, folds, undersample, seeds[0], fast)
        d.to_csv(out / "csv" / "niu.csv", index=False)
        report.write(t / "tab_niu.tex", report.tab_niu(d))
        report.plot_niu(d, f / "niu_auroc")
    if "imbalance" in todo:
        d = run_imbalance(splits, [m for m in models if m in SUPERVISED], seeds[:3], fast)
        d.to_csv(out / "csv" / "niezbalansowanie.csv", index=False)
        report.write(t / "tab_niezbalansowanie.tex", report.tab_imb(d))
        report.plot_imbalance(d, f / "niezbalansowanie")
    if "rolling" in todo:
        long, summ, counts = run_rolling(df, models, seed=seeds[0], fast=fast)
        long.to_csv(out / "csv" / "okna.csv", index=False)
        report.write(t / "tab_okna.tex", report.tab_windows(summ, counts))
        report.plot_windows(long, f / "okna_pr_auc")
    if "sweep" in todo:
        d = run_sweep(splits, models, seeds[0], fast)
        if len(d):
            d.to_csv(out / "csv" / "strojenie.csv", index=False)
            report.write(t / "tab_strojenie.tex", report.tab_sweep(d))
    info = {"created": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "seeds": seeds,
            "bootstrap": n_boot, "folds": folds, "undersample": undersample, "fast": fast, "what": sorted(todo),
            "models": models, "n_rows": len(df)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/creditcard.csv")
    ap.add_argument("--synthetic", type=int, default=0, help="dane syntetyczne (liczba wierszy) zamiast pliku")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="all")
    ap.add_argument("--what", default="all", help="all lub lista: main,niu,imbalance,rolling,sweep,pairs (pairs liczy się też razem z main)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--undersample", type=float, default=1.0, help="stosunek normalnych do oszustw po niedopróbkowaniu ([2]-like)")
    ap.add_argument("--fast", action="store_true", help="mniej drzew (tylko do szybkich testów)")
    a = ap.parse_args()
    if a.synthetic:
        from .synthetic import make_synthetic
        df = make_synthetic(a.synthetic, 0.01, seed=1)
    else:
        df = load_raw(a.data)
    models = list(REGISTRY) if a.models == "all" else a.models.split(",")
    run_all(df, a.out, models, a.what.split(","), a.seeds, a.bootstrap, a.folds, a.undersample, a.fast)
    print("Gotowe:", a.out)


if __name__ == "__main__":
    main()
