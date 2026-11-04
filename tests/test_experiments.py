import numpy as np

from anomaly.data import temporal_split
from anomaly.experiments import bootstrap_ci, rolling_splits, run_all
from anomaly.synthetic import make_synthetic


def test_bootstrap_ci_contains_point_estimate_range():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    s = y * 0.6 + rng.random(2000) * 0.7
    ci = bootstrap_ci(y, s, 0.7, n_boot=100)
    assert 0 <= ci["pr_auc"][0] <= ci["pr_auc"][1] <= 1
    assert ci["roc_auc"][0] > 0.5


def test_rolling_splits_are_ordered():
    df = make_synthetic(4000, 0.03)
    for tr, va, te in rolling_splits(df, 4):
        assert tr["Time"].max() <= va["Time"].min() and va["Time"].max() <= te["Time"].min()


def test_run_all_writes_tables_and_figures(tmp_path):
    df = make_synthetic(6000, 0.03)
    run_all(df, tmp_path, ["logreg", "isolation_forest"], ["all"], n_seeds=2, n_boot=20, folds=3, fast=True)
    for f in ("tab_podzial", "tab_wyniki_glowne", "tab_przedzialy", "tab_niu", "tab_niezbalansowanie", "tab_okna"):
        txt = (tmp_path / "tables" / f"{f}.tex").read_text()
        assert "\\toprule" in txt and "\\label{tab:" in txt
    for f in ("krzywe_pr", "krzywe_roc", "macierze_pomylek", "niezbalansowanie", "okna_pr_auc"):
        assert (tmp_path / "figures" / f"{f}.pdf").exists()
    assert (tmp_path / "run_info.json").exists()
