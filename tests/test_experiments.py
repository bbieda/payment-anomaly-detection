import numpy as np

from anomaly.data import temporal_split
from anomaly.experiments import SWEEPS, bootstrap_ci, run_sweep, paired_bootstrap, rolling_splits, run_all
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


def test_paired_bootstrap_detects_real_difference_and_not_noise():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(int)
    good = y * 0.8 + rng.random(4000) * 0.5
    noise = rng.random(4000)
    d = paired_bootstrap(y, {"good": good, "good_copy": good.copy(), "noise": noise}, n_boot=200)
    row = lambda a, b: d[(d.model_a == a) & (d.model_b == b)].iloc[0]
    assert row("good", "noise").significant and row("good", "noise").delta > 0.3
    same = row("good", "good_copy")
    assert same.delta == 0 and same.lo == 0 and same.hi == 0 and not same.significant
    assert len(d) == 3 and ((d.p_boot >= 0) & (d.p_boot <= 1)).all()


def test_pairs_can_be_recomputed_from_saved_scores(tmp_path):
    df = make_synthetic(6000, 0.03)
    run_all(df, tmp_path, ["logreg", "random_forest"], ["main"], n_seeds=1, n_boot=20, fast=True)
    (tmp_path / "csv" / "pary.csv").unlink()
    run_all(df, tmp_path, ["logreg", "random_forest"], ["pairs"], n_seeds=1, n_boot=20, fast=True)
    assert (tmp_path / "csv" / "pary.csv").exists() and (tmp_path / "tables" / "tab_pary.tex").exists()


def test_run_all_writes_tables_and_figures(tmp_path):
    df = make_synthetic(6000, 0.03)
    run_all(df, tmp_path, ["logreg", "isolation_forest", "ocsvm", "autoencoder"], ["all"],
            n_seeds=2, n_boot=20, folds=3, fast=True)
    assert (tmp_path / "csv" / "pary.csv").exists()
    assert (tmp_path / "figures" / "pary_roznice.pdf").exists()
    for f in ("tab_podzial", "tab_wyniki_glowne", "tab_przedzialy", "tab_pary", "tab_niu", "tab_niezbalansowanie", "tab_okna"):
        txt = (tmp_path / "tables" / f"{f}.tex").read_text()
        assert "\\toprule" in txt and "\\label{tab:" in txt
    for f in ("krzywe_pr", "krzywe_roc", "macierze_pomylek", "niezbalansowanie", "okna_pr_auc"):
        assert (tmp_path / "figures" / f"{f}.pdf").exists()
    assert (tmp_path / "run_info.json").exists()


def test_sweep_picks_config_on_validation_only():
    splits = temporal_split(make_synthetic(6000, 0.03))
    d = run_sweep(splits, ["lof", "ocsvm", "logreg"], fast=True)
    assert set(d.model) == {"lof", "ocsvm"}                       # modele spoza SWEEPS pominięte
    for name, g in d.groupby("model"):
        assert len(g) == len(SWEEPS[name]) and g.wybrana.sum() == 1 and g.domyslna.sum() == 1
        assert g[g.wybrana].val_pr_auc.iloc[0] == g.val_pr_auc.max()
