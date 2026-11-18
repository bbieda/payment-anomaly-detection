import json

import joblib
import numpy as np
import pytest

from anomaly.autoencoder import AutoencoderDetector
from anomaly.data import FEATURES, TARGET, temporal_split
from anomaly.models import REGISTRY, anomaly_score
from anomaly.synthetic import make_synthetic
from anomaly.train import fit_pipeline, train_one

NEW = ["ocsvm", "autoencoder"]


@pytest.fixture(scope="module")
def splits():
    return temporal_split(make_synthetic(6000, 0.03))


def test_new_models_are_registered_with_expected_kind():
    assert REGISTRY["ocsvm"].kind == "unsupervised" and REGISTRY["ocsvm"].max_train_rows == 20_000
    assert REGISTRY["autoencoder"].kind == "unsupervised" and REGISTRY["autoencoder"].normal_only


@pytest.mark.parametrize("name", NEW)
def test_new_models_fit_and_separate_fraud(name, splits):
    train, _, test = splits
    pipe, _, _ = fit_pipeline(name, train, {"max_iter": 30} if name == "autoencoder" else None)
    s = anomaly_score(pipe, test[FEATURES], REGISTRY[name].kind)
    assert s.shape == (len(test),) and np.isfinite(s).all()
    assert s[test[TARGET].to_numpy() == 1].mean() > s[test[TARGET].to_numpy() == 0].mean()


def test_autoencoder_trains_only_on_normal_transactions(splits, monkeypatch):
    train, _, _ = splits
    seen = {}
    orig = AutoencoderDetector.fit

    def spy(self, X, y=None):
        seen["n"] = len(X)
        return orig(self, X, y)

    monkeypatch.setattr(AutoencoderDetector, "fit", spy)
    fit_pipeline("autoencoder", train, {"max_iter": 5})
    assert seen["n"] == int((train[TARGET] == 0).sum())


def test_ocsvm_uses_capped_training_sample(splits, monkeypatch):
    from sklearn.svm import OneClassSVM
    train, _, _ = splits
    monkeypatch.setitem(REGISTRY, "ocsvm", REGISTRY["ocsvm"].__class__(**{**REGISTRY["ocsvm"].__dict__, "max_train_rows": 500}))
    pipe, _, _ = fit_pipeline("ocsvm", train)
    assert isinstance(pipe.named_steps["clf"], OneClassSVM) and pipe.named_steps["clf"].support_.max() < 500


def test_log_amount_option_changes_preprocessing(splits):
    train, _, test = splits
    p, _, _ = fit_pipeline("ocsvm", train, {"log_amount": True})
    s = anomaly_score(p, test[FEATURES], "unsupervised")
    assert np.isfinite(s).all()


@pytest.mark.parametrize("name", NEW)
def test_train_one_roundtrip(name, splits, tmp_path):
    meta = train_one(name, splits, params={"max_iter": 10} if name == "autoencoder" else None, out_dir=tmp_path)
    saved = json.loads((tmp_path / f"{name}.json").read_text())
    assert saved["name"] == name and 0 <= meta["metrics"]["roc_auc"] <= 1
    pipe = joblib.load(tmp_path / f"{name}.joblib")
    one = splits[2][FEATURES].head(1)
    assert anomaly_score(pipe, one, meta["kind"]).shape == (1,)
    if name == "autoencoder":
        assert saved["normal_only"] is True


def test_lof_log_amount_option_builds(splits):
    train, _, test = splits
    pipe, _, _ = fit_pipeline("lof", train, {"log_amount": True, "n_neighbors": 10})
    assert np.isfinite(anomaly_score(pipe, test[FEATURES].head(50), "unsupervised")).all()
