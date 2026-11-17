"""Rejestr modeli. Każdy model to Pipeline sklearn + wspólny interfejs oceny anomalii.

Konwencja: im WYŻSZY wynik (score), tym bardziej podejrzana transakcja.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

from .autoencoder import AutoencoderDetector


def _prep() -> ColumnTransformer:
    # V1..V28 to już PCA (przeskalowane); skalujemy tylko Amount
    return ColumnTransformer([("amount", RobustScaler(), ["Amount"])], remainder="passthrough")


def _prep_log() -> ColumnTransformer:
    # wariant opcjonalny (parametr log_amount=True) dla modeli wrażliwych na skalę: log1p(Amount) + standaryzacja
    amount = Pipeline([("log", FunctionTransformer(np.log1p)), ("std", StandardScaler())])
    return ColumnTransformer([("amount", amount, ["Amount"])], remainder="passthrough")


def _prep_for(p: dict):
    """Zwraca (preprocessing, parametry bez klucza log_amount)."""
    p = dict(p)
    return (_prep_log() if p.pop("log_amount", False) else _prep()), p


def _logreg(p, seed, w):
    return Pipeline([("prep", _prep()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight=None if w is None else "balanced",
                                                random_state=seed, **p))])


def _rf(p, seed, w):
    return Pipeline([("prep", _prep()),
                     ("clf", RandomForestClassifier(class_weight=None if w is None else "balanced_subsample",
                                                    n_jobs=-1, random_state=seed, **p))])


def _xgb(p, seed, w):
    from xgboost import XGBClassifier
    return Pipeline([("prep", _prep()),
                     ("clf", XGBClassifier(scale_pos_weight=1.0 if w is None else w, eval_metric="aucpr",
                                           n_jobs=-1, random_state=seed, **p))])


def _iforest(p, seed, w):
    return Pipeline([("prep", _prep()),
                     ("clf", IsolationForest(n_jobs=-1, random_state=seed, **p))])


def _lof(p, seed, w):
    return Pipeline([("prep", _prep()),
                     ("clf", LocalOutlierFactor(novelty=True, n_jobs=-1, **p))])


def _ocsvm(p, seed, w):
    prep, p = _prep_for(p)
    return Pipeline([("prep", prep), ("clf", OneClassSVM(cache_size=1000, **p))])


def _autoencoder(p, seed, w):
    prep, p = _prep_for(p)
    return Pipeline([("prep", prep), ("clf", AutoencoderDetector(random_state=seed, **p))])


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                         # "supervised" | "unsupervised"
    builder: Callable                 # (params, seed, pos_weight lub None) -> Pipeline
    defaults: dict = field(default_factory=dict)
    max_train_rows: int | None = None  # LOF i OCSVM są kosztowne -- ogranicz próbkę
    normal_only: bool = False          # wymuszone uczenie tylko na transakcjach normalnych (autoenkoder)


# Hiperparametry nowych modeli (OCSVM, autoenkoder) wzięte z pracy [2]; reszta modeli ma własne wartości domyślne.
REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("logreg", "supervised", _logreg, {"C": 1.0}),
    ModelSpec("random_forest", "supervised", _rf, {"n_estimators": 200, "max_depth": None}),
    ModelSpec("xgboost", "supervised", _xgb,
              {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8}),
    ModelSpec("isolation_forest", "unsupervised", _iforest, {"n_estimators": 200}),
    ModelSpec("lof", "unsupervised", _lof, {"n_neighbors": 20}, max_train_rows=50_000),
    ModelSpec("ocsvm", "unsupervised", _ocsvm, {"kernel": "rbf", "nu": 0.1, "gamma": 0.001}, max_train_rows=20_000),
    ModelSpec("autoencoder", "unsupervised", _autoencoder,
              {"hidden_layer_sizes": [16, 32, 32, 16], "max_iter": 100, "alpha": 1e-4}, normal_only=True),
]}


def anomaly_score(pipe: Pipeline, X, kind: str):
    """Ujednolicony wynik: wyższy = bardziej podejrzane."""
    if kind == "supervised":
        return pipe.predict_proba(X)[:, 1]
    return -pipe.score_samples(X)   # IF / LOF / OCSVM / autoenkoder: niższy score_samples = anomalia
