"""Rejestr modeli. Każdy model to Pipeline sklearn + wspólny interfejs oceny anomalii.

Konwencja: im WYŻSZY wynik (score), tym bardziej podejrzana transakcja.
"""
from dataclasses import dataclass, field
from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def _prep() -> ColumnTransformer:
    # V1..V28 to już PCA (przeskalowane); skalujemy tylko Amount
    return ColumnTransformer([("amount", RobustScaler(), ["Amount"])], remainder="passthrough")


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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                         # "supervised" | "unsupervised"
    builder: Callable                 # (params, seed, pos_weight lub None) -> Pipeline
    defaults: dict = field(default_factory=dict)
    max_train_rows: int | None = None  # LOF jest kosztowny -- ogranicz próbkę


REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("logreg", "supervised", _logreg, {"C": 1.0}),
    ModelSpec("random_forest", "supervised", _rf, {"n_estimators": 200, "max_depth": None}),
    ModelSpec("xgboost", "supervised", _xgb,
              {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8}),
    ModelSpec("isolation_forest", "unsupervised", _iforest, {"n_estimators": 200}),
    ModelSpec("lof", "unsupervised", _lof, {"n_neighbors": 20}, max_train_rows=50_000),
    # TODO: autoenkoder (np. sklearn MLPRegressor jako prosty autoencoder lub PyTorch),
    #       One-Class SVM
]}


def anomaly_score(pipe: Pipeline, X, kind: str):
    """Ujednolicony wynik: wyższy = bardziej podejrzane."""
    if kind == "supervised":
        return pipe.predict_proba(X)[:, 1]
    return -pipe.score_samples(X)   # IsolationForest / LOF: niższy score_samples = anomalia
