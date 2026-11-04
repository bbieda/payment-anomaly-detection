import pytest
from fastapi.testclient import TestClient

from anomaly.data import FEATURES, temporal_split
from anomaly.synthetic import make_synthetic
from anomaly.train import train_one
from backend.app.main import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    splits = temporal_split(make_synthetic(6000, 0.03))
    for name in ["logreg", "isolation_forest"]:
        train_one(name, splits, out_dir=d)
    return TestClient(create_app(model_dir=d, database_url=f"sqlite:///{d}/test.db", data_path=str(d / "none.csv")))


def test_models_and_predict(client):
    assert {m["name"] for m in client.get("/api/models").json()} == {"logreg", "isolation_forest"}
    r = client.post("/api/predict", json={"features": {f: 0.0 for f in FEATURES}, "model": "isolation_forest"})
    assert r.status_code == 200 and "score" in r.json()
    assert len(client.get("/api/transactions/recent").json()) >= 1


def test_predict_requires_amount_or_features(client):
    assert client.post("/api/predict", json={"merchant": "x"}).status_code == 422


def test_generic_amount_on_kaggle_model_is_rejected(client):
    r = client.post("/api/predict", json={"amount": 10.0, "model": "logreg"})
    assert r.status_code == 422 and "schematu" in r.json()["detail"]


def test_activate_and_metrics(client):
    assert client.post("/api/models/activate", json={"name": "logreg"}).json()["active_model"] == "logreg"
    assert client.post("/api/models/activate", json={"name": "nope"}).status_code == 404
    assert "roc" in client.get("/api/metrics/logreg").json()["curves"]
    assert set(client.get("/api/metrics").json()) == {"logreg", "isolation_forest"}


def test_stream_and_simulate_with_kaggle_model(client):
    r = client.get("/api/stream/next", params={"model": "logreg", "n": 5})
    assert r.status_code == 200 and len(r.json()) == 5
    s = client.post("/api/sandbox/simulate", json={"n": 300, "fraud_rate": 0.05, "model": "logreg"}).json()
    assert s["n"] == 300 and s["metrics"] is not None
    assert client.get("/api/stream/next", params={"source": "replay", "model": "logreg"}).status_code == 409
