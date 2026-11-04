import io

import pytest
from fastapi.testclient import TestClient

from backend.app.connectors.bank import from_psd2_item
from backend.app.connectors.csv_source import parse_csv
from backend.app.connectors.simulated import SimulatedSource
from backend.app.main import create_app


def _generic_csv(n=300):
    txs = SimulatedSource("generic_tx", fraud_rate=0.03, seed=1).fetch(n)
    lines = ["timestamp,amount,merchant,label"] + [
        f"{t.timestamp.isoformat()},{t.amount},{t.merchant},{t.label}" for t in txs]
    return "\n".join(lines).encode()


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(model_dir=tmp_path, database_url=f"sqlite:///{tmp_path}/t.db",
                                 data_path=str(tmp_path / "none.csv")))


def test_parse_generic_csv():
    txs = parse_csv(_generic_csv(100))
    assert len(txs) == 100 and txs[0].merchant and txs[0].label in (0, 1)


def test_parse_csv_rejects_bad_columns():
    with pytest.raises(ValueError):
        parse_csv(b"a,b\n1,2\n")


def test_psd2_mapping():
    t = from_psd2_item({"transactionId": "abc", "bookingDate": "2026-10-01",
                        "transactionAmount": {"amount": "-12.50", "currency": "EUR"}, "creditorName": "Shop"})
    assert t.amount == 12.5 and t.currency == "EUR" and t.merchant == "Shop"


def test_profile_train_upload_and_stream(client):
    r = client.post("/api/sandbox/profile/train", files={"file": ("h.csv", io.BytesIO(_generic_csv(400)))},
                    data={"name": "my_profile"})
    assert r.status_code == 200 and r.json()["schema"] == "generic_tx"
    up = client.post("/api/sandbox/upload", files={"file": ("n.csv", io.BytesIO(_generic_csv(200)))},
                     data={"model": "my_profile"})
    body = up.json()
    assert up.status_code == 200 and body["n"] == 200 and body["metrics"] is not None
    st = client.get("/api/stream/next", params={"model": "my_profile", "n": 3})
    assert st.status_code == 200 and len(st.json()) == 3
    assert client.post("/api/sandbox/profile/train", files={"file": ("h.csv", io.BytesIO(b"x"))},
                       data={"name": "../evil"}).status_code == 422
