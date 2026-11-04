import os
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from anomaly.profile import train_profile

from ..connectors.csv_source import parse_csv
from ..connectors.simulated import SimulatedSource
from ..deps import get_db
from ..services.scoring import resolve_model, save_results, score_transactions, summarize
from .predict import to_http

router = APIRouter(prefix="/api", tags=["sandbox"])
MAX_UPLOAD = 20 * 1024 * 1024


class SimulateRequest(BaseModel):
    n: int = Field(200, ge=1, le=2000)
    fraud_rate: float = Field(0.02, ge=0, le=0.5)
    seed: int = 0
    model: str | None = None


async def _read(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "Plik jest za duży (limit 20 MB).")
    return content


@router.get("/sources")
def sources(request: Request):
    return [{"name": "simulated", "available": True, "description": "Symulator transakcji (sandbox)"},
            {"name": "replay", "available": os.path.exists(request.app.state.data_path),
             "description": "Odtwarzanie zbioru testowego Kaggle"},
            {"name": "csv_upload", "available": True, "description": "Własne transakcje z pliku CSV"},
            {"name": "bank", "available": False, "description": "Konektor bankowy (PSD2) – do zrobienia"}]


@router.post("/sandbox/simulate")
def simulate(req: SimulateRequest, request: Request, db=Depends(get_db)):
    store = request.app.state.store
    try:
        name = resolve_model(store, req.model)
        schema = store.meta(name).get("schema", "kaggle_pca")
        txs = SimulatedSource(schema, req.fraud_rate, req.seed).fetch(req.n)
        res = score_transactions(store, name, txs)
    except Exception as e:  # noqa: BLE001
        raise to_http(e)
    save_results(db, "sandbox_sim", res)
    return summarize(res)


@router.post("/sandbox/upload")
async def upload(request: Request, file: UploadFile = File(...), model: str | None = Form(None), db=Depends(get_db)):
    store = request.app.state.store
    try:
        txs = parse_csv(await _read(file))
        res = score_transactions(store, resolve_model(store, model), txs)
    except ValueError as e:
        if type(e).__name__ == "SchemaMismatch":
            raise to_http(e)
        raise HTTPException(422, str(e))
    except (KeyError, LookupError) as e:
        raise to_http(e)
    save_results(db, "sandbox_csv", res)
    return summarize(res)


@router.post("/sandbox/profile/train")
async def train_user_profile(request: Request, file: UploadFile = File(...), name: str = Form("profile_isolation_forest")):
    """Buduje model profilu z własnej historii transakcji (Isolation Forest, bez etykiet)."""
    if not re.fullmatch(r"[a-z0-9_]{3,40}", name):
        raise HTTPException(422, "Nazwa modelu: 3-40 znaków [a-z0-9_].")
    try:
        meta = train_profile(parse_csv(await _read(file)), out_dir=request.app.state.model_dir, name=name)
    except ValueError as e:
        raise HTTPException(422, str(e))
    request.app.state.store.refresh()
    return {"name": meta["name"], "schema": meta["schema"], "n_train": meta["n_train"], "threshold": meta["threshold"]}
