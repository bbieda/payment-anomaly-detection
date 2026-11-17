"""API systemu detekcji anomalii.  Uruchom z katalogu głównego repo:

    uvicorn backend.app.main:app --reload      (Swagger: http://127.0.0.1:8000/docs)

Moduły: routers/ (HTTP), services/ (logika oceny), connectors/ (źródła transakcji), store.py (modele),
db.py (historia ocen). Endpointy to zwykłe funkcje (def): predykcja sklearn jest CPU-bound,
a FastAPI uruchamia je w puli wątków.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .db import make_session_factory
from .routers import models, predict, sandbox
from .store import ModelStore


def create_app(model_dir=None, database_url=None, data_path=None) -> FastAPI:
    app = FastAPI(title="Payment Anomaly Detection API", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.model_dir = model_dir or os.getenv("MODEL_DIR", "models")
    app.state.data_path = data_path or os.getenv("DATA_PATH", "data/raw/creditcard.csv")
    app.state.store = ModelStore(app.state.model_dir)
    app.state.Session = make_session_factory(database_url or os.getenv("DATABASE_URL", "sqlite:///./app.db"))
    app.state.jobs = {}
    app.state.streams = {}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "active_model": app.state.store.active}

    for r in (models.router, predict.router, sandbox.router):
        app.include_router(r)

    # Zbudowany panel (npm run build) serwowany przez ten sam serwer: jedna komenda do demo.
    dist = Path(os.getenv("FRONTEND_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
