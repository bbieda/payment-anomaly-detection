"""Magazyn modeli: skanuje katalog models/ i ładuje pipeline'y (cache)."""
import json
import threading
from pathlib import Path

import joblib


class ModelStore:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self._meta: dict[str, dict] = {}
        self._pipes: dict[str, object] = {}
        self._active: str | None = None
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self):
        with self._lock:
            self._meta = {p.stem: json.loads(p.read_text())
                          for p in self.dir.glob("*.json") if (self.dir / f"{p.stem}.joblib").exists()}
            self._pipes.clear()
            if self._active not in self._meta:
                # domyślnie: model z najlepszym PR-AUC (modele bez metryk, np. profile, na końcu)
                self._active = max(self._meta, key=lambda n: (self._meta[n].get("metrics") or {}).get("pr_auc", -1),
                                   default=None)

    @property
    def active(self) -> str | None:
        return self._active

    def activate(self, name: str):
        if name not in self._meta:
            raise KeyError(name)
        self._active = name

    def names(self):
        return list(self._meta)

    def meta(self, name: str) -> dict:
        return self._meta[name]

    def get(self, name: str):
        with self._lock:
            if name not in self._meta:
                raise KeyError(name)
            if name not in self._pipes:
                self._pipes[name] = joblib.load(self.dir / f"{name}.joblib")
            return self._pipes[name], self._meta[name]
