# payment-anomaly-detection

Praca inżynierska (AGH, Teleinformatyka): webowy system detekcji anomalii w transakcjach płatniczych
z porównawczą analizą algorytmów uczenia maszynowego (nadzorowane vs nienadzorowane).

**Stack:** Python, scikit-learn, XGBoost, FastAPI, SQLAlchemy (SQLite → PostgreSQL), React.

## Struktura
```
anomaly/      kod ML: dane, modele (rejestr), metryki, trening (CLI)
backend/app/  API FastAPI (predykcja, modele, metryki, historia)
frontend/     panel React (do zrobienia, patrz frontend/README.md)
notebooks/    EDA i eksperymenty
tests/        pytest
data/raw/     dane (poza gitem)      models/   wytrenowane modele (poza gitem)
docs/         wykresy i notatki do pracy
```

## Szybki start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 1) dane: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud) -> data/raw/creditcard.csv
#    bez danych: python -m anomaly.synthetic data/raw/creditcard.csv   (dane syntetyczne do testów)

# 2) trening wszystkich modeli (zapis w models/)
python -m anomaly.train --data data/raw/creditcard.csv --models all

# 3) API
uvicorn backend.app.main:app --reload       # http://127.0.0.1:8000/docs

# 4) testy
pytest
```

## Architektura i rozszerzenia
Moduły: `connectors/` (źródła transakcji: symulator, replay, CSV, szkielet banku), `anomaly/schemas.py` (schematy cech),
`services/scoring.py` (ocena), `routers/` (HTTP). Szczegóły i jak dodać nowe źródło/model: `docs/architecture.md`,
wybór API bankowego: `docs/bank_api.md`, plan nauki: `docs/pojecia-do-nauki.md`.

Nowe endpointy: `GET /api/sources`, `GET /api/stream/next`, `POST /api/sandbox/simulate`, `POST /api/sandbox/upload`,
`POST /api/sandbox/profile/train`.

## Eksperymenty do pracy
`python -m anomaly.experiments --data data/raw/creditcard.csv --out results` generuje tabele LaTeX, wykresy i CSV
(wyniki główne z seedami i bootstrapem, ustawienie zbliżone do [2], wagi klas vs SMOTE, okna czasowe). Opis: `docs/experiments.md`.

## Założenia metodyczne
- podział **czasowy** train/val/test (70/15/15), próg decyzyjny wybierany na walidacji,
- metryki: PR-AUC, ROC-AUC, precision, recall, F1, macierz pomyłek (bez accuracy),
- wszystkie modele zwracają wynik „im wyższy, tym bardziej podejrzane”.

## Status
- [x] szkielet repo, ML, API, testy
- [ ] eksperymenty na prawdziwych danych
- [ ] frontend
- [ ] retrain + wersjonowanie modeli w bazie
- [x] sandbox: symulator, upload CSV, strumień (`/api/stream/next`), model profilu
- [ ] konektor bankowy (PoC)
