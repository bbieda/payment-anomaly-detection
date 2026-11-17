# payment-anomaly-detection

[English](#english) | [Polski](#polski)

---

## English

Bachelor's thesis project (AGH University, Teleinformatics): a web-based system for detecting anomalies (fraud) in
payment transactions, with a comparative analysis of supervised and unsupervised machine learning algorithms on a
highly imbalanced dataset.

**Stack:** Python, scikit-learn, XGBoost, FastAPI, SQLAlchemy (SQLite, PostgreSQL planned), React (Vite, Recharts).

### Features
- **Models:** logistic regression (baseline), Random Forest, XGBoost, Isolation Forest, Local Outlier Factor.
- **Evaluation:** chronological train/validation/test split, decision threshold chosen on validation data, metrics for
  imbalanced classes (PR-AUC, ROC AUC, precision, recall, F1, confusion matrix), multiple seeds, bootstrap confidence intervals.
- **API (FastAPI):** scoring, model management (activate, retrain), metrics, prediction history and statistics.
- **Web dashboard (React):** live transaction stream with alerts, model management, metrics and curves, single-transaction
  scoring, sandbox (simulation, CSV import, personal profile model), history. Modular: a page is one file plus one registry line.
- **Modular transaction sources:** simulator, replay of the test set, CSV upload, bank connector skeleton (PSD2-style mapping).
  Models declare a feature schema, so only compatible data can be scored.
- **Experiment runner:** exports LaTeX tables, PDF/PNG figures and CSV files for the thesis.

### Project structure
```
anomaly/      ML code: data, models (registry), metrics, training, experiments, reports
backend/app/  FastAPI app: routers, scoring service, connectors, model store, database
frontend/     React dashboard (see frontend/README.md)
notebooks/    EDA and experiments
tests/        pytest suite
docs/         architecture, experiments, bank API options, concepts to learn, thesis (LaTeX)
data/raw/     datasets (not in git)      models/   trained models (not in git)
```

### Quick start
Requirements: Python 3.10+, Node.js 18+ (only for the dashboard).
```bash
git clone <repo-url> && cd payment-anomaly-detection
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Data: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud) -> data/raw/creditcard.csv
#    No data yet? python -m anomaly.synthetic data/raw/creditcard.csv   (synthetic, for testing only)

# 2) Train all models (saved to models/)
python -m anomaly.train --models all

# 3) Run the API (Swagger UI: http://127.0.0.1:8000/docs)
uvicorn backend.app.main:app --reload

# 4) Tests
pytest
```
Defaults work without configuration. Optional settings (`MODEL_DIR`, `DATA_PATH`, `DATABASE_URL`, see `.env.example`)
are read from the environment, e.g. `cp .env.example .env` and `uvicorn backend.app.main:app --env-file .env`.

### Web dashboard
Two ways to run it (the API from step 3 must be running or be started by option B).

**A) Development mode** (hot reload, proxy `/api` to the backend):
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (backend: uvicorn ... --reload on port 8000)
```

**B) Single command for demos** (the backend serves the built dashboard):
```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app          # dashboard: http://127.0.0.1:8000/   API docs: /docs
```

| Page | What it does |
|---|---|
| Dashboard | live stream (simulator or replay of the test set), alerts, score chart with threshold, TP/FP/FN counters |
| Models | list with metrics, activate a model, retrain with custom hyperparameters |
| Metrics | PR and ROC curves, confusion matrix, model comparison |
| Prediction | score a single transaction |
| Sandbox | simulate data, import your own CSV, build a personal profile model |
| History | statistics and prediction history from the database, clear history |

Notes:
- The dashboard needs trained models (step 2). Without them it shows a hint instead of data.
- Source `replay` needs `data/raw/creditcard.csv` on the server. Frauds are about 0.1% of the data, so the dashboard offers a
  demo option that enriches the stream with real frauds from the test set (clearly marked as a demo setting).
- Models trained on the Kaggle data use PCA features (V1-V28) and score only data in that format. Own or bank transactions use
  the `generic_tx` schema (amount and time features), e.g. via the personal profile model in the Sandbox.
- Dashboard tests: `cd frontend && npm test`. Another API address: set `VITE_API_URL`. Custom build directory: `FRONTEND_DIST`.
- The prototype has no authentication; `DELETE /api/transactions` clears the history.

### Experiments for the thesis
```bash
python -m anomaly.experiments --data data/raw/creditcard.csv --out results --what all
# parts: main | niu | imbalance | rolling   (see docs/experiments.md)
python -m anomaly.experiments --replot --out results    # rebuild figures/tables from saved CSV/NPZ (no retraining)
```
Outputs: `results/tables/*.tex`, `results/figures/*.pdf|png`, `results/csv/*.csv`.

### Main API endpoints
| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | score one transaction |
| `GET /api/models`, `POST /api/models/activate`, `POST /api/models/retrain` | model management |
| `GET /api/metrics`, `GET /api/metrics/{name}` | metrics and curves |
| `GET /api/transactions/recent`, `GET /api/stats`, `DELETE /api/transactions` | history and statistics |
| `GET /api/stream/next` | next transactions from a source (`source`, `model`, `n`, `fraud_rate`) |
| `GET /api/sources`, `POST /api/sandbox/simulate`, `POST /api/sandbox/upload`, `POST /api/sandbox/profile/train` | sandbox |

### Troubleshooting
- `numpy.dtype size changed` when importing pandas: use a virtual environment (`python3 -m venv .venv`), not user-level packages.
- Dashboard shows "no connection to API": start the backend; in dev mode it must listen on port 8000.
- "No trained models": run `python -m anomaly.train --models all`.
- Source `replay` disabled: put the Kaggle file in `data/raw/creditcard.csv` (or set `DATA_PATH`).

### Methodological notes
- Chronological split (70/15/15) instead of a random one; the threshold is selected on the validation set.
- The Kaggle dataset covers two days and a few hundred frauds (52 in the test set), so confidence intervals are wide.
  Results are academic and not production-ready.

### Documentation
`docs/architecture.md`, `docs/experiments.md`, `docs/bank_api.md`, `docs/pojecia-do-nauki.md` (Polish), `frontend/README.md`.

### Status
- [x] ML pipeline, API, tests, sandbox, experiment runner
- [x] Web dashboard (live stream, models, metrics, prediction, sandbox, history)
- [x] First experiments on the full dataset
- [ ] LOF diagnosis, paired bootstrap, optional One-Class SVM / autoencoder, hyperparameter tuning
- [ ] Model versioning in the database, authentication
- [ ] Bank connector proof of concept

### License
No license yet (all rights reserved). Academic project.

---

## Polski

Projekt pracy inżynierskiej (AGH, Teleinformatyka): webowy system do wykrywania anomalii (oszustw) w transakcjach
płatniczych wraz z porównawczą analizą algorytmów uczenia nadzorowanego i nienadzorowanego na silnie niezbalansowanych danych.

**Stack:** Python, scikit-learn, XGBoost, FastAPI, SQLAlchemy (SQLite, docelowo PostgreSQL), React (Vite, Recharts).

### Funkcje
- **Modele:** regresja logistyczna (baseline), Random Forest, XGBoost, Isolation Forest, Local Outlier Factor.
- **Ewaluacja:** podział czasowy train/walidacja/test, próg decyzyjny wybierany na walidacji, miary dla klas niezbalansowanych
  (PR-AUC, ROC AUC, precision, recall, F1, macierz pomyłek), wiele seedów, przedziały ufności (bootstrap).
- **API (FastAPI):** ocena transakcji, zarządzanie modelami (aktywacja, ponowny trening), metryki, historia i statystyki.
- **Panel webowy (React):** strumień transakcji na żywo z alertami, zarządzanie modelami, metryki i krzywe, ocena pojedynczej
  transakcji, sandbox (symulacja, import CSV, model profilu użytkownika), historia. Modułowy: strona to jeden plik i jedna linia w rejestrze.
- **Modułowe źródła transakcji:** symulator, odtwarzanie zbioru testowego, import CSV, szkielet konektora bankowego (mapowanie w stylu PSD2).
  Modele mają zadeklarowany schemat cech, więc oceniają tylko zgodne dane.
- **Skrypt eksperymentów:** eksport tabel LaTeX, wykresów PDF/PNG i plików CSV do pracy.

### Struktura projektu
```
anomaly/      kod ML: dane, modele (rejestr), metryki, trening, eksperymenty, raporty
backend/app/  aplikacja FastAPI: routery, serwis oceny, konektory, magazyn modeli, baza danych
frontend/     panel React (zob. frontend/README.md)
notebooks/    EDA i eksperymenty
tests/        testy pytest
docs/         architektura, eksperymenty, API bankowe, pojęcia do nauki, praca (LaTeX)
data/raw/     dane (poza gitem)          models/   wytrenowane modele (poza gitem)
```

### Szybki start
Wymagania: Python 3.10+, Node.js 18+ (tylko do panelu).
```bash
git clone <adres-repo> && cd payment-anomaly-detection
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Dane: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud) -> data/raw/creditcard.csv
#    Brak danych? python -m anomaly.synthetic data/raw/creditcard.csv   (syntetyczne, tylko do testów)

# 2) Trening wszystkich modeli (zapis w models/)
python -m anomaly.train --models all

# 3) Uruchomienie API (Swagger: http://127.0.0.1:8000/docs)
uvicorn backend.app.main:app --reload

# 4) Testy
pytest
```
Domyślne ustawienia działają bez konfiguracji. Opcjonalne zmienne (`MODEL_DIR`, `DATA_PATH`, `DATABASE_URL`, zob. `.env.example`)
są czytane ze środowiska, np. `cp .env.example .env` i `uvicorn backend.app.main:app --env-file .env`.

### Panel webowy
Dwa sposoby uruchomienia (API z kroku 3 musi działać albo zostanie uruchomione w wariancie B).

**A) Tryb deweloperski** (przeładowanie na żywo, proxy `/api` do backendu):
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (backend: uvicorn ... --reload na porcie 8000)
```

**B) Jedna komenda do demonstracji** (backend serwuje zbudowany panel):
```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app          # panel: http://127.0.0.1:8000/   dokumentacja API: /docs
```

| Strona | Co robi |
|---|---|
| Dashboard | strumień na żywo (symulator lub odtwarzanie zbioru testowego), alerty, wykres wyniku z progiem, liczniki TP/FP/FN |
| Modele | lista z metrykami, aktywacja modelu, ponowny trening z własnymi hiperparametrami |
| Metryki | krzywe PR i ROC, macierz pomyłek, porównanie modeli |
| Predykcja | ocena pojedynczej transakcji |
| Sandbox | symulacja danych, import własnego CSV, budowa modelu profilu użytkownika |
| Historia | statystyki i historia ocen z bazy, czyszczenie historii |

Uwagi:
- Panel wymaga wytrenowanych modeli (krok 2). Bez nich pokazuje podpowiedź zamiast danych.
- Źródło `replay` wymaga pliku `data/raw/creditcard.csv` na serwerze. Oszustw jest w danych ok. 0,1%, więc panel ma opcję demo
  wzbogacającą strumień o prawdziwe oszustwa ze zbioru testowego (oznaczoną jako ustawienie demonstracyjne).
- Modele uczone na zbiorze Kaggle używają cech po PCA (V1-V28) i oceniają tylko dane w tym formacie. Własne i bankowe transakcje
  używają schematu `generic_tx` (cechy z kwoty i czasu), np. przez model profilu użytkownika w Sandboxie.
- Testy panelu: `cd frontend && npm test`. Inny adres API: zmienna `VITE_API_URL`. Własny katalog z buildem: `FRONTEND_DIST`.
- Prototyp nie ma uwierzytelniania; `DELETE /api/transactions` czyści historię.

### Eksperymenty do pracy
```bash
python -m anomaly.experiments --data data/raw/creditcard.csv --out results --what all
# części: main | niu | imbalance | rolling   (zob. docs/experiments.md)
python -m anomaly.experiments --replot --out results    # odtworzenie wykresów i tabel z zapisanych CSV/NPZ (bez treningu)
```
Wyniki: `results/tables/*.tex`, `results/figures/*.pdf|png`, `results/csv/*.csv`.

### Najważniejsze endpointy API
| Endpoint | Zastosowanie |
|---|---|
| `POST /api/predict` | ocena pojedynczej transakcji |
| `GET /api/models`, `POST /api/models/activate`, `POST /api/models/retrain` | zarządzanie modelami |
| `GET /api/metrics`, `GET /api/metrics/{name}` | metryki i krzywe |
| `GET /api/transactions/recent`, `GET /api/stats`, `DELETE /api/transactions` | historia i statystyki |
| `GET /api/stream/next` | kolejne transakcje ze źródła (`source`, `model`, `n`, `fraud_rate`) |
| `GET /api/sources`, `POST /api/sandbox/simulate`, `POST /api/sandbox/upload`, `POST /api/sandbox/profile/train` | sandbox |

### Rozwiązywanie problemów
- `numpy.dtype size changed` przy imporcie pandas: użyj środowiska wirtualnego (`python3 -m venv .venv`), a nie pakietów z `~/.local`.
- Panel pokazuje „brak połączenia z API”: uruchom backend; w trybie deweloperskim musi działać na porcie 8000.
- „Brak wytrenowanych modeli”: uruchom `python -m anomaly.train --models all`.
- Źródło `replay` nieaktywne: umieść plik Kaggle w `data/raw/creditcard.csv` (albo ustaw `DATA_PATH`).

### Uwagi metodyczne
- Podział czasowy (70/15/15) zamiast losowego; próg wybierany na zbiorze walidacyjnym.
- Zbiór Kaggle obejmuje dwa dni i kilkaset oszustw (52 w zbiorze testowym), więc przedziały ufności są szerokie.
  Wyniki mają charakter akademicki i nie nadają się do zastosowań produkcyjnych.

### Dokumentacja
`docs/architecture.md`, `docs/experiments.md`, `docs/bank_api.md`, `docs/pojecia-do-nauki.md`, `frontend/README.md`.

### Status
- [x] Pipeline ML, API, testy, sandbox, skrypt eksperymentów
- [x] Panel webowy (strumień na żywo, modele, metryki, predykcja, sandbox, historia)
- [x] Pierwsze eksperymenty na pełnym zbiorze
- [ ] Diagnoza LOF, bootstrap parami, opcjonalnie One-Class SVM i autoenkoder, strojenie hiperparametrów
- [ ] Wersjonowanie modeli w bazie, uwierzytelnianie
- [ ] PoC konektora bankowego

### Licencja
Na razie brak licencji (wszelkie prawa zastrzeżone). Projekt akademicki.
