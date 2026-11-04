# payment-anomaly-detection

[English](#english) | [Polski](#polski)

---

## English

Bachelor's thesis project (AGH University, Teleinformatics): a web-based system for detecting anomalies
(fraud) in payment transactions, with a comparative analysis of supervised and unsupervised machine learning
algorithms on a highly imbalanced dataset.

### Features
- **Models:** logistic regression (baseline), Random Forest, XGBoost, Isolation Forest, Local Outlier Factor.
- **Evaluation:** chronological train/validation/test split, decision threshold chosen on validation data,
  metrics for imbalanced classes (PR-AUC, ROC AUC, precision, recall, F1, confusion matrix), multiple seeds,
  bootstrap confidence intervals.
- **API (FastAPI):** scoring, model management (activate, retrain), metrics, prediction history.
- **Modular transaction sources:** simulator, replay of the test set, CSV upload, bank connector skeleton
  (PSD2-style mapping). Models declare a feature schema, so only compatible data can be scored.
- **Sandbox:** upload your own transactions, simulate data, train a personal profile model (Isolation Forest).
- **Experiment runner:** exports LaTeX tables, PDF/PNG figures and CSV files for the thesis.

### Project structure
```
anomaly/      ML code: data, models (registry), metrics, training, experiments, reports
backend/app/  FastAPI app: routers, scoring service, connectors, model store, database
frontend/     React dashboard (planned, see frontend/README.md)
notebooks/    EDA and experiments
tests/        pytest suite
docs/         architecture, experiments, bank API options, concepts to learn, thesis LaTeX
data/raw/     datasets (not in git)      models/   trained models (not in git)
```

### Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 1) Data: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud) -> data/raw/creditcard.csv
#    No data yet? python -m anomaly.synthetic data/raw/creditcard.csv   (synthetic, for testing only)

# 2) Train all models (saved to models/)
python -m anomaly.train --models all

# 3) Run the API (Swagger UI: http://127.0.0.1:8000/docs)
uvicorn backend.app.main:app --reload

# 4) Tests
pytest
```

### Experiments for the thesis
```bash
python -m anomaly.experiments --data data/raw/creditcard.csv --out results --what all
# parts: main | niu | imbalance | rolling   (see docs/experiments.md)
```
Outputs: `results/tables/*.tex`, `results/figures/*.pdf|png`, `results/csv/*.csv`.

### Main API endpoints
| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | score one transaction |
| `GET /api/models`, `POST /api/models/activate`, `POST /api/models/retrain` | model management |
| `GET /api/metrics`, `GET /api/metrics/{name}` | metrics and curves |
| `GET /api/transactions/recent` | prediction history |
| `GET /api/stream/next` | next transactions from a source (for the dashboard) |
| `GET /api/sources`, `POST /api/sandbox/simulate`, `POST /api/sandbox/upload`, `POST /api/sandbox/profile/train` | sandbox |

### Methodological notes
- Chronological split (70/15/15) instead of a random one; the threshold is selected on the validation set.
- The Kaggle dataset has PCA-transformed features (V1–V28), so models trained on it cannot score real bank
  transactions. Real or uploaded transactions use the `generic_tx` schema (amount and time features).
- The Kaggle dataset covers two days and a few hundred frauds, so confidence intervals are wide. Results are
  academic and not production-ready.

### Documentation
`docs/architecture.md`, `docs/experiments.md`, `docs/bank_api.md`, `docs/pojecia-do-nauki.md` (Polish).

### Status
- [x] ML pipeline, API, tests, sandbox, experiment runner
- [ ] Experiments on the full dataset and hyperparameter tuning
- [ ] React dashboard
- [ ] Bank connector proof of concept

### License
No license yet (all rights reserved). Academic project.

---

## Polski

Projekt pracy inżynierskiej (AGH, Teleinformatyka): webowy system do wykrywania anomalii (oszustw) w transakcjach
płatniczych wraz z porównawczą analizą algorytmów uczenia nadzorowanego i nienadzorowanego na silnie
niezbalansowanych danych.

### Funkcje
- **Modele:** regresja logistyczna (baseline), Random Forest, XGBoost, Isolation Forest, Local Outlier Factor.
- **Ewaluacja:** podział czasowy train/walidacja/test, próg decyzyjny wybierany na walidacji, miary dla klas
  niezbalansowanych (PR-AUC, ROC AUC, precision, recall, F1, macierz pomyłek), wiele seedów, przedziały ufności
  (bootstrap).
- **API (FastAPI):** ocena transakcji, zarządzanie modelami (aktywacja, ponowny trening), metryki, historia ocen.
- **Modułowe źródła transakcji:** symulator, odtwarzanie zbioru testowego, import CSV, szkielet konektora
  bankowego (mapowanie w stylu PSD2). Modele mają zadeklarowany schemat cech, więc oceniają tylko zgodne dane.
- **Sandbox:** własne transakcje z pliku, symulacja danych, model profilu użytkownika (Isolation Forest).
- **Skrypt eksperymentów:** eksport tabel LaTeX, wykresów PDF/PNG i plików CSV do pracy.

### Struktura projektu
```
anomaly/      kod ML: dane, modele (rejestr), metryki, trening, eksperymenty, raporty
backend/app/  aplikacja FastAPI: routery, serwis oceny, konektory, magazyn modeli, baza danych
frontend/     panel React (planowany, zob. frontend/README.md)
notebooks/    EDA i eksperymenty
tests/        testy pytest
docs/         architektura, eksperymenty, API bankowe, pojęcia do nauki, LaTeX pracy
data/raw/     dane (poza gitem)          models/   wytrenowane modele (poza gitem)
```

### Szybki start
```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 1) Dane: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud) -> data/raw/creditcard.csv
#    Brak danych? python -m anomaly.synthetic data/raw/creditcard.csv   (syntetyczne, tylko do testów)

# 2) Trening wszystkich modeli (zapis w models/)
python -m anomaly.train --models all

# 3) Uruchomienie API (Swagger: http://127.0.0.1:8000/docs)
uvicorn backend.app.main:app --reload

# 4) Testy
pytest
```

### Eksperymenty do pracy
```bash
python -m anomaly.experiments --data data/raw/creditcard.csv --out results --what all
# części: main | niu | imbalance | rolling   (zob. docs/experiments.md)
```
Wyniki: `results/tables/*.tex`, `results/figures/*.pdf|png`, `results/csv/*.csv`.

### Najważniejsze endpointy API
| Endpoint | Zastosowanie |
|---|---|
| `POST /api/predict` | ocena pojedynczej transakcji |
| `GET /api/models`, `POST /api/models/activate`, `POST /api/models/retrain` | zarządzanie modelami |
| `GET /api/metrics`, `GET /api/metrics/{name}` | metryki i krzywe |
| `GET /api/transactions/recent` | historia ocen |
| `GET /api/stream/next` | kolejne transakcje ze źródła (dla panelu) |
| `GET /api/sources`, `POST /api/sandbox/simulate`, `POST /api/sandbox/upload`, `POST /api/sandbox/profile/train` | sandbox |

### Uwagi metodyczne
- Podział czasowy (70/15/15) zamiast losowego; próg wybierany na zbiorze walidacyjnym.
- Zbiór Kaggle ma cechy po PCA (V1–V28), więc modele na nim uczone nie ocenią prawdziwych transakcji bankowych.
  Dane własne i bankowe używają schematu `generic_tx` (cechy z kwoty i czasu).
- Zbiór obejmuje dwa dni i kilkaset oszustw, więc przedziały ufności są szerokie. Wyniki mają charakter
  akademicki i nie nadają się do zastosowań produkcyjnych.

### Dokumentacja
`docs/architecture.md`, `docs/experiments.md`, `docs/bank_api.md`, `docs/pojecia-do-nauki.md`.

### Status
- [x] Pipeline ML, API, testy, sandbox, skrypt eksperymentów
- [ ] Eksperymenty na pełnym zbiorze i strojenie hiperparametrów
- [ ] Panel React
- [ ] PoC konektora bankowego

### Licencja
Na razie brak licencji (wszelkie prawa zastrzeżone). Projekt akademicki.