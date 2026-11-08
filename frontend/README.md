# Frontend (React + Vite + Recharts)

Panel monitorujący do systemu detekcji anomalii. Komunikuje się z API FastAPI (`/api/...`).

## Uruchomienie

```bash
# 1) backend (z katalogu głównego repo; modele muszą być wytrenowane: python -m anomaly.train --models all)
uvicorn backend.app.main:app --reload            # http://127.0.0.1:8000  (Swagger: /docs)

# 2a) tryb deweloperski (proxy /api -> :8000), Node.js 18+
cd frontend && npm install && npm run dev        # http://localhost:5173

# 2b) albo jedna komenda: zbudowany panel serwuje sam backend
cd frontend && npm install && npm run build      # powstaje frontend/dist
uvicorn backend.app.main:app                     # panel pod http://127.0.0.1:8000/
```

Testy (smoke testy renderowania z zamockowanym API): `npm test`. Inny adres API: zmienna `VITE_API_URL`.

## Moduły (strony)
| Moduł | Co robi | Endpointy |
|---|---|---|
| Dashboard | strumień transakcji na żywo, alerty, wykres wyniku z progiem, liczniki TP/FP/FN | `/api/stream/next`, `/api/sources` |
| Modele | lista, aktywacja, ponowny trening z parametrami | `/api/models*` |
| Metryki | PR/ROC, macierz pomyłek, porównanie modeli | `/api/metrics*` |
| Predykcja | ocena pojedynczej transakcji | `/api/predict` |
| Sandbox | symulacja, import CSV, profil użytkownika, lista źródeł | `/api/sandbox/*`, `/api/sources` |
| Historia | statystyki i historia ocen z bazy, czyszczenie | `/api/stats`, `/api/transactions*` |

## Jak dodać nowy moduł
1. Plik `src/modules/MojModul.jsx` (komponent domyślny; dane przez `api` z `src/api.js`, modele przez `useModels()`).
2. Jedna linia w `src/modules/index.js`: `{ id: 'moj', label: 'Mój moduł', component: MojModul }`.
Nawigacja, routing (`#/moj`) i test nawigacji biorą listę z rejestru. Nowy endpoint backendu = nowa funkcja w `src/api.js`.

## Struktura
```
src/api.js        klient API (jedyne miejsce z adresami URL)
src/store.jsx     wspólny stan: modele, status serwera
src/ui.jsx        elementy wielokrotnego użytku (Card, Table, Stat, ...)
src/modules/      strony + rejestr index.js
src/__tests__/    smoke testy (vitest + jsdom)
```
