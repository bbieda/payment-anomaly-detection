# Architektura (v0.2) i jak ją rozszerzać

```
źródła transakcji            ocena                         wyniki
connectors/                  services/scoring.py           routers/ (HTTP)  ->  frontend
  simulated  (sandbox)   ->  Transaction -> cechy (schemat) -> model -> score/próg  -> db.py (historia)
  replay     (zbiór Kaggle)           anomaly/schemas.py       store.py (joblib + json)
  csv_source (własny plik)
  bank       (PSD2, szkielet)
```

**Zasada:** reszta systemu zna tylko `Transaction` (connectors/base.py) i `FeatureSchema` (anomaly/schemas.py).

## Ważne ograniczenie: schematy cech
Modele ze zbioru Kaggle używają cech V1–V28 (PCA), których **nie ma** w prawdziwej transakcji bankowej.
Dlatego każdy model ma `schema` w metadanych (`kaggle_pca` albo `generic_tx`).
- `kaggle_pca`: wszystkie modele z porównania (zgodność z [2]); oceniają tylko dane w formacie Kaggle.
- `generic_tx`: cechy z kwoty i czasu (log kwoty, godzina, dzień tygodnia); model profilu użytkownika
  (`anomaly/profile.py`, Isolation Forest uczony na własnej historii) oraz przyszłe dane z banku.
Próba oceny transakcji innego schematu zwraca czytelny błąd 422.

## Jak dodać nowe źródło (np. bank)
1. Klasa dziedzicząca z `TransactionSource` w `connectors/` (metoda `fetch`) i mapowanie danych na `Transaction`
   (wzór: `from_psd2_item` w `bank.py`).
2. Rejestracja w `routers/predict.py` (`/api/stream/next`) i `routers/sandbox.py` (`/api/sources`).
3. Test z zamockowaną odpowiedzią API (bez sieci).

## Jak dodać nowy model
Wpis w `REGISTRY` (anomaly/models.py): nazwa, rodzaj, builder, parametry domyślne. Reszta (trening, metryki,
API, panel) działa bez zmian.

## Frontend (v0.3)
Panel React (`frontend/`) jest osobną warstwą nad API: `api.js` (klient), `store.jsx` (modele), `modules/` (strony) z rejestrem
`modules/index.js`. Backend może serwować zbudowany panel (`frontend/dist`) pod `/`, więc demo wymaga jednej komendy.
Dla źródeł i modeli panel jest generyczny: lista źródeł (`/api/sources`) i modeli (`/api/models`) pochodzi z backendu,
a schemat cech modelu (`kaggle_pca` / `generic_tx`) steruje tym, co panel pozwala ocenić. Nowe źródło (np. bank) pojawi się
w panelu po dodaniu konektora w backendzie bez zmian w kodzie stron.
