# Frontend (React + Vite)

Inicjalizacja (raz, w katalogu głównym repo; wymaga Node.js 20+):

```bash
npm create vite@latest frontend -- --template react   # przy pytaniu o istniejący katalog: zachowaj README
cd frontend
npm install
npm install recharts
```

Proxy do API w `vite.config.js` (unika problemów z CORS w trakcie developmentu):

```js
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
```

Widoki do zrobienia (kolejność):
1. **Modele** – `GET /api/models`, przełączanie aktywnego (`POST /api/models/activate`).
2. **Metryki** – `GET /api/metrics` (słupki: PR-AUC, F1...) i `GET /api/metrics/{name}` (krzywe ROC/PR, macierz pomyłek).
3. **Predykcja** – formularz/przycisk wysyłający transakcję do `POST /api/predict`.
4. **Dashboard** – `GET /api/transactions/recent` (polling co 2 s), anomalie na czerwono.
5. **Retrain** – `POST /api/models/retrain` + `GET /api/models/retrain/status`.
