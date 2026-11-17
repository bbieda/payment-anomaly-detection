// Cienki klient API. Każdy endpoint backendu = jedna funkcja; reszta aplikacji nie zna adresów URL.
const BASE = import.meta.env.VITE_API_URL || '';

async function req(path, opts) {
  const r = await fetch(BASE + path, opts);
  if (!r.ok) {
    let detail;
    try { detail = (await r.json()).detail; } catch { /* brak treści */ }
    throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Błąd HTTP ${r.status}`);
  }
  return r.json();
}
const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const form = (fields) => { const f = new FormData(); Object.entries(fields).forEach(([k, v]) => v != null && f.append(k, v)); return { method: 'POST', body: f }; };
const qs = (o) => { const p = new URLSearchParams(); Object.entries(o).forEach(([k, v]) => v != null && v !== '' && p.set(k, v)); const s = p.toString(); return s ? `?${s}` : ''; };

export const api = {
  health: () => req('/api/health'),
  models: () => req('/api/models'),
  registry: () => req('/api/models/registry'),
  activate: (name) => req('/api/models/activate', json('POST', { name })),
  retrain: (body) => req('/api/models/retrain', json('POST', body)),
  retrainStatus: () => req('/api/models/retrain/status'),
  metrics: () => req('/api/metrics'),
  modelMetrics: (name) => req(`/api/metrics/${encodeURIComponent(name)}`),
  predict: (body) => req('/api/predict', json('POST', body)),
  recent: (limit = 50, source) => req(`/api/transactions/recent${qs({ limit, source })}`),
  clearHistory: () => req('/api/transactions', { method: 'DELETE' }),
  stats: () => req('/api/stats'),
  sources: () => req('/api/sources'),
  streamNext: (source, model, n = 1, fraudRate) => req(`/api/stream/next${qs({ source, model, n, fraud_rate: fraudRate })}`),
  simulate: (body) => req('/api/sandbox/simulate', json('POST', body)),
  upload: (file, model) => req('/api/sandbox/upload', form({ file, model })),
  trainProfile: (file, name) => req('/api/sandbox/profile/train', form({ file, name })),
};
