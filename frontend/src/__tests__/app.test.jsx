import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from '../App.jsx';

const metrics = { pr_auc: 0.76, roc_auc: 0.97, f1: 0.8, precision: 0.9, recall: 0.7, n: 1000, n_fraud: 5,
  confusion_matrix: { tn: 990, fp: 1, fn: 2, tp: 3 } };
const model = { name: 'xgboost', kind: 'supervised', schema: 'kaggle_pca', active: true, threshold: 0.5, params: {},
  trained_at: '2026-01-01T00:00:00Z', metrics };
const item = { id: 'sim-1', timestamp: '2026-01-01T00:00:00Z', amount: 12.5, merchant: 'Cafe', label: 1, model: 'xgboost',
  score: 0.9, threshold: 0.5, is_anomaly: true, latency_ms: 2 };

const routes = {
  '/api/health': { status: 'ok' },
  '/api/models': [model],
  '/api/models/registry': [{ name: 'xgboost', kind: 'supervised', defaults: { n_estimators: 300 } }],
  '/api/models/retrain/status': {},
  '/api/metrics': { xgboost: { kind: 'supervised', ...metrics, fit_seconds: 1, predict_latency_ms: 2 } },
  '/api/metrics/xgboost': { name: 'xgboost', threshold: 0.5, fit_seconds: 1, predict_latency_ms: 2, metrics,
    curves: { roc: { fpr: [0, 1], tpr: [0, 1] }, pr: { recall: [0, 1], precision: [1, 0.5] } } },
  '/api/sources': [{ name: 'simulated', available: true }, { name: 'replay', available: false }],
  '/api/stream/next': [item],
  '/api/transactions/recent': [{ id: 1, tx_id: 'sim-1', source: 'simulated', created_at: '2026-01-01T00:00:00Z', model: 'xgboost', score: 0.9, is_anomaly: true, amount: 12.5, merchant: null, label: 1 }],
  '/api/stats': { total: 1, anomalies: 1, avg_latency_ms: 2, by_source: [], by_model: [{ model: 'xgboost', count: 1, anomalies: 1, avg_latency_ms: 2 }] },
};

beforeEach(() => {
  globalThis.fetch = vi.fn(async (url) => {
    const path = String(url).split('?')[0];
    if (!(path in routes)) return { ok: false, status: 404, json: async () => ({ detail: `brak ${path}` }) };
    return { ok: true, status: 200, json: async () => routes[path] };
  });
});
afterEach(cleanup);

test('shows header, active model and navigates through all modules', async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText('xgboost', { selector: 'strong' })).toBeTruthy());
  for (const [label, heading] of [['Modele', 'Ponowny trening'], ['Metryki', 'Krzywa ROC'], ['Predykcja', 'Ocena pojedynczej transakcji'],
    ['Sandbox', 'Symulacja danych'], ['Historia', 'Historia ocen'], ['Dashboard', 'Strumień transakcji (na żywo)']]) {
    fireEvent.click(screen.getByRole('button', { name: label }));
    await waitFor(() => expect(screen.getByText(heading)).toBeTruthy());
  }
});

test('dashboard streams a transaction and shows an alert', async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText('xgboost', { selector: 'strong' })).toBeTruthy());
  fireEvent.click(screen.getByRole('button', { name: 'Start' }));
  await waitFor(() => expect(screen.getAllByText('ALERT').length).toBeGreaterThan(0));
  fireEvent.click(screen.getByRole('button', { name: 'Zatrzymaj' }));
});

test('predict shows decision', async () => {
  routes['/api/predict'] = { model: 'xgboost', score: 0.9, threshold: 0.5, is_anomaly: true, risk: 'high', latency_ms: 2, id: 'x' };
  render(<App />);
  await waitFor(() => expect(screen.getByText('xgboost', { selector: 'strong' })).toBeTruthy());
  fireEvent.click(screen.getByRole('button', { name: 'Predykcja' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Oceń' }));
  await waitFor(() => expect(screen.getByText('wysokie')).toBeTruthy());
});
