import { useState } from 'react';
import { api } from '../api.js';
import { Badge, Card, ErrorBox, Field, ModelSelect, Stat, fmt } from '../ui.jsx';
import { useModels } from '../store.jsx';

export default function Predict() {
  const { models, active } = useModels();
  const [model, setModel] = useState('');
  const [amount, setAmount] = useState('120.50');
  const [merchant, setMerchant] = useState('');
  const [when, setWhen] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [featJson, setFeatJson] = useState('');
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const name = model || active;
  const schema = models.find((m) => m.name === name)?.schema;
  const kaggle = schema === 'kaggle_pca';

  const submit = async () => {
    setError(null); setRes(null);
    const body = { amount: Number(amount), model: name || undefined };
    if (merchant) body.merchant = merchant;
    if (when) body.timestamp = new Date(when).toISOString();
    if (kaggle) {
      try {
        body.features = advanced && featJson ? JSON.parse(featJson) : Object.fromEntries([...Array(28)].map((_, i) => [`V${i + 1}`, 0]).concat([['Amount', Number(amount)]]));
      } catch { setError('Cechy muszą być poprawnym JSON-em.'); return; }
    }
    setBusy(true);
    try { setRes(await api.predict(body)); } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  return (
    <Card title="Ocena pojedynczej transakcji">
      <div className="row">
        <Field label="Model"><ModelSelect models={models} value={model} onChange={setModel} /></Field>
        <Field label="Kwota"><input type="number" step="0.01" value={amount} onChange={(e) => setAmount(e.target.value)} /></Field>
        <Field label="Sprzedawca (opcjonalnie)"><input value={merchant} onChange={(e) => setMerchant(e.target.value)} /></Field>
        <Field label="Data i czas (opcjonalnie)"><input type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} /></Field>
        <button className="btn" onClick={submit} disabled={busy || !name || amount === ''}>Oceń</button>
      </div>
      {kaggle && (
        <>
          <p className="muted">Model ma schemat <code>kaggle_pca</code>: wymaga cech V1–V28. Domyślnie użyto wartości 0 (transakcja „neutralna”), więc wynik ma charakter demonstracyjny.
            Dla własnych transakcji użyj modelu o schemacie <code>generic_tx</code> (zakładka Sandbox, „Profil użytkownika”).</p>
          <label className="muted"><input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} /> zaawansowane: własne cechy (JSON)</label>
          {advanced && <textarea rows={4} placeholder='{"V1": 0.1, ..., "V28": 0, "Amount": 120.5}' value={featJson} onChange={(e) => setFeatJson(e.target.value)} />}
        </>
      )}
      <ErrorBox error={error} />
      {res && (
        <div className="stats" style={{ marginTop: 12 }}>
          <Stat label="Decyzja" value={res.is_anomaly ? <Badge tone="alert">ALERT</Badge> : <Badge>OK</Badge>} />
          <Stat label="Wynik" value={fmt(res.score)} hint={`próg ${fmt(res.threshold)}`} />
          <Stat label="Ryzyko" value={res.risk === 'high' ? 'wysokie' : 'niskie'} />
          <Stat label="Opóźnienie" value={`${fmt(res.latency_ms, 1)} ms`} hint={res.model} />
        </div>
      )}
    </Card>
  );
}
