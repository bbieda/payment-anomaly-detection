import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Badge, Card, ErrorBox, Field, Table, fmt, time } from '../ui.jsx';
import { useModels } from '../store.jsx';

export default function Models() {
  const { models, refresh } = useModels();
  const [registry, setRegistry] = useState([]);
  const [name, setName] = useState('');
  const [params, setParams] = useState('{}');
  const [normalOnly, setNormalOnly] = useState(false);
  const [status, setStatus] = useState({});
  const [msg, setMsg] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.registry().then((r) => { setRegistry(r); if (r[0]) { setName(r[0].name); setParams(JSON.stringify(r[0].defaults, null, 2)); } }).catch((e) => setError(e.message));
  }, []);

  const pick = (n) => {
    setName(n);
    const spec = registry.find((r) => r.name === n);
    if (spec) setParams(JSON.stringify(spec.defaults, null, 2));
  };

  // Odpytuj status treningu, dopóki któryś model ma stan "running".
  useEffect(() => {
    if (!Object.values(status).includes('running')) return undefined;
    const id = setInterval(async () => {
      try {
        const s = await api.retrainStatus();
        setStatus(s);
        if (!Object.values(s).includes('running')) refresh();
      } catch (e) { setError(e.message); }
    }, 2000);
    return () => clearInterval(id);
  }, [status, refresh]);

  const activate = async (n) => {
    try { await api.activate(n); await refresh(); setError(null); } catch (e) { setError(e.message); }
  };

  const retrain = async () => {
    setMsg(null); setError(null);
    let p;
    try { p = JSON.parse(params || '{}'); } catch { setError('Parametry muszą być poprawnym JSON-em.'); return; }
    try {
      await api.retrain({ name, params: p, normal_only: normalOnly });
      setStatus(await api.retrainStatus());
      setMsg(`Trening modelu ${name} uruchomiony w tle.`);
    } catch (e) { setError(e.message); }
  };

  const sel = registry.find((r) => r.name === name);
  return (
    <>
      <Card title="Modele">
        <ErrorBox error={error} />
        <Table
          rows={models}
          empty="Brak wytrenowanych modeli."
          columns={[
            { key: 'name', header: 'Model', render: (m) => <strong>{m.name}</strong> },
            { key: 'kind', header: 'Rodzaj', render: (m) => (m.kind === 'supervised' ? 'nadzorowany' : 'nienadzorowany') },
            { key: 'schema', header: 'Schemat cech' },
            { key: 'pr', header: 'PR-AUC', align: 'right', render: (m) => fmt(m.metrics?.pr_auc) },
            { key: 'roc', header: 'ROC AUC', align: 'right', render: (m) => fmt(m.metrics?.roc_auc) },
            { key: 'f1', header: 'F1', align: 'right', render: (m) => fmt(m.metrics?.f1) },
            { key: 'thr', header: 'Próg', align: 'right', render: (m) => fmt(m.threshold) },
            { key: 'at', header: 'Wytrenowany', render: (m) => time(m.trained_at) },
            { key: 'act', header: '', render: (m) => (m.active ? <button className="btn is-active" disabled aria-pressed="true">Aktywny</button> : <button className="btn ghost" onClick={() => activate(m.name)}>Aktywuj</button>) },
          ]}
        />
        <p className="muted">Schemat <code>kaggle_pca</code> przyjmuje tylko dane w formacie zbioru Kaggle (V1–V28). Schemat <code>generic_tx</code> działa na dowolnych transakcjach (kwota, czas), np. z własnego pliku.</p>
      </Card>

      <Card title="Ponowny trening">
        <div className="row">
          <Field label="Model">
            <select value={name} onChange={(e) => pick(e.target.value)}>
              {registry.map((r) => <option key={r.name} value={r.name}>{r.name} ({r.kind === 'supervised' ? 'nadzorowany' : 'nienadzorowany'})</option>)}
            </select>
          </Field>
          {sel?.kind === 'unsupervised' && (
            <Field label="Uczenie tylko na normalnych">
              <input type="checkbox" checked={normalOnly} onChange={(e) => setNormalOnly(e.target.checked)} />
            </Field>
          )}
          <button className="btn" onClick={retrain} disabled={!name || status[name] === 'running'}>Trenuj</button>
        </div>
        <Field label="Hiperparametry (JSON)"><textarea rows={5} value={params} onChange={(e) => setParams(e.target.value)} /></Field>
        {msg && <div className="ok">{msg}</div>}
        {Object.entries(status).map(([n, s]) => <p key={n} className="muted">{n}: {s}</p>)}
        <p className="muted">Trening używa pliku danych wskazanego w <code>DATA_PATH</code> na serwerze i może trwać od kilku sekund (XGBoost) do kilkudziesięciu (Random Forest).</p>
      </Card>
    </>
  );
}
