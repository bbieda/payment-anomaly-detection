import { useCallback, useEffect, useState } from 'react';
import { api } from '../api.js';
import { Badge, Card, ErrorBox, Field, Stat, Table, fmt, pct, time } from '../ui.jsx';

const SOURCES = ['', 'api', 'simulated', 'replay', 'sandbox_sim', 'sandbox_csv'];

export default function History() {
  const [source, setSource] = useState('');
  const [limit, setLimit] = useState(50);
  const [rows, setRows] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [r, s] = await Promise.all([api.recent(limit, source || undefined), api.stats()]);
      setRows(r); setStats(s); setError(null);
    } catch (e) { setError(e.message); }
  }, [limit, source]);
  useEffect(() => { load(); }, [load]);

  const clear = async () => {
    if (!window.confirm('Usunąć całą historię ocen z bazy?')) return;
    try { await api.clearHistory(); await load(); } catch (e) { setError(e.message); }
  };

  return (
    <>
      <Card title="Statystyki (z bazy danych)" actions={<button className="btn ghost" onClick={load}>Odśwież</button>}>
        <ErrorBox error={error} />
        {stats && (
          <>
            <div className="stats">
              <Stat label="Ocenione transakcje" value={stats.total} />
              <Stat label="Alerty" value={stats.anomalies} hint={stats.total ? pct(stats.anomalies / stats.total) : undefined} />
              <Stat label="Średnie opóźnienie" value={stats.avg_latency_ms == null ? '–' : `${fmt(stats.avg_latency_ms, 1)} ms`} />
            </div>
            <Table rows={stats.by_model} empty="Brak danych." columns={[
              { key: 'model', header: 'Model' },
              { key: 'count', header: 'Transakcje', align: 'right' },
              { key: 'anomalies', header: 'Alerty', align: 'right' },
              { key: 'lat', header: 'Śr. opóźnienie', align: 'right', render: (r) => `${fmt(r.avg_latency_ms, 1)} ms` },
            ]} />
          </>
        )}
      </Card>
      <Card title="Historia ocen" actions={
        <>
          <Field label="Źródło"><select value={source} onChange={(e) => setSource(e.target.value)}>{SOURCES.map((s) => <option key={s} value={s}>{s || 'wszystkie'}</option>)}</select></Field>
          <Field label="Limit"><select value={limit} onChange={(e) => setLimit(Number(e.target.value))}><option>25</option><option>50</option><option>100</option></select></Field>
          <button className="btn ghost" onClick={clear}>Wyczyść historię</button>
        </>}>
        <Table rows={rows} empty="Brak zapisanych ocen." rowClass={(r) => (r.is_anomaly ? 'alert' : '')} columns={[
          { key: 'created_at', header: 'Zapisano', render: (r) => time(r.created_at) },
          { key: 'source', header: 'Źródło' },
          { key: 'model', header: 'Model' },
          { key: 'amount', header: 'Kwota', align: 'right', render: (r) => fmt(r.amount, 2) },
          { key: 'score', header: 'Wynik', align: 'right', render: (r) => fmt(r.score) },
          { key: 'status', header: 'Status', render: (r) => (r.is_anomaly ? <Badge tone="alert">ALERT</Badge> : <Badge>OK</Badge>) },
          { key: 'label', header: 'Etykieta', render: (r) => (r.label == null ? '–' : r.label ? 'oszustwo' : 'normalna') },
        ]} />
      </Card>
    </>
  );
}
