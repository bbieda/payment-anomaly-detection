import { useEffect, useRef, useState } from 'react';
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api.js';
import { Badge, Card, ErrorBox, Field, ModelSelect, Stat, Table, fmt, pct } from '../ui.jsx';
import { useModels } from '../store.jsx';

const MAX_KEEP = 300;
const MAX_CHART = 120;

export default function Dashboard() {
  const { models, active } = useModels();
  const [source, setSource] = useState('simulated');
  const [model, setModel] = useState('');
  const [fraudRate, setFraudRate] = useState('');
  const [every, setEvery] = useState(1500);
  const [running, setRunning] = useState(false);
  const [items, setItems] = useState([]);
  const [error, setError] = useState(null);
  const [sources, setSources] = useState([]);
  const busy = useRef(false);
  const seq = useRef(0);
  const modelName = model || active;

  useEffect(() => { api.sources().then(setSources).catch(() => {}); }, []);

  useEffect(() => {
    if (!running) return undefined;
    const tick = async () => {
      if (busy.current) return;
      busy.current = true;
      try {
        const res = await api.streamNext(source, modelName, 1, fraudRate === '' ? undefined : fraudRate);
        setItems((prev) => [...prev, ...res.map((r) => ({ ...r, seq: seq.current++ }))].slice(-MAX_KEEP));
        setError(null);
      } catch (e) {
        setError(e.message);
        setRunning(false);
      } finally {
        busy.current = false;
      }
    };
    tick();
    const id = setInterval(tick, every);
    return () => clearInterval(id);
  }, [running, source, modelName, every, fraudRate]);

  const alerts = items.filter((r) => r.is_anomaly);
  const hasLabels = items.some((r) => r.label != null);
  const tp = items.filter((r) => r.is_anomaly && r.label === 1).length;
  const fp = items.filter((r) => r.is_anomaly && r.label === 0).length;
  const fn = items.filter((r) => !r.is_anomaly && r.label === 1).length;
  const avgLat = items.length ? items.reduce((a, r) => a + r.latency_ms, 0) / items.length : null;
  const threshold = items.length ? items[items.length - 1].threshold : null;
  const chart = items.slice(-MAX_CHART).map((r) => ({ x: r.seq, y: r.score, anomaly: r.is_anomaly }));
  const replayOk = sources.find((s) => s.name === 'replay')?.available;

  return (
    <>
      <Card title="Strumień transakcji (na żywo)" actions={
        <>
          <button className="btn" onClick={() => setRunning((v) => !v)} disabled={!modelName}>{running ? 'Zatrzymaj' : 'Start'}</button>
          <button className="btn ghost" onClick={() => setItems([])}>Wyczyść widok</button>
        </>}>
        <div className="row">
          <Field label="Źródło">
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="simulated">symulator</option>
              <option value="replay" disabled={sources.length > 0 && !replayOk}>odtwarzanie zbioru testowego</option>
            </select>
          </Field>
          <Field label="Model"><ModelSelect models={models} value={model} onChange={setModel} /></Field>
          <Field label="Wzbogacenie o oszustwa (demo)">
            <select value={fraudRate} onChange={(e) => setFraudRate(e.target.value)}>
              <option value="">domyślne</option><option value="0.05">5%</option><option value="0.2">20%</option>
            </select>
          </Field>
          <Field label="Odświeżanie">
            <select value={every} onChange={(e) => setEvery(Number(e.target.value))}>
              <option value={700}>0,7 s</option><option value={1500}>1,5 s</option><option value={3000}>3 s</option>
            </select>
          </Field>
        </div>
        <p className="muted">Strumień jest symulowany (generowany lub odtwarzany ze zbioru). Wzbogacenie zwiększa odsetek oszustw tylko na potrzeby demonstracji, bo w danych naturalnie jest ich ok. 0,1%.</p>
        <ErrorBox error={error} />
        {!models.length && <p className="muted">Brak modeli. Uruchom trening: <code>python -m anomaly.train --models all</code>.</p>}
      </Card>

      <div className="stats">
        <Stat label="Przetworzone" value={items.length} />
        <Stat label="Alerty" value={alerts.length} hint={items.length ? pct(alerts.length / items.length) : undefined} />
        <Stat label="Średnie opóźnienie" value={avgLat == null ? '–' : `${fmt(avgLat, 1)} ms`} />
        {hasLabels && <Stat label="Trafione (TP)" value={tp} />}
        {hasLabels && <Stat label="Fałszywe alarmy (FP)" value={fp} />}
        {hasLabels && <Stat label="Przeoczone (FN)" value={fn} />}
      </div>

      <Card title="Wynik modelu w czasie">
        {chart.length ? (
          <div className="chart">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#eee" />
                <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} name="transakcja" tick={{ fontSize: 11 }} />
                <YAxis dataKey="y" type="number" name="wynik" tick={{ fontSize: 11 }}
                  domain={[(min) => Math.min(min, threshold ?? min), (max) => Math.max(max, threshold ?? max)]} tickFormatter={(v) => fmt(v, 2)} />
                <Tooltip formatter={(v) => fmt(v)} />
                {threshold != null && <ReferenceLine y={threshold} stroke="#b00020" strokeDasharray="4 4" label={{ value: 'próg', fontSize: 11, fill: '#b00020' }} />}
                <Scatter data={chart.filter((d) => !d.anomaly)} fill="#888" isAnimationActive={false} />
                <Scatter data={chart.filter((d) => d.anomaly)} fill="#b00020" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ) : <p className="muted">Uruchom strumień przyciskiem Start.</p>}
      </Card>

      <Card title="Ostatnie transakcje">
        <Table
          rows={[...items].reverse().slice(0, 15)}
          empty="Brak transakcji."
          rowClass={(r) => (r.is_anomaly ? 'alert' : '')}
          columns={[
            { key: 'id', header: 'ID' },
            { key: 'amount', header: 'Kwota', align: 'right', render: (r) => fmt(r.amount, 2) },
            { key: 'merchant', header: 'Sprzedawca', render: (r) => r.merchant ?? '–' },
            { key: 'score', header: 'Wynik', align: 'right', render: (r) => fmt(r.score) },
            { key: 'status', header: 'Status', render: (r) => (r.is_anomaly ? <Badge tone="alert">ALERT</Badge> : <Badge>OK</Badge>) },
            { key: 'label', header: 'Etykieta', render: (r) => (r.label == null ? '–' : r.label ? 'oszustwo' : 'normalna') },
            { key: 'lat', header: 'Opóźnienie', align: 'right', render: (r) => `${fmt(r.latency_ms, 1)} ms` },
          ]}
        />
      </Card>
    </>
  );
}
