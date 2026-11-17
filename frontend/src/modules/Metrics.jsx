import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api.js';
import { Card, ErrorBox, Field, Stat, fmt } from '../ui.jsx';
import { useModels } from '../store.jsx';

const zip = (xs, ys) => xs.map((x, i) => ({ x, y: ys[i] }));

function Curve({ data, xLabel, yLabel, baseline }) {
  return (
    <div className="chart">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 16, left: 0 }}>
          <CartesianGrid stroke="#eee" />
          <XAxis dataKey="x" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: xLabel, position: 'insideBottom', offset: -8, fontSize: 11 }} />
          <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <Tooltip formatter={(v) => fmt(v)} labelFormatter={(v) => fmt(v)} />
          {baseline != null && <ReferenceLine y={baseline} stroke="#999" strokeDasharray="4 4" />}
          <Line dataKey="y" stroke="#111" dot={false} strokeWidth={1.8} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Metrics() {
  const { models, active } = useModels();
  const withMetrics = models.filter((m) => m.metrics);
  const [sel, setSel] = useState('');
  const [detail, setDetail] = useState(null);
  const [all, setAll] = useState({});
  const [error, setError] = useState(null);
  const name = sel || (withMetrics.find((m) => m.name === active) ?? withMetrics[0])?.name;

  useEffect(() => { api.metrics().then(setAll).catch((e) => setError(e.message)); }, [models]);
  useEffect(() => {
    if (!name) return;
    api.modelMetrics(name).then((d) => { setDetail(d); setError(null); }).catch((e) => setError(e.message));
  }, [name]);

  const m = detail?.metrics;
  const cm = m?.confusion_matrix;
  const prevalence = m ? m.n_fraud / m.n : null;
  const compare = Object.entries(all).map(([n, v]) => ({ name: n, 'PR-AUC': v.pr_auc, 'ROC AUC': v.roc_auc }));

  if (!withMetrics.length) {
    return <Card title="Metryki"><p className="muted">Brak modeli z metrykami. Uruchom trening: <code>python -m anomaly.train --models all</code>.</p></Card>;
  }
  return (
    <>
      <Card title="Metryki modelu" actions={
        <Field label="Model">
          <select value={name} onChange={(e) => setSel(e.target.value)}>
            {withMetrics.map((x) => <option key={x.name} value={x.name}>{x.name}</option>)}
          </select>
        </Field>}>
        <ErrorBox error={error} />
        {m && (
          <div className="stats">
            <Stat label="PR-AUC" value={fmt(m.pr_auc)} hint={`poziom losowy ${fmt(prevalence, 4)}`} />
            <Stat label="ROC AUC" value={fmt(m.roc_auc)} />
            <Stat label="F1" value={fmt(m.f1)} />
            <Stat label="Precision" value={fmt(m.precision)} />
            <Stat label="Recall" value={fmt(m.recall)} />
            <Stat label="Oszustwa w teście" value={`${m.n_fraud}`} hint={`z ${m.n}`} />
          </div>
        )}
        {detail && <p className="muted">Próg: {fmt(detail.threshold)} (wybrany na zbiorze walidacyjnym). Trening: {fmt(detail.fit_seconds, 1)} s, opóźnienie oceny: {fmt(detail.predict_latency_ms, 1)} ms.</p>}
      </Card>

      <div className="grid2">
        <Card title="Krzywa precision–recall">
          {detail?.curves ? <Curve data={zip(detail.curves.pr.recall, detail.curves.pr.precision)} xLabel="Recall" yLabel="Precision" baseline={prevalence} /> : <p className="muted">Brak krzywych.</p>}
        </Card>
        <Card title="Krzywa ROC">
          {detail?.curves ? <Curve data={zip(detail.curves.roc.fpr, detail.curves.roc.tpr)} xLabel="FPR" yLabel="TPR" /> : <p className="muted">Brak krzywych.</p>}
        </Card>
      </div>

      <div className="grid2">
        <Card title="Macierz pomyłek (zbiór testowy)">
          {cm ? (
            <div className="cm">
              <div className="h" /><div className="h">przewidziano: norma</div><div className="h">przewidziano: oszustwo</div>
              <div className="h">rzeczywiście: norma</div><div className="hit">{cm.tn}</div><div>{cm.fp}</div>
              <div className="h">rzeczywiście: oszustwo</div><div>{cm.fn}</div><div className="hit">{cm.tp}</div>
            </div>
          ) : <p className="muted">Brak danych.</p>}
        </Card>
        <Card title="Porównanie modeli">
          <div className="chart">
            <ResponsiveContainer>
              <BarChart data={compare} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#eee" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => fmt(v)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="PR-AUC" fill="#111" isAnimationActive={false} />
                <Bar dataKey="ROC AUC" fill="#bbb" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="muted">ROC AUC bywa zbyt optymistyczne przy rzadkiej klasie; PR-AUC lepiej oddaje jakość wykrywania oszustw.</p>
        </Card>
      </div>
    </>
  );
}
