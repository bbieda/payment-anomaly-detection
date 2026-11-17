// Małe, wielokrotnego użytku elementy interfejsu.
export const fmt = (x, d = 3) => (x == null || Number.isNaN(Number(x)) ? '–' : Number(x).toFixed(d).replace('.', ','));
export const pct = (x, d = 1) => (x == null ? '–' : `${(Number(x) * 100).toFixed(d).replace('.', ',')}%`);
export const time = (iso) => (iso ? new Date(iso).toLocaleString('pl-PL') : '–');

export function Card({ title, actions, children }) {
  return (
    <section className="card">
      {(title || actions) && (
        <header className="card-head"><h2>{title}</h2><div className="row">{actions}</div></header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }) {
  return <div className="stat"><div className="stat-value">{value}</div><div className="stat-label">{label}</div>{hint && <div className="stat-hint">{hint}</div>}</div>;
}

export function Field({ label, children }) {
  return <label className="field"><span>{label}</span>{children}</label>;
}

export function Badge({ children, tone = 'neutral' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function ErrorBox({ error }) {
  return error ? <div className="error" role="alert">{error}</div> : null;
}

export function Table({ columns, rows, empty = 'Brak danych', rowClass }) {
  if (!rows?.length) return <p className="muted">{empty}</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{columns.map((c) => <th key={c.key} className={c.align}>{c.header}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.id ?? i} className={rowClass?.(r) ?? ''}>
              {columns.map((c) => <td key={c.key} className={c.align}>{c.render ? c.render(r) : r[c.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ModelSelect({ models, value, onChange, filter }) {
  const list = filter ? models.filter(filter) : models;
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">(aktywny)</option>
      {list.map((m) => <option key={m.name} value={m.name}>{m.name} ({m.schema})</option>)}
    </select>
  );
}

// Wynik oceny wielu transakcji (symulacja, import CSV).
export function ResultSummary({ data }) {
  if (!data) return null;
  const m = data.metrics;
  return (
    <div>
      <div className="stats">
        <Stat label="Transakcje" value={data.n} />
        <Stat label="Oflagowane" value={data.n_flagged} hint={pct(data.flag_rate)} />
        {m && <Stat label="PR-AUC" value={fmt(m.pr_auc)} />}
        {m && <Stat label="Recall" value={fmt(m.recall)} />}
        {m && <Stat label="Precision" value={fmt(m.precision)} />}
      </div>
      {!m && <p className="muted">Brak etykiet w danych, więc metryki nie są liczone (pokazano tylko liczbę alertów).</p>}
      <h3>Najwyższe wyniki</h3>
      <Table
        rows={data.results.slice(0, 10)}
        rowClass={(r) => (r.is_anomaly ? 'alert' : '')}
        columns={[
          { key: 'id', header: 'ID' },
          { key: 'amount', header: 'Kwota', align: 'right', render: (r) => fmt(r.amount, 2) },
          { key: 'merchant', header: 'Sprzedawca', render: (r) => r.merchant ?? '–' },
          { key: 'score', header: 'Wynik', align: 'right', render: (r) => fmt(r.score) },
          { key: 'status', header: 'Status', render: (r) => (r.is_anomaly ? <Badge tone="alert">ALERT</Badge> : <Badge>OK</Badge>) },
          { key: 'label', header: 'Etykieta', render: (r) => (r.label == null ? '–' : r.label ? 'oszustwo' : 'normalna') },
        ]}
      />
    </div>
  );
}
