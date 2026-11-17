import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Badge, Card, ErrorBox, Field, ModelSelect, ResultSummary } from '../ui.jsx';
import { useModels } from '../store.jsx';

// Przykładowy CSV (format ogólny) do pobrania i wgrania z powrotem.
function sampleCsv() {
  const m = ['Grocery', 'Cafe', 'Transport', 'Online shop', 'Pharmacy', 'Restaurant'];
  const rows = ['timestamp,amount,merchant,label'];
  for (let i = 0; i < 150; i++) {
    const fraud = i % 37 === 36;
    const d = new Date(Date.UTC(2026, 0, 1 + (i % 50), fraud ? 2 : 8 + (i % 12), (i * 7) % 60));
    const amount = fraud ? 900 + (i % 5) * 120 : 8 + ((i * 53) % 140);
    rows.push(`${d.toISOString()},${amount.toFixed(2)},${m[i % m.length]},${fraud ? 1 : 0}`);
  }
  return rows.join('\n');
}
const download = () => {
  const url = URL.createObjectURL(new Blob([sampleCsv()], { type: 'text/csv' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: 'przykladowe_transakcje.csv' });
  a.click();
  URL.revokeObjectURL(url);
};

function useAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = async (fn) => { setBusy(true); setError(null); try { return await fn(); } catch (e) { setError(e.message); return null; } finally { setBusy(false); } };
  return { busy, error, run };
}

export default function Sandbox() {
  const { models, refresh } = useModels();
  const [sources, setSources] = useState([]);
  useEffect(() => { api.sources().then(setSources).catch(() => {}); }, []);

  const [simModel, setSimModel] = useState('');
  const [n, setN] = useState(300);
  const [rate, setRate] = useState(0.02);
  const [seed, setSeed] = useState(0);
  const sim = useAction();
  const [simRes, setSimRes] = useState(null);

  const [upModel, setUpModel] = useState('');
  const [file, setFile] = useState(null);
  const up = useAction();
  const [upRes, setUpRes] = useState(null);

  const [hist, setHist] = useState(null);
  const [pname, setPname] = useState('moj_profil');
  const prof = useAction();
  const [profRes, setProfRes] = useState(null);

  return (
    <>
      <Card title="Źródła transakcji">
        <div className="row">
          {sources.map((s) => <Badge key={s.name} tone={s.available ? 'active' : 'neutral'}>{s.name}{s.available ? '' : ' (niedostępne)'}</Badge>)}
        </div>
        <p className="muted">Każde źródło dostarcza ujednolicone transakcje; konektor bankowy (PSD2) jest przygotowany jako szkielet i zostanie dołączony w kolejnym etapie.</p>
      </Card>

      <Card title="Symulacja danych">
        <div className="row">
          <Field label="Model"><ModelSelect models={models} value={simModel} onChange={setSimModel} /></Field>
          <Field label="Liczba transakcji"><input type="number" min="1" max="2000" value={n} onChange={(e) => setN(Number(e.target.value))} /></Field>
          <Field label="Odsetek oszustw"><input type="number" step="0.01" min="0" max="0.5" value={rate} onChange={(e) => setRate(Number(e.target.value))} /></Field>
          <Field label="Seed"><input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></Field>
          <button className="btn" disabled={sim.busy || !models.length}
            onClick={async () => setSimRes(await sim.run(() => api.simulate({ n, fraud_rate: rate, seed, model: simModel || null })))}>Symuluj i oceń</button>
        </div>
        <ErrorBox error={sim.error} />
        <ResultSummary data={simRes} />
      </Card>

      <Card title="Import własnych transakcji (CSV)" actions={<button className="btn ghost" onClick={download}>Pobierz przykładowy CSV</button>}>
        <p className="muted">Kolumny: <code>timestamp</code> (lub <code>date</code>), <code>amount</code>; opcjonalnie <code>merchant</code>, <code>category</code>, <code>label</code>. Obsługiwany jest też format Kaggle (V1–V28, Amount).</p>
        <div className="row">
          <Field label="Model"><ModelSelect models={models} value={upModel} onChange={setUpModel} /></Field>
          <Field label="Plik CSV"><input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files[0] ?? null)} /></Field>
          <button className="btn" disabled={up.busy || !file || !models.length}
            onClick={async () => setUpRes(await up.run(() => api.upload(file, upModel || null)))}>Oceń plik</button>
        </div>
        <ErrorBox error={up.error} />
        <ResultSummary data={upRes} />
      </Card>

      <Card title="Profil użytkownika (model na własnej historii)">
        <p className="muted">Wgraj historię swoich transakcji (min. 50). Powstanie model Isolation Forest o schemacie <code>generic_tx</code>, którego możesz użyć w pozostałych zakładkach. Uczenie odbywa się bez etykiet.</p>
        <div className="row">
          <Field label="Nazwa modelu (a–z, 0–9, _)"><input value={pname} onChange={(e) => setPname(e.target.value)} /></Field>
          <Field label="Plik z historią (CSV)"><input type="file" accept=".csv,text/csv" onChange={(e) => setHist(e.target.files[0] ?? null)} /></Field>
          <button className="btn" disabled={prof.busy || !hist}
            onClick={async () => { const r = await prof.run(() => api.trainProfile(hist, pname)); if (r) { setProfRes(r); refresh(); } }}>Zbuduj profil</button>
        </div>
        <ErrorBox error={prof.error} />
        {profRes && <div className="ok">Model „{profRes.name}” gotowy ({profRes.n_train} transakcji, schemat {profRes.schema}). Jest dostępny na liście modeli.</div>}
      </Card>
    </>
  );
}
