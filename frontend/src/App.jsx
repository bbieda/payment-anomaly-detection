import { useEffect, useState } from 'react';
import { ModelsProvider, useModels } from './store.jsx';
import { modules } from './modules/index.js';

const fromHash = () => modules.find((m) => `#/${m.id}` === window.location.hash)?.id ?? modules[0].id;

function Shell() {
  const [page, setPage] = useState(fromHash);
  const { online, active } = useModels();
  useEffect(() => {
    const on = () => setPage(fromHash());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const Current = modules.find((m) => m.id === page).component;
  return (
    <>
      <header className="app-header">
        <h1>Detekcja anomalii w transakcjach płatniczych</h1>
        <nav aria-label="Moduły">
          {modules.map((m) => (
            <button key={m.id} aria-current={m.id === page ? 'page' : undefined}
              onClick={() => { window.location.hash = `/${m.id}`; setPage(m.id); }}>{m.label}</button>
          ))}
        </nav>
        <div className="status">
          <span className={`dot ${online ? 'on' : online === false ? 'off' : ''}`} />
          {online === false ? 'brak połączenia z API' : `API: ${online ? 'ok' : '…'}`}
          {active && <> · aktywny model: <strong>{active}</strong></>}
        </div>
      </header>
      <main><Current /></main>
    </>
  );
}

export default function App() {
  return <ModelsProvider><Shell /></ModelsProvider>;
}
