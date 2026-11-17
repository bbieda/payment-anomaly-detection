import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { api } from './api.js';

// Wspólny stan: lista modeli i status serwera. Moduły (strony) pobierają go przez useModels().
const Ctx = createContext(null);

export function ModelsProvider({ children }) {
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);
  const [online, setOnline] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setModels(await api.models());
      setError(null);
      setOnline(true);
    } catch (e) {
      setError(e.message);
      setOnline(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  const active = models.find((m) => m.active)?.name ?? '';
  return <Ctx.Provider value={{ models, active, error, online, refresh }}>{children}</Ctx.Provider>;
}

export const useModels = () => useContext(Ctx);
