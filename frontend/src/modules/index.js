// Rejestr modułów panelu. Nowa strona = nowy plik w tym katalogu + jedna linia tutaj
// (nawigacja i routing budują się z tej listy).
import Dashboard from './Dashboard.jsx';
import Models from './Models.jsx';
import Metrics from './Metrics.jsx';
import Predict from './Predict.jsx';
import Sandbox from './Sandbox.jsx';
import History from './History.jsx';

export const modules = [
  { id: 'dashboard', label: 'Dashboard', component: Dashboard },
  { id: 'models', label: 'Modele', component: Models },
  { id: 'metrics', label: 'Metryki', component: Metrics },
  { id: 'predict', label: 'Predykcja', component: Predict },
  { id: 'sandbox', label: 'Sandbox', component: Sandbox },
  { id: 'history', label: 'Historia', component: History },
];
