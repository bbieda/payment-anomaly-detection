"""Autoenkoder do detekcji anomalii: uczy się odtwarzać transakcje normalne, a wynikiem jest błąd rekonstrukcji.

Architektura wg [2]: koder 16-32, dekoder 32-16 (jednostki ReLU), warstwa wyjściowa liniowa.
Zaimplementowany na sklearn.neural_network.MLPRegressor (cel = wejście), więc działa w Pipeline
i serializuje się joblib-em tak samo jak pozostałe modele.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor


class AutoencoderDetector(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(16, 32, 32, 16), activation="relu", alpha=1e-4, max_iter=100,
                 batch_size=256, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                 random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.net_ = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes), activation=self.activation, alpha=self.alpha,
            max_iter=self.max_iter, batch_size=self.batch_size, early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state)
        self.net_.fit(X, X)
        return self

    def reconstruction_error(self, X):
        """Średni błąd kwadratowy rekonstrukcji dla każdej transakcji (wyższy = bardziej podejrzana)."""
        X = np.asarray(X, dtype=float)
        return np.mean((self.net_.predict(X) - X) ** 2, axis=1)

    def score_samples(self, X):
        # konwencja sklearn (jak IsolationForest/LOF/OneClassSVM): wyższy = bardziej normalna;
        # anomaly_score() w models.py odwraca znak, więc ostateczny wynik = błąd rekonstrukcji
        return -self.reconstruction_error(X)
