import numpy as np

from anomaly.data import temporal_split
from anomaly.metrics import best_f1_threshold, evaluate
from anomaly.synthetic import make_synthetic


def test_temporal_split_is_ordered_and_complete():
    df = make_synthetic(1000)
    tr, va, te = temporal_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["Time"].max() <= va["Time"].min() <= va["Time"].max() <= te["Time"].min()


def test_evaluate_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr = best_f1_threshold(y, s)
    m = evaluate(y, s, thr)
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 2}
