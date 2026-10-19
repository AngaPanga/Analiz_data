import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid


def best_threshold(y, probabilities):
    """Считает macro F1 для всех различных порогов без повторного обучения."""
    order = np.argsort(-probabilities, kind="stable")
    probabilities = probabilities[order]
    y = np.asarray(y)[order]
    # При одинаковой вероятности весь блок получает один класс.
    ends = np.r_[np.flatnonzero(np.diff(probabilities)), len(y) - 1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    fn = y.sum() - tp
    tn = len(y) - y.sum() - fp
    scores = tp / (2 * tp + fp + fn) + tn / (2 * tn + fp + fn)
    best = int(np.argmax(scores))
    all_negative = (len(y) - y.sum()) / (2 * len(y) - y.sum())
    if all_negative > scores[best]:
        return float(np.nextafter(probabilities[0], np.inf)), float(all_negative)
    return float(probabilities[ends[best]]), float(scores[best])


def tune_model(model, grid, x_fit, y_fit, x_valid, y_valid):
    """Подбирает параметры и порог исключительно на валидационной выборке."""
    rows = []
    best_score = -1
    best_model = None
    best_cutoff = 0.5
    for params in [{}, *ParameterGrid(grid)]:
        candidate = clone(model).set_params(**params)
        candidate.fit(x_fit, y_fit)
        probabilities = candidate.predict_proba(x_valid)[:, 1]
        cutoff, score = best_threshold(y_valid, probabilities)
        rows.append({
            "parameters": params, "threshold": cutoff, "valid_macro_f1": score,
            "valid_macro_f1_at_0.5": f1_score(
                y_valid, probabilities >= 0.5, average="macro", zero_division=0,
            ),
        })
        if score > best_score:
            best_score, best_model, best_cutoff = score, clone(candidate), cutoff
    return best_model, best_cutoff, pd.DataFrame(rows)
