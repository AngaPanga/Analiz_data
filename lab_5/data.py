from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
SEED = 42


def load_data(path=ROOT / "creditcard.csv"):
    """Проверяет структуру данных и выделяет 30% для итогового теста."""
    df = pd.read_csv(path)
    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    if set(df.columns) != set(columns):
        raise ValueError("Ожидаются столбцы Time, V1–V28, Amount, Class.")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("В данных есть пропуски или бесконечные значения.")
    if set(df.Class.unique()) != {0, 1}:
        raise ValueError("Class должен содержать классы 0 и 1.")
    return train_test_split(
        df.drop(columns="Class"), df.Class,
        test_size=0.3, stratify=df.Class, random_state=SEED,
    )
