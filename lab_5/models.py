from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from data import SEED


def get_models():
    """Базовые модели и небольшие сетки параметров для локального CPU."""
    return {
        "Bagging": (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=SEED),
                n_estimators=30, max_samples=0.3, n_jobs=4, random_state=SEED,
            ),
            {"estimator__max_depth": [8, 16],
             "estimator__class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=150, n_jobs=4, random_state=SEED),
            {"max_depth": [12, None], "class_weight": [None, "balanced_subsample"]},
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=400, learning_rate=0.05, verbosity=-1,
                n_jobs=4, random_state=SEED, deterministic=True, force_col_wise=True,
            ),
            {"num_leaves": [15, 31], "reg_lambda": [1, 5]},
        ),
        "CatBoost": (
            CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                loss_function="Logloss", verbose=False, allow_writing_files=False,
                thread_count=4, random_seed=SEED,
            ),
            {"depth": [5, 7], "l2_leaf_reg": [3, 10]},
        ),
    }
