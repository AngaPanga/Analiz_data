import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from data import ROOT, SEED, load_data
from models import get_models
from plots import plot_results
from training import tune_model


def run_experiment(csv_path, output):
    output.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = load_data(csv_path)
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train, random_state=SEED,
    )
    pd.DataFrame({
        "all": y_train.value_counts() + y_test.value_counts(),
        "train": y_train.value_counts(), "test": y_test.value_counts(),
        "fit": y_fit.value_counts(), "validation": y_valid.value_counts(),
    }).rename_axis("Class").to_csv(output / "split_counts.csv")
    rows, predictions = [], {}
    for name, (model, grid) in get_models().items():
        yield f"Обучение и настройка {name}..."
        tuned, cutoff, search = tune_model(model, grid, x_fit, y_fit, x_valid, y_valid)
        search.to_csv(output / f"search_{name}.csv", index=False)
        tuned.fit(x_train, y_train)
        probabilities = tuned.predict_proba(x_test)[:, 1]
        predictions[name] = (probabilities >= cutoff).astype(int)
        predictions[name + " (0.5)"] = (probabilities >= 0.5).astype(int)
        rows.append({
            "model": name, "threshold": cutoff,
            "valid_macro_f1": search.valid_macro_f1.max(),
            "test_macro_f1": f1_score(y_test, predictions[name], average="macro"),
            "test_macro_f1_at_0.5": f1_score(y_test, probabilities >= 0.5, average="macro"),
        })
        (output / f"parameters_{name}.json").write_text(
            json.dumps(tuned.get_params(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        yield f"{name}: macro F1 = {rows[-1]['test_macro_f1']:.6f}"
    results = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    results.to_csv(output / "metrics.csv", index=False)
    winner = results.iloc[0].model
    standard = results.loc[results["test_macro_f1_at_0.5"].idxmax()]
    if standard["test_macro_f1_at_0.5"] > results.iloc[0].test_macro_f1:
        winner = standard.model + " (0.5)"
    matrix = plot_results(results, y_test, predictions[winner], winner, output)
    pd.DataFrame(matrix, index=[0, 1], columns=[0, 1]).to_csv(output / "confusion_matrix.csv")
    report = classification_report(y_test, predictions[winner], digits=6, zero_division=0)
    (output / "classification_report.txt").write_text(report, encoding="utf-8")
    yield "\n" + results.to_string(index=False)
    yield f"\nЛучшая модель на тесте: {winner}\n{report}"
    yield f"Результаты сохранены в {output}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Лабораторная №5: обнаружение мошенничества")
    parser.add_argument("--data", type=Path, default=ROOT / "creditcard.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--no-show", action="store_true", help="Не открывать окна графиков")
    args = parser.parse_args()
    for message in run_experiment(args.data, args.output):
        print(message, flush=True)
    if not args.no_show:
        plt.show()
