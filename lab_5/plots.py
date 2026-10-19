import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


def plot_results(results, y_test, prediction, name, output):
    """Сохраняет сравнение моделей и матрицу ошибок на чёрном фоне."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    positions = np.arange(len(results))
    for shift, column, color, label in [
        (-0.2, "test_macro_f1", "#00e5ff", "Подобранный порог"),
        (0.2, "test_macro_f1_at_0.5", "#ffbf00", "Порог 0.5"),
    ]:
        bars = ax.bar(positions + shift, results[column], width=0.4, color=color, label=label)
        ax.bar_label(bars, fmt="%.4f", padding=5, color="white", fontsize=9)
    ax.set_xticks(positions, results.model)
    ax.legend(loc="lower right")
    ax.set(ylim=(0, 1.05), ylabel="Macro F1", title="Сравнение моделей на тестовой выборке")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "model_comparison.png", dpi=180, facecolor="black")

    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    # Цвет показывает долю внутри истинного класса, подписи — число транзакций.
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("contrast", ["#101010", "#006b83"])
    ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    for (row, col), count in np.ndenumerate(matrix):
        ax.text(col, row, f"{count:,}\n({normalized[row, col]:.3%})",
                ha="center", va="center", color="white", fontsize=17)
    labels = ["Легальная (0)", "Мошенническая (1)"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Предсказанный класс", ylabel="Истинный класс",
           title=f"Матрица ошибок: {name}")
    fig.savefig(output / "confusion_matrix.png", dpi=180, facecolor="black", bbox_inches="tight")
    return matrix
