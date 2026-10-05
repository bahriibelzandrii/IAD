# -*- coding: utf-8 -*-
"""
Аналітичний модуль для Лабораторної роботи №4 (ІАД)
Тема: Методи класифікації. Дерева рішень у Python.
Виконавець: Багрій-Белз Андрій, група КН-2327Б.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------------------------------------------------------------------
# Константи та шляхи
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
REPORT_JSON_PATH = os.path.join(BASE_DIR, "report.json")
DATA_PATH = os.path.join(BASE_DIR, "heart.xlsx")

os.makedirs(FIG_DIR, exist_ok=True)

# Стиль оформлення графіків (висока якість для звіту)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 13,
})


def run_analysis():
    print("=== Лабораторна робота №4: Методи класифікації (Дерева рішень) ===")

    # 1. Завантаження даних
    print(f"Зчитування даних із {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    print(f"Розмір датасету: {df.shape[0]} рядків, {df.shape[1]} стовпців.")
    print(f"Пропущені значення:\n{df.isnull().sum()}")
    print("Розподіл цільової мітки (target):")
    print(df["target"].value_counts())

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    # 2. Розділення на train та test (test_size=0.3, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nРозмір навчальної вибірки: {X_train.shape[0]}, тестової: {X_test.shape[0]}")

    # 3. Базове дерево рішень (необмежене)
    print("\n--- 1. Навчання базового дерева рішень (Full Tree) ---")
    clf_full = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf_full.fit(X_train, y_train)

    y_pred_full = clf_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    prec_full = precision_score(y_test, y_pred_full)
    rec_full = recall_score(y_test, y_pred_full)
    f1_full = f1_score(y_test, y_pred_full)
    depth_full = clf_full.get_depth()
    leaves_full = clf_full.get_n_leaves()

    print(f"Базове дерево: Глибина = {depth_full}, Кількість листків = {leaves_full}")
    print(f"Метрики на тесті: Accuracy = {acc_full:.4f}, Precision = {prec_full:.4f}, Recall = {rec_full:.4f}, F1 = {f1_full:.4f}")
    print("Classification Report (Базове дерево):")
    print(classification_report(y_test, y_pred_full, target_names=["Здоровий (0)", "Хворий (1)"]))

    # Збереження графіку повного дерева
    fig_full_path = os.path.join(FIG_DIR, "tree_full.png")
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    plot_tree(
        clf_full,
        feature_names=feature_cols,
        class_names=["Здоровий (0)", "Хворий (1)"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Повне незрізане дерево рішень (DecisionTreeClassifier, criterion='entropy')", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_full_path, dpi=300)
    plt.close()
    print(f"Збережено: {fig_full_path}")

    # 4. Крос-валідація для підбору min_samples_split (2..50)
    print("\n--- 2. 5-кратна крос-валідація (підбір min_samples_split) ---")
    splits_range = np.arange(2, 51)
    cv_errors = []
    cv_accuracies = []

    for s in splits_range:
        clf_cv = DecisionTreeClassifier(criterion="entropy", min_samples_split=s, random_state=42)
        scores = cross_val_score(clf_cv, X_train, y_train, cv=5, scoring="accuracy")
        cv_errors.append(1.0 - np.mean(scores))
        cv_accuracies.append(np.mean(scores))

    best_idx = int(np.argmin(cv_errors))
    optimal_min_samples_split = int(splits_range[best_idx])
    min_cv_error = float(cv_errors[best_idx])
    best_cv_accuracy = float(cv_accuracies[best_idx])

    print(f"Оптимальний параметр min_samples_split: {optimal_min_samples_split}")
    print(f"Мінімальна помилка крос-валідації (CV Error): {min_cv_error:.4f} (CV Accuracy = {best_cv_accuracy:.4f})")

    # Збереження графіка помилки крос-валідації
    fig_cv_path = os.path.join(FIG_DIR, "cv_error_plot.png")
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax.plot(splits_range, cv_errors, marker="o", markersize=4, linestyle="-", color="#1976d2", label="Помилка крос-валідації (1 - Accuracy)")
    ax.scatter(optimal_min_samples_split, min_cv_error, color="#d32f2f", s=90, zorder=5, label=f"Оптимум: min_samples_split={optimal_min_samples_split} (Err={min_cv_error:.4f})")
    ax.axvline(optimal_min_samples_split, color="#d32f2f", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title("Залежність помилки крос-валідації від параметра min_samples_split", fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel("Мінімальна кількість зразків для розщеплення (min_samples_split)", fontsize=11, labelpad=8)
    ax.set_ylabel("Помилка крос-валідації (CV Error)", fontsize=11, labelpad=8)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_cv_path, dpi=300)
    plt.close()
    print(f"Збережено: {fig_cv_path}")

    # 5. Обрізане дерево (Pruned Tree) з оптимальним параметром
    print("\n--- 3. Навчання та оцінка обрізаного дерева (Pruned Tree) ---")
    clf_pruned = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_split=optimal_min_samples_split,
        random_state=42,
    )
    clf_pruned.fit(X_train, y_train)

    y_pred_pruned = clf_pruned.predict(X_test)
    acc_pruned = accuracy_score(y_test, y_pred_pruned)
    prec_pruned = precision_score(y_test, y_pred_pruned)
    rec_pruned = recall_score(y_test, y_pred_pruned)
    f1_pruned = f1_score(y_test, y_pred_pruned)
    depth_pruned = clf_pruned.get_depth()
    leaves_pruned = clf_pruned.get_n_leaves()

    print(f"Обрізане дерево: Глибина = {depth_pruned}, Кількість листків = {leaves_pruned}")
    print(f"Метрики на тесті: Accuracy = {acc_pruned:.4f}, Precision = {prec_pruned:.4f}, Recall = {rec_pruned:.4f}, F1 = {f1_pruned:.4f}")
    print("Classification Report (Обрізане дерево):")
    print(classification_report(y_test, y_pred_pruned, target_names=["Здоровий (0)", "Хворий (1)"]))

    # Збереження візуалізації обрізаного дерева
    fig_pruned_path = os.path.join(FIG_DIR, "tree_pruned.png")
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    plot_tree(
        clf_pruned,
        feature_names=feature_cols,
        class_names=["Здоровий (0)", "Хворий (1)"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(f"Оптимізоване дерево рішень (min_samples_split={optimal_min_samples_split}, листків={leaves_pruned})", fontsize=13, pad=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_pruned_path, dpi=300)
    plt.close()
    print(f"Збережено: {fig_pruned_path}")

    # 6. Матриця невідповідностей (Confusion Matrix)
    cm_full = confusion_matrix(y_test, y_pred_full)
    cm_pruned = confusion_matrix(y_test, y_pred_pruned)

    fig_cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    sns.heatmap(
        cm_full,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Здоровий (0)", "Хворий (1)"],
        yticklabels=["Здоровий (0)", "Хворий (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title(f"Базове дерево (Листків: {leaves_full})\nAcc: {acc_full:.3f}, F1: {f1_full:.3f}", fontsize=11, pad=8)
    axes[0].set_xlabel("Прогнозований клас", fontsize=10, labelpad=6)
    axes[0].set_ylabel("Справжній клас", fontsize=10, labelpad=6)

    sns.heatmap(
        cm_pruned,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        ax=axes[1],
        xticklabels=["Здоровий (0)", "Хворий (1)"],
        yticklabels=["Здоровий (0)", "Хворий (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[1].set_title(f"Обрізане дерево (Листків: {leaves_pruned})\nAcc: {acc_pruned:.3f}, F1: {f1_pruned:.3f}", fontsize=11, pad=8)
    axes[1].set_xlabel("Прогнозований клас", fontsize=10, labelpad=6)
    axes[1].set_ylabel("Справжній клас", fontsize=10, labelpad=6)

    plt.suptitle("Матриці невідповідностей для базової та оптимізованої моделей", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_cm_path, dpi=300)
    plt.close()
    print(f"Збережено: {fig_cm_path}")

    # 7. Збереження метрик у report.json
    report_data = {
        "dataset_info": {
            "total_samples": int(df.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": feature_cols,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "target_distribution": {
                "0": int((y == 0).sum()),
                "1": int((y == 1).sum()),
            },
        },
        "hyperparameter_tuning": {
            "param_name": "min_samples_split",
            "search_range": [int(s) for s in splits_range],
            "optimal_value": optimal_min_samples_split,
            "min_cv_error": round(min_cv_error, 4),
            "best_cv_accuracy": round(best_cv_accuracy, 4),
            "cv_folds": 5,
        },
        "full_tree": {
            "criterion": "entropy",
            "depth": int(depth_full),
            "leaf_count": int(leaves_full),
            "accuracy": round(float(acc_full), 4),
            "precision": round(float(prec_full), 4),
            "recall": round(float(rec_full), 4),
            "f1_score": round(float(f1_full), 4),
            "confusion_matrix": [[int(v) for v in row] for row in cm_full],
        },
        "pruned_tree": {
            "criterion": "entropy",
            "min_samples_split": int(optimal_min_samples_split),
            "depth": int(depth_pruned),
            "leaf_count": int(leaves_pruned),
            "accuracy": round(float(acc_pruned), 4),
            "precision": round(float(prec_pruned), 4),
            "recall": round(float(rec_pruned), 4),
            "f1_score": round(float(f1_pruned), 4),
            "confusion_matrix": [[int(v) for v in row] for row in cm_pruned],
        },
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nЗбережено всі метрики у {REPORT_JSON_PATH}")
    print("Аналіз успішно завершено!")


if __name__ == "__main__":
    run_analysis()
