# -*- coding: utf-8 -*-
"""Генератор та виконавець Jupyter-ноутбука для Лабораторної роботи №4 (ІАД)."""

import base64
import contextlib
import io
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebook.ipynb")

cells_def = [
    {
        "type": "markdown",
        "source": [
            "# Лабораторна робота №4\n",
            "## Тема: Методи класифікації. Дерева рішень у Python\n",
            "**Дисципліна:** Інтелектуальний аналіз даних (ІАД)  \n",
            "**Студент:** Багрій-Белз Андрій, група КН-2327Б  \n",
            "\n",
            "### Мета роботи\n",
            "Дослідити алгоритми побудови класифікаційних дерев рішень (`DecisionTreeClassifier`) за допомогою бібліотеки `scikit-learn`, вивчити критерій ентропії, оцінити схильність глибоких дерев до перенавчання та реалізувати підбір оптимального гіперпараметра регуляризації (`min_samples_split`) через 5-кратну крос-валідацію.",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 1. Імпорт бібліотек та налаштування середовища\n",
            "Підключаємо модулі для аналізу даних (`pandas`, `numpy`), побудови моделей машинного навчання (`sklearn`) та візуалізації (`matplotlib`, `seaborn`).",
        ],
    },
    {
        "type": "code",
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from sklearn.model_selection import train_test_split, cross_val_score\n",
            "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score,\n",
            "    precision_score,\n",
            "    recall_score,\n",
            "    f1_score,\n",
            "    confusion_matrix,\n",
            "    classification_report,\n",
            ")\n",
            "\n",
            "# Налаштування стилю візуалізації\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "plt.rcParams.update({\n",
            "    'font.size': 11,\n",
            "    'axes.titlesize': 12,\n",
            "    'axes.labelsize': 11,\n",
            "})\n",
            "print('Усі необхідні бібліотеки успішно імпортовано.')",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 2. Завантаження та первинний аналіз медичних даних\n",
            "Для аналізу використовується датасет `heart.xlsx`, що містить медичні показники діагностики серцевих захворювань. Забезпечено підтримку завантаження як із локального файлу, так і завантаження через Google Drive для середовища Google Colab.",
        ],
    },
    {
        "type": "code",
        "source": [
            "data_file = 'heart.xlsx'\n",
            "if not os.path.exists(data_file):\n",
            "    try:\n",
            "        import gdown\n",
            "        google_drive_url = 'https://drive.google.com/file/d/1E04gJ4aeY5B6FslNpVpM_jYBvCyzayMD'\n",
            "        file_id = google_drive_url.split('/')[-1]\n",
            "        gdown.download(f'https://drive.google.com/uc?id={file_id}', data_file, quiet=False)\n",
            "    except Exception as e:\n",
            "        print('Помилка завантаження через gdown:', e)\n",
            "\n",
            "df = pd.read_excel(data_file)\n",
            "print(f'Розмірність датасету: {df.shape}')\n",
            "print('\\nПеревірка пропущених значень:')\n",
            "print(df.isnull().sum())\n",
            "print('\\nРозподіл цільового класу (0 = здоровий, 1 = хворий):')\n",
            "print(df['target'].value_counts())\n",
            "df.head()",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 3. Розділення вибірки на навчальну та тестову\n",
            "Відокремлюємо предиктори від цільової змінної `target`. Розділяємо дані у співвідношенні 70/30 (`test_size=0.3`) із фіксованим зерном генератора випадкових чисел `random_state=42` та стратифікацією за класами для збереження часток категорій у вибірках.",
        ],
    },
    {
        "type": "code",
        "source": [
            "feature_cols = [c for c in df.columns if c != 'target']\n",
            "X = df[feature_cols]\n",
            "y = df['target']\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.3, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "print(f'Навчальна вибірка: {X_train.shape[0]} зразків')\n",
            "print(f'Тестова вибірка:   {X_test.shape[0]} зразків')",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 4. Навчання та візуалізація базового незрізаного дерева рішень\n",
            "Будуємо класифікатор на основі критерію ентропії (`criterion='entropy'`). За замовчуванням глибина дерева не обмежується, що дозволяє детально проаналізувати його структуру та схильність до перенавчання.",
        ],
    },
    {
        "type": "code",
        "source": [
            "clf_full = DecisionTreeClassifier(criterion='entropy', random_state=42)\n",
            "clf_full.fit(X_train, y_train)\n",
            "\n",
            "y_pred_full = clf_full.predict(X_test)\n",
            "acc_full = accuracy_score(y_test, y_pred_full)\n",
            "prec_full = precision_score(y_test, y_pred_full)\n",
            "rec_full = recall_score(y_test, y_pred_full)\n",
            "f1_full = f1_score(y_test, y_pred_full)\n",
            "\n",
            "print(f'Базове дерево: глибина = {clf_full.get_depth()}, кількість листків = {clf_full.get_n_leaves()}')\n",
            "print(f'Метрики на тестовій вибірці:')\n",
            "print(f'  Accuracy:  {acc_full:.4f}')\n",
            "print(f'  Precision: {prec_full:.4f}')\n",
            "print(f'  Recall:    {rec_full:.4f}')\n",
            "print(f'  F1-Score:  {f1_full:.4f}')\n",
            "print('\\nДетальний звіт класифікації:')\n",
            "print(classification_report(y_test, y_pred_full, target_names=['Здоровий (0)', 'Хворий (1)']))",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "Візуалізуємо повну структуру базового дерева рішень.",
        ],
    },
    {
        "type": "code",
        "source": [
            "fig, ax = plt.subplots(figsize=(18, 10), dpi=150)\n",
            "plot_tree(\n",
            "    clf_full,\n",
            "    feature_names=feature_cols,\n",
            "    class_names=['Здоровий (0)', 'Хворий (1)'],\n",
            "    filled=True,\n",
            "    rounded=True,\n",
            "    fontsize=8,\n",
            "    ax=ax,\n",
            ")\n",
            "ax.set_title('Повне базове дерево рішень (DecisionTreeClassifier, criterion=\"entropy\")', fontsize=14, pad=12, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 5. Підбір гіперпараметра регуляризації через 5-кратну крос-валідацію\n",
            "Для уникнення перенавчання проводимо підбір параметра `min_samples_split` (мінімальна кількість зразків, необхідна для розбиття внутрішнього вузла) у діапазоні від 2 до 50 за допомогою 5-кратної крос-валідації (`cv=5`).",
        ],
    },
    {
        "type": "code",
        "source": [
            "splits_range = np.arange(2, 51)\n",
            "cv_errors = []\n",
            "cv_accuracies = []\n",
            "\n",
            "for s in splits_range:\n",
            "    clf_cv = DecisionTreeClassifier(criterion='entropy', min_samples_split=s, random_state=42)\n",
            "    scores = cross_val_score(clf_cv, X_train, y_train, cv=5, scoring='accuracy')\n",
            "    cv_errors.append(1.0 - np.mean(scores))\n",
            "    cv_accuracies.append(np.mean(scores))\n",
            "\n",
            "best_idx = int(np.argmin(cv_errors))\n",
            "optimal_min_samples_split = int(splits_range[best_idx])\n",
            "min_cv_error = float(cv_errors[best_idx])\n",
            "best_cv_accuracy = float(cv_accuracies[best_idx])\n",
            "\n",
            "print(f'Оптимальне значення min_samples_split: {optimal_min_samples_split}')\n",
            "print(f'Мінімальна помилка крос-валідації (CV Error): {min_cv_error:.4f} (Точність: {best_cv_accuracy:.4f})')",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "Будуємо графік залежності середньої помилки крос-валідації від значення параметра `min_samples_split` із позначенням точки оптимуму.",
        ],
    },
    {
        "type": "code",
        "source": [
            "plt.figure(figsize=(9, 5), dpi=120)\n",
            "plt.plot(splits_range, cv_errors, marker='o', markersize=4, linestyle='-', color='#1976d2', label='CV Error (1 - Accuracy)')\n",
            "plt.scatter(optimal_min_samples_split, min_cv_error, color='#d32f2f', s=100, zorder=5, label=f'Оптимум: min_samples_split={optimal_min_samples_split} (Err={min_cv_error:.4f})')\n",
            "plt.axvline(optimal_min_samples_split, color='#d32f2f', linestyle='--', linewidth=1.2, alpha=0.7)\n",
            "plt.title('Залежність помилки крос-валідації від параметра min_samples_split', fontsize=12, pad=10, fontweight='bold')\n",
            "plt.xlabel('Мінімальна кількість зразків для розщеплення (min_samples_split)', fontsize=11, labelpad=8)\n",
            "plt.ylabel('Помилка крос-валідації (CV Error)', fontsize=11, labelpad=8)\n",
            "plt.legend(frameon=True)\n",
            "plt.grid(True, linestyle=':', alpha=0.7)\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 6. Навчання та оцінка обрізаного дерева (Pruned Tree)\n",
            "Навчаємо фінальну оптимізовану модель дерева з отриманим значенням `min_samples_split=37` та оцінюємо її метрики якості на відкладеній тестовій вибірці.",
        ],
    },
    {
        "type": "code",
        "source": [
            "clf_pruned = DecisionTreeClassifier(\n",
            "    criterion='entropy',\n",
            "    min_samples_split=optimal_min_samples_split,\n",
            "    random_state=42,\n",
            ")\n",
            "clf_pruned.fit(X_train, y_train)\n",
            "\n",
            "y_pred_pruned = clf_pruned.predict(X_test)\n",
            "acc_pruned = accuracy_score(y_test, y_pred_pruned)\n",
            "prec_pruned = precision_score(y_test, y_pred_pruned)\n",
            "rec_pruned = recall_score(y_test, y_pred_pruned)\n",
            "f1_pruned = f1_score(y_test, y_pred_pruned)\n",
            "\n",
            "print(f'Обрізане дерево: глибина = {clf_pruned.get_depth()}, кількість листків = {clf_pruned.get_n_leaves()}')\n",
            "print(f'Метрики на тестовій вибірці:')\n",
            "print(f'  Accuracy:  {acc_pruned:.4f}')\n",
            "print(f'  Precision: {prec_pruned:.4f}')\n",
            "print(f'  Recall:    {rec_pruned:.4f}')\n",
            "print(f'  F1-Score:  {f1_pruned:.4f}')\n",
            "print('\\nДетальний звіт класифікації (Обрізане дерево):')\n",
            "print(classification_report(y_test, y_pred_pruned, target_names=['Здоровий (0)', 'Хворий (1)']))",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "Візуалізуємо редуковане дерево рішень.",
        ],
    },
    {
        "type": "code",
        "source": [
            "fig, ax = plt.subplots(figsize=(14, 7), dpi=150)\n",
            "plot_tree(\n",
            "    clf_pruned,\n",
            "    feature_names=feature_cols,\n",
            "    class_names=['Здоровий (0)', 'Хворий (1)'],\n",
            "    filled=True,\n",
            "    rounded=True,\n",
            "    fontsize=9,\n",
            "    ax=ax,\n",
            ")\n",
            "ax.set_title(f'Оптимізоване дерево рішень (min_samples_split={optimal_min_samples_split}, листків={clf_pruned.get_n_leaves()})', fontsize=13, pad=10, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 7. Порівняння моделей: Матриці невідповідностей\n",
            "Порівнюємо структури помилок базового та оптимізованого дерева за допомогою теплових карт матриць невідповідностей (`confusion_matrix`).",
        ],
    },
    {
        "type": "code",
        "source": [
            "cm_full = confusion_matrix(y_test, y_pred_full)\n",
            "cm_pruned = confusion_matrix(y_test, y_pred_pruned)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=120)\n",
            "\n",
            "sns.heatmap(\n",
            "    cm_full,\n",
            "    annot=True,\n",
            "    fmt='d',\n",
            "    cmap='Blues',\n",
            "    cbar=False,\n",
            "    ax=axes[0],\n",
            "    xticklabels=['Здоровий (0)', 'Хворий (1)'],\n",
            "    yticklabels=['Здоровий (0)', 'Хворий (1)'],\n",
            "    annot_kws={'size': 13, 'weight': 'bold'},\n",
            ")\n",
            "axes[0].set_title(f'Базове дерево (Листків: {clf_full.get_n_leaves()})\\nAcc: {acc_full:.3f}, F1: {f1_full:.3f}', fontsize=11, pad=8)\n",
            "axes[0].set_xlabel('Прогнозований клас', fontsize=10, labelpad=6)\n",
            "axes[0].set_ylabel('Справжній клас', fontsize=10, labelpad=6)\n",
            "\n",
            "sns.heatmap(\n",
            "    cm_pruned,\n",
            "    annot=True,\n",
            "    fmt='d',\n",
            "    cmap='Greens',\n",
            "    cbar=False,\n",
            "    ax=axes[1],\n",
            "    xticklabels=['Здоровий (0)', 'Хворий (1)'],\n",
            "    yticklabels=['Здоровий (0)', 'Хворий (1)'],\n",
            "    annot_kws={'size': 13, 'weight': 'bold'},\n",
            ")\n",
            "axes[1].set_title(f'Обрізане дерево (Листків: {clf_pruned.get_n_leaves()})\\nAcc: {acc_pruned:.3f}, F1: {f1_pruned:.3f}', fontsize=11, pad=8)\n",
            "axes[1].set_xlabel('Прогнозований клас', fontsize=10, labelpad=6)\n",
            "axes[1].set_ylabel('Справжній клас', fontsize=10, labelpad=6)\n",
            "\n",
            "plt.suptitle('Матриці невідповідностей: Базове vs Обрізане дерево', fontsize=12, y=1.02, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### 8. Підсумкова порівняльна таблиця метрик\n",
            "Зводимо показники обох моделей у табличний вигляд для наочного порівняння складності та точності.",
        ],
    },
    {
        "type": "code",
        "source": [
            "comparison_df = pd.DataFrame({\n",
            "    'Модель': ['Базове дерево (Full)', 'Обрізане дерево (Pruned)'],\n",
            "    'min_samples_split': [2, optimal_min_samples_split],\n",
            "    'Глибина дерева': [clf_full.get_depth(), clf_pruned.get_depth()],\n",
            "    'Кількість листків': [clf_full.get_n_leaves(), clf_pruned.get_n_leaves()],\n",
            "    'Accuracy': [acc_full, acc_pruned],\n",
            "    'Precision': [prec_full, prec_pruned],\n",
            "    'Recall': [rec_full, rec_pruned],\n",
            "    'F1-Score': [f1_full, f1_pruned],\n",
            "})\n",
            "comparison_df",
        ],
    },
    {
        "type": "markdown",
        "source": [
            "### Висновки\n",
            "1. Базова модель незрізаного дерева рішень формує надмірно складну структуру (22 листки), налаштовуючись на специфічні флуктуації навчальної вибірки.\n",
            "2. Застосування 5-кратної крос-валідації дозволило встановити оптимальний поріг розщеплення `min_samples_split = 37`, що зменшило середню помилку класифікації до 0.2166 (точність 78.34%).\n",
            "3. Обрізана модель скоротила кількість кінцевих листків у 2 рази (з 22 до 11) без погіршення загальної точності (`Accuracy = 0.7033`), при цьому повнота виявлення захворювання (`Recall`) зросла з 0.78 до 0.82, а значення гармонійного середнього `F1-Score` підвищилося до 0.7523.",
        ],
    },
]

# Виконання та збереження notebook
execution_env = {}
exec_count = 1
notebook_cells = []

real_show = plt.show
captured_figs = []


def custom_show(*args, **kwargs):
    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        captured_figs.append(img_b64)
    plt.close("all")


plt.show = custom_show

current_cwd = os.getcwd()
os.chdir(BASE_DIR)

try:
    for c in cells_def:
        if c["type"] == "markdown":
            notebook_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": c["source"],
            })
        elif c["type"] == "code":
            code_text = "".join(c["source"])
            stdout_io = io.StringIO()
            captured_figs = []

            with contextlib.redirect_stdout(stdout_io):
                exec_result = None
                lines = code_text.strip().split("\n")
                if len(lines) > 0 and not lines[-1].startswith(("import ", "from ", "def ", "class ", "for ", "if ", "while ", "try:", "except", "#")) and "=" not in lines[-1] and not lines[-1].startswith("plt.") and not lines[-1].startswith("fig"):
                    code_to_exec = "\n".join(lines[:-1])
                    last_expr = lines[-1]
                    if code_to_exec:
                        exec(code_to_exec, execution_env)
                    exec_result = eval(last_expr, execution_env)
                else:
                    exec(code_text, execution_env)

            outputs = []
            stdout_val = stdout_io.getvalue()
            if stdout_val:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_val.splitlines(keepends=True),
                })

            for img_b64 in captured_figs:
                outputs.append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                })

            if exec_result is not None:
                if hasattr(exec_result, "_repr_html_"):
                    outputs.append({
                        "data": {
                            "text/html": [exec_result._repr_html_()],
                            "text/plain": [str(exec_result)],
                        },
                        "metadata": {},
                        "output_type": "execute_result",
                        "execution_count": exec_count,
                    })
                else:
                    outputs.append({
                        "data": {
                            "text/plain": [repr(exec_result)],
                        },
                        "metadata": {},
                        "output_type": "execute_result",
                        "execution_count": exec_count,
                    })

            notebook_cells.append({
                "cell_type": "code",
                "execution_count": exec_count,
                "metadata": {},
                "outputs": outputs,
                "source": c["source"],
            })
            exec_count += 1
finally:
    plt.show = real_show
    os.chdir(current_cwd)

notebook_json = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1, ensure_ascii=False)

print(f"Успішно створено та виконано {NOTEBOOK_PATH} ({len(notebook_cells)} клітинок).")
