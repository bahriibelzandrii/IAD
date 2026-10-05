# -*- coding: utf-8 -*-
"""
Лабораторна робота №3 з курсу "Інтелектуальний аналіз даних"
Тема: Дисперсійний аналіз даних (ANOVA)
Виконав: Багрій-Белз Андрій (КН-2327б)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Налаштування стилів графіків
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Дослідження Voice.txt (Однофакторний ANOVA)
# ---------------------------------------------------------------------------
print("=== 1. Дослідження Voice.txt ===")
voice_path = os.path.join(BASE_DIR, "Voice.txt")
df_voice = pd.read_csv(voice_path, sep=r"\s+")
print(f"Завантажено {len(df_voice)} записів.")
print("Розподіл за фактором голосу:\n", df_voice["Factor"].value_counts())

# Повний аналіз для всіх типів голосу
model_voice_all = ols("Hei ~ Factor", data=df_voice).fit()
anova_voice_all = anova_lm(model_voice_all)
print("\nANOVA (Усі співаки):\n", anova_voice_all)

# Підгрупа жінок (Soprano, Alto)
df_women = df_voice[df_voice["Factor"].isin(["Soprano", "Alto"])].copy()
model_women = ols("Hei ~ Factor", data=df_women).fit()
anova_women = anova_lm(model_women)
print("\nANOVA (Жінки: Soprano vs Alto):\n", anova_women)

# Підгрупа чоловіків (Tenor, Bass)
df_men = df_voice[df_voice["Factor"].isin(["Tenor", "Bass"])].copy()
model_men = ols("Hei ~ Factor", data=df_men).fit()
anova_men = anova_lm(model_men)
print("\nANOVA (Чоловіки: Tenor vs Bass):\n", anova_men)

# Графік 1: Boxplot для Voice
order_voice = ["Soprano", "Alto", "Tenor", "Bass"]
palette_voice = {"Soprano": "#f48fb1", "Alto": "#ec407a", "Tenor": "#64b5f6", "Bass": "#1e88e5"}

plt.figure(figsize=(8, 5.5), dpi=300)
ax1 = sns.boxplot(x="Factor", y="Hei", hue="Factor", data=df_voice, order=order_voice, palette=palette_voice, width=0.5, boxprops=dict(alpha=0.85), legend=False)
sns.stripplot(x="Factor", y="Hei", data=df_voice, order=order_voice, color="black", alpha=0.35, jitter=0.2, size=5)
plt.title("Розподіл зросту співаків за тембром голосу (Voice.txt)", pad=12, fontweight="bold")
plt.xlabel("Тембр голосу (Factor)", labelpad=8)
plt.ylabel("Зріст (дюйми, Hei)", labelpad=8)
plt.axvline(1.5, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
plt.text(0.5, df_voice["Hei"].max() - 1, "Жінки (p = 0.235)", horizontalalignment="center", fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#fce4ec", ec="#f48fb1"))
plt.text(2.5, df_voice["Hei"].max() - 1, "Чоловіки (p = 0.492)", horizontalalignment="center", fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#64b5f6"))
plt.tight_layout()
fig_voice_path = os.path.join(FIG_DIR, "voice_boxplot.png")
plt.savefig(fig_voice_path, dpi=300)
plt.close()
print(f"Збережено: {fig_voice_path}")

# ---------------------------------------------------------------------------
# 2. Дослідження rat.txt (Двофакторний ANOVA)
# ---------------------------------------------------------------------------
print("\n=== 2. Дослідження rat.txt ===")
rat_path = os.path.join(BASE_DIR, "rat.txt")
df_rat = pd.read_csv(rat_path, sep=r"\s+")
print(f"Завантажено {len(df_rat)} записів.")
print(df_rat.head())

# Двофакторна модель із взаємодією Type II ANOVA
model_rat = ols("ERRORS ~ C(ENVIRNMNT) + C(STRAIN) + C(ENVIRNMNT):C(STRAIN)", data=df_rat).fit()
anova_rat = anova_lm(model_rat, typ=2)
print("\nANOVA (rat.txt Type 2):\n", anova_rat)

# Графік 2: Interaction plot
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
interaction_plot(
    x=df_rat["STRAIN"],
    trace=df_rat["ENVIRNMNT"],
    response=df_rat["ERRORS"],
    colors=["#1976d2", "#d32f2f"],
    markers=["o", "s"],
    ms=8,
    linestyles=["-", "--"],
    ax=ax,
    legendtitle="Середовище"
)
ax.set_title("Графік взаємодії факторів середовища та генетичної лінії (Rat)", pad=12, fontweight="bold")
ax.set_xlabel("Генетична лінія (STRAIN)", labelpad=8)
ax.set_ylabel("Середня кількість помилок (ERRORS)", labelpad=8)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig_rat_interaction_path = os.path.join(FIG_DIR, "rat_interaction.png")
plt.savefig(fig_rat_interaction_path, dpi=300)
plt.close()
print(f"Збережено: {fig_rat_interaction_path}")

# Графік 3: Boxplot для rat.txt
plt.figure(figsize=(8, 5.5), dpi=300)
sns.boxplot(
    x="STRAIN",
    y="ERRORS",
    hue="ENVIRNMNT",
    data=df_rat,
    palette={"Free": "#42a5f5", "Restricted": "#ef5350"},
    width=0.55,
    boxprops=dict(alpha=0.85)
)
sns.stripplot(
    x="STRAIN",
    y="ERRORS",
    hue="ENVIRNMNT",
    data=df_rat,
    dodge=True,
    palette={"Free": "#0d47a1", "Restricted": "#b71c1c"},
    alpha=0.6,
    size=6
)
# Прибираємо дублікати в легенді від stripplot
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Середовище")
plt.title("Розподіл помилок щурів за середовищем та генетичною лінією", pad=12, fontweight="bold")
plt.xlabel("Генетична лінія (STRAIN)", labelpad=8)
plt.ylabel("Кількість помилок (ERRORS)", labelpad=8)
plt.tight_layout()
fig_rat_box_path = os.path.join(FIG_DIR, "rat_boxplot.png")
plt.savefig(fig_rat_box_path, dpi=300)
plt.close()
print(f"Збережено: {fig_rat_box_path}")

# ---------------------------------------------------------------------------
# 3. Дослідження власного датасету (tips - загальний чек за днем тижня)
# ---------------------------------------------------------------------------
print("\n=== 3. Дослідження власного датасету (tips) ===")
df_tips = sns.load_dataset("tips")
print(f"Завантажено датасет Tips: {len(df_tips)} рядків.")
print(df_tips.groupby("day", observed=False)["total_bill"].describe())

# Перевірка умов ANOVA
# Тест Левена на однорідність дисперсій
groups_tips = [group["total_bill"].values for _, group in df_tips.groupby("day", observed=False)]
stat_levene, p_levene = stats.levene(*groups_tips)
print(f"Тест Левена: W = {stat_levene:.4f}, p = {p_levene:.4f}")

# Однофакторний ANOVA
model_tips = ols("total_bill ~ C(day)", data=df_tips).fit()
anova_tips = anova_lm(model_tips)
print("\nANOVA (Tips: total_bill ~ day):\n", anova_tips)

# Графік 4: Boxplot + розсіювання для tips
plt.figure(figsize=(8, 5.5), dpi=300)
day_order = ["Thur", "Fri", "Sat", "Sun"]
palette_tips = {"Thur": "#81c784", "Fri": "#4db6ac", "Sat": "#ffb74d", "Sun": "#ff8a65"}
sns.boxplot(
    x="day",
    y="total_bill",
    hue="day",
    data=df_tips,
    order=day_order,
    palette=palette_tips,
    width=0.5,
    boxprops=dict(alpha=0.85),
    legend=False
)
sns.stripplot(
    x="day",
    y="total_bill",
    data=df_tips,
    order=day_order,
    color="#37474f",
    alpha=0.35,
    jitter=0.2,
    size=5
)
# Додаємо середні точки
day_means = df_tips.groupby("day", observed=False)["total_bill"].mean().loc[day_order]
plt.plot(range(4), day_means, color="#b71c1c", marker="D", markersize=6, linestyle="--", linewidth=1.5, label="Середнє значення")
plt.title("Розподіл суми загального рахунку за днями тижня (Tips dataset)", pad=12, fontweight="bold")
plt.xlabel("День тижня (day)", labelpad=8)
plt.ylabel("Сума рахунку ($, total_bill)", labelpad=8)
plt.legend(loc="upper left")
plt.tight_layout()
fig_custom_path = os.path.join(FIG_DIR, "custom_anova.png")
plt.savefig(fig_custom_path, dpi=300)
plt.close()
print(f"Збережено: {fig_custom_path}")

# ---------------------------------------------------------------------------
# 4. Формування та збереження report.json
# ---------------------------------------------------------------------------
def anova_to_dict(df_aov):
    res = {}
    for idx, row in df_aov.iterrows():
        res[str(idx)] = {
            "df": float(row["df"]) if not np.isnan(row["df"]) else None,
            "sum_sq": float(row["sum_sq"]) if not np.isnan(row["sum_sq"]) else None,
            "mean_sq": float(row["mean_sq"]) if "mean_sq" in row and not np.isnan(row["mean_sq"]) else (
                float(row["sum_sq"] / row["df"]) if not np.isnan(row["df"]) and row["df"] > 0 else None
            ),
            "F": float(row["F"]) if not np.isnan(row["F"]) else None,
            "p_value": float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
        }
    return res

report_data = {
    "voice_all": anova_to_dict(anova_voice_all),
    "voice_women": anova_to_dict(anova_women),
    "voice_men": anova_to_dict(anova_men),
    "rat_two_way": anova_to_dict(anova_rat),
    "custom_anova": {
        "levene_test": {
            "statistic": float(stat_levene),
            "p_value": float(p_levene)
        },
        "anova": anova_to_dict(anova_tips)
    }
}

report_json_path = os.path.join(BASE_DIR, "report.json")
with open(report_json_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)
print(f"Збережено результати у {report_json_path}")
