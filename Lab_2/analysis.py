# -*- coding: utf-8 -*-
"""Лабораторна робота №2: Парний лінійний та кореляційний аналіз даних.

Студент: Багрій-Белз Андрій
Група: КН-2327б
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Стилізація графіків
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "figure.autolayout": True
})


def analyze_anaconda():
    """Частина 1: Дослідження морфометрії анаконд (довжина X та вага Y)."""
    data_path = os.path.join(BASE_DIR, "anaconda.dat")
    # Зчитування файлу: роздільник - пробільні символи (\s+ або \t)
    df = pd.read_csv(data_path, sep=r"\s+", names=["X", "Y", "Gender"], header=None)

    # Кодування статі: M -> 0 (Male), F -> 1 (Female)
    df["Gender_code"] = df["Gender"].map({"M": 0, "F": 1})

    # Кореляційна матриця Пірсона
    corr_matrix = df[["X", "Y", "Gender_code"]].corr(method="pearson")
    r_xy, p_val_xy = stats.pearsonr(df["X"], df["Y"])

    # Парна лінійна регресія: X (довжина) -> Y (вага)
    X = df[["X"]].values
    Y = df["Y"].values

    model = LinearRegression()
    model.fit(X, Y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    Y_pred = model.predict(X)

    # Метрики якості
    r2 = float(r2_score(Y, Y_pred))
    mse = float(mean_squared_error(Y, Y_pred))
    rmse = float(np.sqrt(mse))

    # Обчислення та діагностика залишків
    residuals = Y - Y_pred
    res_mean = float(np.mean(residuals))
    shapiro_w, shapiro_p = stats.shapiro(residuals)

    # 1. Графік регресії: reg_plot.png
    plt.figure(figsize=(7.5, 5), dpi=300)
    plt.scatter(df["X"], df["Y"], c=df["Gender_code"], cmap="coolwarm", s=50, alpha=0.85, edgecolors="k", label="Анаконди (точкові дані)")
    plt.plot(df["X"], Y_pred, color="crimson", linewidth=2.2, label=f"Регресія: $\\hat{{Y}} = {slope:.4f}X + ({intercept:.4f})$")
    plt.title(f"Лінійна регресія: довжина vs вага ($r = {r_xy:.4f}$, $R^2 = {r2:.4f}$)")
    plt.xlabel("Довжина тіла Snout-vent length (X, см)")
    plt.ylabel("Вага тіла (Y, кг)")
    plt.legend(frameon=True)
    reg_plot_path = os.path.join(FIG_DIR, "reg_plot.png")
    plt.savefig(reg_plot_path, dpi=300)
    plt.close()

    # 2. Графік залишків: residuals_plot.png
    plt.figure(figsize=(7.5, 5), dpi=300)
    plt.scatter(df["X"], residuals, color="teal", s=50, alpha=0.85, edgecolors="k")
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1.8, label="Нульова лінія ($e_i = 0$)")
    plt.title("Діагностичний графік залишків (Residuals vs Predictor X)")
    plt.xlabel("Довжина тіла Snout-vent length (X, см)")
    plt.ylabel("Залишки ($e_i = Y_i - \\hat{Y}_i$, кг)")
    plt.legend(frameon=True)
    res_plot_path = os.path.join(FIG_DIR, "residuals_plot.png")
    plt.savefig(res_plot_path, dpi=300)
    plt.close()

    # 3. Гістограма залишків: residuals_hist.png
    plt.figure(figsize=(7.5, 5), dpi=300)
    sns.histplot(residuals, kde=True, color="darkslateblue", bins=12, stat="density", edgecolor="black")
    # Накладемо теоретичний нормальний розподіл
    norm_x = np.linspace(np.min(residuals), np.max(residuals), 100)
    norm_y = stats.norm.pdf(norm_x, loc=np.mean(residuals), scale=np.std(residuals, ddof=1))
    plt.plot(norm_x, norm_y, "r--", linewidth=2, label="Теоретичний нормальний розподіл")
    plt.title(f"Розподіл залишків регресії (Shapiro-Wilk $W = {shapiro_w:.4f}$, $p = {shapiro_p:.4e}$)")
    plt.xlabel("Величина залишку ($e_i$, кг)")
    plt.ylabel("Щільність ймовірності")
    plt.legend(frameon=True)
    res_hist_path = os.path.join(FIG_DIR, "residuals_hist.png")
    plt.savefig(res_hist_path, dpi=300)
    plt.close()

    return {
        "dataset": "anaconda.dat",
        "sample_size": int(len(df)),
        "gender_counts": {"M": int((df["Gender"] == "M").sum()), "F": int((df["Gender"] == "F").sum())},
        "gender_encoding": "M -> 0, F -> 1",
        "correlation_matrix": {
            "X": corr_matrix["X"].to_dict(),
            "Y": corr_matrix["Y"].to_dict(),
            "Gender_code": corr_matrix["Gender_code"].to_dict()
        },
        "correlation_r": round(float(r_xy), 4),
        "correlation_p_value": float(p_val_xy),
        "regression": {
            "slope_a": round(slope, 4),
            "intercept_b": round(intercept, 4),
            "equation": f"Y = {slope:.4f} * X + ({intercept:.4f})",
            "r2": round(r2, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4)
        },
        "residuals": {
            "mean": float(res_mean),
            "shapiro_w": round(float(shapiro_w), 4),
            "shapiro_p": float(shapiro_p),
            "is_normal_005": bool(shapiro_p > 0.05)
        }
    }


def analyze_custom():
    """Частина 2: Дослідження залежності між площею квартири та її вартістю (flats.csv)."""
    # Шлях до flats.csv (з репозиторію Lab_1 або локально)
    candidates = [
        os.path.join(BASE_DIR, "flats.csv"),
        os.path.join(BASE_DIR, "..", "Lab_1", "flats.csv")
    ]
    flats_path = None
    for p in candidates:
        if os.path.exists(p):
            flats_path = p
            break

    if not flats_path:
        raise FileNotFoundError("Файл flats.csv не знайдено.")

    df_flats = pd.read_csv(flats_path)
    area = pd.to_numeric(df_flats["Загальна_площа"], errors="coerce")
    price = pd.to_numeric(df_flats["Ціна"], errors="coerce")

    # Фільтрація коректних числових значень та усунення явних аномалій
    mask = area.notna() & price.notna()
    df_clean = pd.DataFrame({"area": area[mask], "price": price[mask]})
    df_clean = df_clean[(df_clean["area"] >= 15) & (df_clean["area"] <= 250)]
    df_clean = df_clean[(df_clean["price"] >= 100000) & (df_clean["price"] <= 15000000)]

    X_cust = df_clean[["area"]].values
    Y_cust = df_clean["price"].values

    r_cust, p_cust = stats.pearsonr(df_clean["area"], df_clean["price"])

    model_cust = LinearRegression()
    model_cust.fit(X_cust, Y_cust)

    slope_c = float(model_cust.coef_[0])
    intercept_c = float(model_cust.intercept_)
    Y_pred_c = model_cust.predict(X_cust)

    r2_c = float(r2_score(Y_cust, Y_pred_c))
    mse_c = float(mean_squared_error(Y_cust, Y_pred_c))
    rmse_c = float(np.sqrt(mse_c))

    # 4. Графік регресії власного датасету: custom_regression.png
    plt.figure(figsize=(7.5, 5), dpi=300)
    plt.scatter(df_clean["area"], df_clean["price"] / 1000, color="royalblue", alpha=0.5, s=35, edgecolors="none", label="Квартири (вибірка)")
    sorted_idx = np.argsort(df_clean["area"].values)
    sorted_x = df_clean["area"].values[sorted_idx]
    sorted_pred = (Y_pred_c / 1000)[sorted_idx]
    plt.plot(sorted_x, sorted_pred, color="darkorange", linewidth=2.5,
             label=f"Регресія: $\\hat{{Y}} = {slope_c/1000:.2f}X + ({intercept_c/1000:.2f})$ тис. грн")
    plt.title(f"Залежність ціни від загальної площі квартири ($r = {r_cust:.4f}$, $R^2 = {r2_c:.4f}$)")
    plt.xlabel("Загальна площа ($м^2$)")
    plt.ylabel("Ціна квартири (тис. грн)")
    plt.legend(frameon=True)
    custom_plot_path = os.path.join(FIG_DIR, "custom_regression.png")
    plt.savefig(custom_plot_path, dpi=300)
    plt.close()

    return {
        "dataset": "flats.csv (нерухомість України)",
        "sample_size": int(len(df_clean)),
        "feature_x": "Загальна площа (м²)",
        "target_y": "Ціна квартири (грн)",
        "correlation_r": round(float(r_cust), 4),
        "correlation_p_value": float(p_cust),
        "regression": {
            "slope_a": round(slope_c, 2),
            "intercept_b": round(intercept_c, 2),
            "equation": f"Ціна = {slope_c:.2f} * Площа + ({intercept_c:.2f})",
            "r2": round(r2_c, 4),
            "mse": round(mse_c, 2),
            "rmse": round(rmse_c, 2)
        }
    }


def main():
    print("=== Лабораторна робота №2: Запуск розрахунків ===")
    res_anaconda = analyze_anaconda()
    print("[1] Анаконда датасет:")
    print(f"  r = {res_anaconda['correlation_r']}, p = {res_anaconda['correlation_p_value']:.4e}")
    print(f"  Модель: {res_anaconda['regression']['equation']}")
    print(f"  R^2 = {res_anaconda['regression']['r2']}, MSE = {res_anaconda['regression']['mse']}, RMSE = {res_anaconda['regression']['rmse']}")
    print(f"  Залишки: середнє = {res_anaconda['residuals']['mean']:.2e}, Shapiro p-value = {res_anaconda['residuals']['shapiro_p']:.4e}")

    res_custom = analyze_custom()
    print("\n[2] Власний датасет (flats):")
    print(f"  r = {res_custom['correlation_r']}, p = {res_custom['correlation_p_value']:.4e}")
    print(f"  Модель: {res_custom['regression']['equation']}")
    print(f"  R^2 = {res_custom['regression']['r2']}, RMSE = {res_custom['regression']['rmse']}")

    report_data = {
        "meta": {
            "title": "Регресійний аналіз даних. Кореляція",
            "discipline": "Інтелектуальний аналіз даних",
            "code": "ІАД",
            "lab_no": 2,
            "student": "Багрій-Белз Андрій",
            "group": "КН-2327б"
        },
        "part_1_anaconda": res_anaconda,
        "part_2_custom": res_custom
    }

    report_path = os.path.join(BASE_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Результати збережено в {report_path}")
    print(f"[OK] Усі 4 графіки згенеровано у {FIG_DIR}")


if __name__ == "__main__":
    main()
