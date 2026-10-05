# -*- coding: utf-8 -*-
"""Аналіз та прогнозування часових рядів (Лабораторна робота №7).

Цей скрипт реалізує повний цикл дослідження:
1. Згладжування ряду kings.txt простим ковзним середнім (SMA, n=9).
2. Адитивна декомпозиція ряду babyboom.txt (period=12).
3. Розподіл ряду Airline Passengers на Train / Test (test_size=36).
4. Прогнозування базовою моделлю NaiveForecaster та моделлю Holt-Winters (ExponentialSmoothing).
5. Розрахунок метрик похибок (MAE, RMSE, MAPE) та збереження у report.json.
6. Збереження візуалізацій у figures/.
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sktime.utils.plotting import plot_series
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# Налаштування стилю графіків
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def task1_smooth_kings():
    """Дослідження та згладжування ряду тривалості життя королів Англії (kings.txt)."""
    kings_path = os.path.join(BASE_DIR, "kings.txt")
    kings = np.loadtxt(kings_path)
    kings_series = pd.Series(kings, index=range(1, len(kings) + 1), name="Age")
    kings_sma = kings_series.rolling(window=9, center=True).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kings_series.index, kings_series.values, marker="o", color="#2b5c8f", label="Фактичний вік (kings.txt)", alpha=0.75, linewidth=1.5)
    ax.plot(kings_sma.index, kings_sma.values, color="#d9534f", linewidth=2.5, label="SMA (n=9, center=True)")
    ax.set_title("Згладжування ряду віку королів Англії простим ковзним середнім (SMA)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Номер короля (хронологічний порядок)", fontsize=11)
    ax.set_ylabel("Вік у момент смерті (років)", fontsize=11)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "kings_sma.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[1/5] Збережено: {out_path}")
    return kings_series, kings_sma


def task2_decompose_babyboom():
    """Адитивна декомпозиція ряду ваги / показників новонароджених (babyboom.txt)."""
    babyboom_path = os.path.join(BASE_DIR, "babyboom.txt")
    babyboom = np.loadtxt(babyboom_path)
    babyboom_series = pd.Series(babyboom, name="Birth_Metric")
    decomp = seasonal_decompose(babyboom_series, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    components = [
        (decomp.observed, "Observed (Спостережуваний ряд)", "#2b5c8f"),
        (decomp.trend, "Trend (Трендова складова)", "#d9534f"),
        (decomp.seasonal, "Seasonal (Сезонна складова, period=12)", "#2ca02c"),
        (decomp.resid, "Residual (Випадкові залишки / Шум)", "#7f7f7f"),
    ]

    for ax, (data, title, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=1.8)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=4)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Номер спостереження", fontsize=11)
    fig.suptitle("Адитивна декомпозиція часового ряду babyboom.txt (period=12)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "babyboom_decomposition.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[2/5] Збережено: {out_path}")
    return decomp


def task3_split_series():
    """Завантаження ряду пасажиропотоку та хронологічний поділ на Train і Test."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(y_train, y_test, labels=["Тренувальна вибірка (Train)", "Тестова вибірка (Test, 36 міс.)"], ax=ax)
    ax.set_title("Хронологічний поділ часового ряду Airline Passengers (Train / Test)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Рік / Місяць", fontsize=11)
    ax.set_ylabel("Кількість пасажирів (тис.)", fontsize=11)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "split_series.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[3/5] Збережено: {out_path}")
    return y, y_train, y_test


def task4_forecast_and_evaluate(y_train, y_test):
    """Навчання NaiveForecaster і ExponentialSmoothing, порівняння та метрики."""
    # 1. Naive Baseline (стратегія 'last')
    forecaster_naive = NaiveForecaster(strategy="last")
    forecaster_naive.fit(y_train)
    fh = np.arange(1, len(y_test) + 1)
    y_pred_naive = forecaster_naive.predict(fh)

    # 2. Holt-Winters Exponential Smoothing (адитивний тренд і сезонність period=12)
    # Перетворюємо у часові мітки для сумісності зі statsmodels
    y_train_ts = y_train.to_timestamp()
    y_test_ts = y_test.to_timestamp()
    model_hw = ExponentialSmoothing(
        y_train_ts,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit()
    y_pred_hw = model_hw.forecast(len(y_test))
    # Перетворимо прогноз Holt-Winters назад у PeriodIndex для сумісності відображення
    y_pred_hw_series = pd.Series(y_pred_hw.values, index=y_test.index, name="Holt-Winters")

    # Візуалізація прогнозу
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_series(
        y_train,
        y_test,
        y_pred_naive,
        y_pred_hw_series,
        labels=[
            "Train (навчальні дані)",
            "Test (фактичні дані)",
            "NaiveForecaster (last)",
            "Holt-Winters (Trend+Seasonality)",
        ],
        colors=["#2b5c8f", "#17becf", "#d9534f", "#2ca02c"],
        ax=ax,
    )
    ax.set_title("Порівняння прогнозів часового ряду Airline Passengers (горизонт fh=36 міс.)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Рік / Місяць", fontsize=11)
    ax.set_ylabel("Кількість пасажирів (тис.)", fontsize=11)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "forecast_comparison.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[4/5] Збережено: {out_path}")

    # Розрахунок метрик якості (MAE, RMSE, MAPE)
    mae_naive = float(mean_absolute_error(y_test, y_pred_naive))
    rmse_naive = float(mean_squared_error(y_test, y_pred_naive, square_root=True))
    mape_naive = float(mean_absolute_percentage_error(y_test, y_pred_naive))

    mae_hw = float(mean_absolute_error(y_test_ts, y_pred_hw))
    rmse_hw = float(mean_squared_error(y_test_ts, y_pred_hw, square_root=True))
    mape_hw = float(mean_absolute_percentage_error(y_test_ts, y_pred_hw))

    metrics = {
        "dataset": "Airline Passengers",
        "horizon_periods": int(len(y_test)),
        "models": {
            "NaiveForecaster_last": {
                "name": "Базова наївна модель (NaiveForecaster 'last')",
                "MAE": round(mae_naive, 3),
                "RMSE": round(rmse_naive, 3),
                "MAPE_percent": round(mape_naive * 100, 2),
            },
            "HoltWinters_ExponentialSmoothing": {
                "name": "Експоненційне згладжування Гольта-Вінтерса (Add Trend + Add Seasonality)",
                "MAE": round(mae_hw, 3),
                "RMSE": round(rmse_hw, 3),
                "MAPE_percent": round(mape_hw * 100, 2),
            },
        },
    }

    report_json_path = os.path.join(BASE_DIR, "report.json")
    with open(report_json_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    print(f"[5/5] Збережено метрики: {report_json_path}")
    print(f"      Naive: MAE={mae_naive:.3f}, RMSE={rmse_naive:.3f}, MAPE={mape_naive * 100:.2f}%")
    print(f"      H-W:   MAE={mae_hw:.3f}, RMSE={rmse_hw:.3f}, MAPE={mape_hw * 100:.2f}%")
    return metrics


def main():
    print("=== Запуск аналізу Lab_7 ===")
    task1_smooth_kings()
    task2_decompose_babyboom()
    y, y_train, y_test = task3_split_series()
    task4_forecast_and_evaluate(y_train, y_test)
    print("=== Усі розрахунки та візуалізації успішно завершено ===")


if __name__ == "__main__":
    main()
