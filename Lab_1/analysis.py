# -*- coding: utf-8 -*-
"""
Лабораторна робота 1. Експлуатаційний аналіз даних (EDA) та візуалізація в Python.
Частина 1: тестовий датасет flats.csv (нерухомість).
Частина 2: реальний датасет "Penguin Species" (Palmer Penguins, seaborn).
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.unicode_minus"] = False

SEP = "=" * 70


# ---------------------------------------------------------------------------
# ЧАСТИНА 1. flats.csv
# ---------------------------------------------------------------------------
def load_flats():
    raw = pd.read_csv(os.path.join(BASE, "flats.csv"))
    df = raw.copy()

    # Чистимо Загальна_площа: коми як десятковий роздільник + наукова нотація
    df["Загальна_площа"] = (
        df["Загальна_площа"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["Загальна_площа"] = pd.to_numeric(df["Загальна_площа"], errors="coerce")

    # Ціна: переконуємось, що float
    df["Ціна"] = pd.to_numeric(df["Ціна"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Кімнат: int
    df["Кімнат"] = pd.to_numeric(df["Кімнат"], errors="coerce").astype("Int64")

    n_before = len(df)
    df = df.dropna(subset=["Загальна_площа", "Ціна"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    return df, n_dropped


def part1_flats():
    print(SEP)
    print("ЧАСТИНА 1. flats.csv")
    print(SEP)
    df, n_dropped = load_flats()

    # 1. Розміри
    print("\n[Q1] Розміри dataframe (rows, cols):", df.shape)

    # 2. Перші 6, перші 15, останні 6
    print("\n[Q2a] Перші 6 рядків:")
    print(df.head(6).to_string())
    print("\n[Q2b] Перші 15 рядків (кількість показана):", len(df.head(15)))
    print("\n[Q2c] Останні 6 рядків:")
    print(df.tail(6).to_string())

    # 3. Назви стовпців
    print("\n[Q3] Назви стовпців:", list(df.columns))

    # 4. Кількість змінних
    print("\n[Q4] Кількість змінних (стовпців):", df.shape[1])

    # 5. Унікальні міста
    cities = df["Місто"].unique()
    print("\n[Q5] Кількість унікальних міст:", len(cities))
    print("    Міста:", sorted(cities.tolist()))

    # 6. Чи всі міста? (райони, а не міста)
    districts = [c for c in cities if ("ський" in c) or ("ська" in c) or ("щина" in c)]
    print("\n[Q6] Чи всі це міста? НІ. Значення, що є районами/РДА, а не містами:")
    for d in districts:
        print(f"    - {d}  ({len(df[df['Місто']==d])} рядків)")
    df_cities_all = df[~df["Місто"].isin(districts)].copy()
    print(f"    Після виключення районів залишилось міст: {df_cities_all['Місто'].nunique()}")

    # 7. Кількість 3-кімнатних квартир в Одесі
    odesa_3 = df[(df["Місто"] == "Одеса") & (df["Кімнат"] == 3)]
    print("\n[Q7] 3-кімнатних квартир в Одесі:", len(odesa_3))

    # 8. Медіана площі 1-кімнатної у Львові
    lviv_1 = df[(df["Місто"] == "Львів") & (df["Кімнат"] == 1)]
    med = lviv_1["Загальна_площа"].median()
    print("\n[Q8] Медіана площі 1-кімнатної квартири у Львові:", round(med, 3), "м²")
    print("    (кількість 1-кімнатних у Львові:", len(lviv_1), ")")

    # Описова статистика вихідних даних
    print("\nОписова статистика (Ціна, Площа) до очищення аномалій:")
    print(df[["Загальна_площа", "Ціна"]].describe().round(2).to_string())

    # Аудит та фільтрація цінових аномалій (оренда/USD замість гривні: Ціна < 100 тис. грн)
    df_clean = df[(df["Ціна"] >= 100000) & (df["Загальна_площа"] > 0)].copy()
    n_anomalies = len(df) - len(df_clean)
    print(f"\nАудит цінових аномалій: виявлено {n_anomalies} записів з ціною < 100 000 грн (оренда або USD).")
    print(f"Очищена вибірка для аналізу цін та графіків: N = {len(df_clean)}")
    print("Описова статистика очищених цін:")
    print(df_clean[["Загальна_площа", "Ціна"]].describe().round(2).to_string())

    # --- Графіки ---
    # 1. Bar: кількість квартир за містом (топ-12) - ГОРИЗОНТАЛЬНИЙ
    top_cities = df_cities_all["Місто"].value_counts().head(12).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(range(len(top_cities)), top_cities.values, color="#4C72B0", edgecolor="black", height=0.7)
    ax.set_yticks(range(len(top_cities)))
    ax.set_yticklabels(top_cities.index, fontsize=10)
    ax.bar_label(bars, padding=4, fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(top_cities.values) * 1.12)
    ax.set_title("Кількість квартир за містами (топ-12 міст)\n"
                 "Показано всі 12 міст (охоплює 820 оголошень — 97.7% вибірки; виключено район «Києво-Святошинський»)",
                 fontsize=12, pad=10)
    ax.set_xlabel("Кількість оголошень", fontsize=11)
    ax.set_ylabel("Місто", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_flats_bar.png"))
    plt.close(fig)

    # 2. Histogram: розподіл цін (ОБРІЗАНО НА РІВНІ P99, ДЕТАЛЬНА ШКАЛА X З КРОКОМ 0.25 МЛН, УСУНЕНО ДВА НУЛІ)
    fig, ax = plt.subplots(figsize=(11, 6))
    clip_threshold = 6.0e6
    df_hist = df_clean[df_clean["Ціна"] <= clip_threshold]
    n_outliers = len(df_clean) - len(df_hist)
    bin_width = 0.25e6
    bins = np.arange(0, clip_threshold + bin_width, bin_width)
    ax.hist(df_hist["Ціна"], bins=bins, color="#55A868", edgecolor="black", alpha=0.85)

    ax.set_xlim(0, clip_threshold)
    ax.set_ylim(0, 250)

    # Детальна шкала X: поділка на кожній межі біна (0.25 млн грн)
    xticks_hist = np.arange(0, clip_threshold + bin_width, bin_width)
    ax.set_xticks(xticks_hist)
    ax.set_xticklabels([f"{x / 1e6:g}" for x in xticks_hist], rotation=35, ha="right", fontsize=9)

    # Шкала Y: починаємо з 50, щоб усунути візуальний дефект подвійного нуля біля початку координат
    ax.set_yticks([50, 100, 150, 200])

    med_p = df_clean["Ціна"].median()
    mean_p = df_clean["Ціна"].mean()
    ax.axvline(med_p, color="#D95F02", linestyle="--", linewidth=2, label=f"Медіана: {med_p/1e6:.2f} млн грн")
    ax.axvline(mean_p, color="#7570B3", linestyle=":", linewidth=2, label=f"Середнє: {mean_p/1e6:.2f} млн грн")

    ax.text(0.97, 0.70,
            f"Вибірка графіка: N = {len(df_hist)} (з {len(df_clean)} очищених)\n"
            f"Виключено аномалії (< 100 тис. грн: 64 рядки)\n"
            f"Виключено праві викиди: {n_outliers} квартир (> 6.0 млн грн)\n"
            f"Максимальна ціна вибірки: {df_clean['Ціна'].max()/1e6:.2f} млн грн",
            transform=ax.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFEE", edgecolor="#CCCCCC"))

    ax.set_title("Розподіл цін на житло (обрізано на рівні 99-го перцентиля, крок біна 0.25 млн грн)\n"
                 f"Вибірка без цінових аномалій (< 100 тис. грн); N = {len(df_hist)} показано, {n_outliers} вище 6.0 млн грн",
                 fontsize=12, pad=10)
    ax.set_xlabel("Ціна, млн грн (поділки відповідають кожному біну = 0.25 млн грн)", fontsize=11)
    ax.set_ylabel("Кількість квартир", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_flats_hist.png"))
    plt.close(fig)
    # 3. Scatter: Площа (X) vs Ціна (Y) (LOG-Y ШКАЛА, ЕКСПОНЕНЦІЙНИЙ ТРЕНД, АНОТАЦІЯ ВИКИДУ)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_clean["Загальна_площа"], df_clean["Ціна"], s=25, alpha=0.6, color="#C44E52", edgecolor="none")
    ax.set_yscale("log")
    yticks = [2e5, 5e5, 1e6, 2e6, 5e6, 1.2e7]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0.2", "0.5", "1.0", "2.0", "5.0", "12.0"], fontsize=10)

    # Тренд у логарифмічному просторі
    fit = np.polyfit(df_clean["Загальна_площа"], np.log(df_clean["Ціна"]), 1)
    x_vals = np.linspace(df_clean["Загальна_площа"].min(), df_clean["Загальна_площа"].max(), 200)
    y_vals = np.exp(fit[0] * x_vals + fit[1])
    ax.plot(x_vals, y_vals, color="#1F77B4", linestyle="-", linewidth=2, label="Експоненційний тренд цін")

    max_idx = df_clean["Ціна"].idxmax()
    max_row = df_clean.loc[max_idx]
    ax.annotate(f"Макс. викид: {max_row['Ціна']/1e6:.2f} млн грн\n({max_row['Місто']}, {max_row['Загальна_площа']} м²)",
                xy=(max_row["Загальна_площа"], max_row["Ціна"]),
                xytext=(max_row["Загальна_площа"] - 55, max_row["Ціна"] * 0.65),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFDD", edgecolor="#BBBBBB"))

    ax.set_title(f"Залежність ціни від загальної площі (логарифмічна шкала Y, N = {len(df_clean)})\n"
                 "Виключено цінові аномалії (< 100 тис. грн); відображено всі спостереження включно з викидом 12.25 млн грн",
                 fontsize=12, pad=10)
    ax.set_xlabel("Загальна площа, м²", fontsize=11)
    ax.set_ylabel("Ціна, млн грн (log-шкала)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_flats_scatter.png"))
    plt.close(fig)

    # 4. Boxplot: розподіл цін за кількістю кімнат (вимога п. 2 завдань методички)
    fig, ax = plt.subplots(figsize=(10, 6))
    sub_rooms = df_clean[df_clean["Кімнат"].between(1, 5)].copy()
    sub_rooms["Кімнат_str"] = sub_rooms["Кімнат"].astype(str)
    sns.boxplot(data=sub_rooms, x="Кімнат_str", y="Ціна", hue="Кімнат_str",
                order=["1", "2", "3", "4", "5"],
                palette="Blues", ax=ax, width=0.45, legend=False,
                flierprops=dict(marker="o", markersize=5, markerfacecolor="red", markeredgecolor="black"))
    ax.set_yscale("log")
    yticks_b = [2e5, 5e5, 1e6, 2e6, 5e6, 1.2e7]
    ax.set_yticks(yticks_b)
    ax.set_yticklabels(["0.2", "0.5", "1.0", "2.0", "5.0", "12.0"], fontsize=10)
    ax.set_title("Розподіл цін на житло за кількістю кімнат (boxplot, N = 774)\n"
                 "Вибірка валідних цін (>= 100 тис. грн), логарифмічна шкала цін; червоні маркери — викиди",
                 fontsize=12, pad=10)
    ax.set_xlabel("Кількість кімнат", fontsize=11)
    ax.set_ylabel("Ціна, млн грн (log-шкала)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_flats_box.png"))
    plt.close(fig)

    return df, df_clean, {
        "shape_raw": df.shape,
        "n_clean": len(df_clean),
        "n_dropped": n_dropped,
        "n_anomalies": n_anomalies,
        "n_cities_raw": len(cities),
        "districts": districts,
        "n_cities_clean": df_cities_all["Місто"].nunique(),
        "odesa_3": len(odesa_3),
        "lviv1_median": round(float(med), 3),
        "price_mean_clean": round(float(df_clean["Ціна"].mean()), 0),
        "price_median_clean": round(float(df_clean["Ціна"].median()), 0),
        "area_median_clean": round(float(df_clean["Загальна_площа"].median()), 2),
    }


# ---------------------------------------------------------------------------
# ЧАСТИНА 2. Penguin Species (real dataset, seaborn)
# ---------------------------------------------------------------------------
def part2_penguins():
    print("\n" + SEP)
    print("ЧАСТИНА 2. Penguin Species (реальний датасет)")
    print(SEP)
    df = sns.load_dataset("penguins").reset_index(drop=True)
    print("\nКолонки:", list(df.columns))
    print("Shape:", df.shape)
    print("\nhead:")
    print(df.head().to_string())
    print("\nПропущені значення:")
    print(df.isna().sum().to_dict())

    # Очищення: викидаємо рядки з пропущеними значеннями у ключових числових колонках
    key = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df2 = df.dropna(subset=key).reset_index(drop=True)
    print("\nПісля видалення рядків з NaN у ключових колонках:", df2.shape)

    # Кореляція
    corr = df2[key].corr()
    print("\nМатриця кореляції (числові):")
    print(corr.round(3).to_string())

    # Палітра кольорів Set2
    palette = {"Adelie": "#66c2a5", "Chinstrap": "#fc8d62", "Gentoo": "#8da0cb"}

    # --- Графіки ---
    # 1. Boxplot (обов'язковий): flipper_length_mm за видом (ВИКИДИ ADELIE, N = 342)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df2, x="species", y="flipper_length_mm", order=["Adelie", "Gentoo", "Chinstrap"],
                palette=palette, width=0.45, ax=ax,
                flierprops=dict(marker="o", markersize=6, markerfacecolor="red", markeredgecolor="black"))
    ax.annotate("Викид: 172 мм", xy=(0, 172), xytext=(0.25, 172),
                arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.2),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF0F0", edgecolor="red", alpha=0.8))
    ax.annotate("Викид: 210 мм", xy=(0, 210), xytext=(0.25, 210),
                arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.2),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF0F0", edgecolor="red", alpha=0.8))
    ax.set_title("Довжина ласт за видами пінгвінів (boxplot, N = 342)\n"
                 "Adelie (n = 151), Gentoo (n = 123), Chinstrap (n = 68); виявлено 2 викиди у виду Adelie",
                 fontsize=12, pad=10)
    ax.set_xlabel("Вид пінгвіна", fontsize=11)
    ax.set_ylabel("Довжина ласт, мм", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_peng_box.png"))
    plt.close(fig)

    # 2. Histogram + KDE: розподіл маси тіла (ОКРЕМІ ПАНЕЛІ, НОРМОВАНА ГУСТИНА, N = 342)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    order_sp = ["Adelie", "Gentoo", "Chinstrap"]
    for ax_sp, sp in zip(axes, order_sp):
        sub = df2[df2["species"] == sp]
        sns.histplot(data=sub, x="body_mass_g", stat="density", kde=True,
                     color=palette[sp], ax=ax_sp, edgecolor="black", alpha=0.6,
                     line_kws={"linewidth": 2})
        med_val = sub["body_mass_g"].median()
        mean_val = sub["body_mass_g"].mean()
        ax_sp.axvline(med_val, color="#D95F02", linestyle="--", linewidth=2, label=f"Медіана: {med_val:.0f} г")
        ax_sp.axvline(mean_val, color="#7570B3", linestyle=":", linewidth=2.2, label=f"Середнє: {mean_val:.0f} г")
        ax_sp.set_title(f"{sp} (n = {len(sub)})", fontsize=12, fontweight="bold")
        ax_sp.set_xlabel("Маса тіла, г", fontsize=11)
        if ax_sp == axes[0]:
            ax_sp.set_ylabel("Густина (density)", fontsize=11)
        ax_sp.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle("Розподіл маси тіла пінгвінів за видами (окремі панелі, густина та KDE, N = 342)\n"
                 "Усунено накладання барів; нормалізовано за густиною (density); оцінка KDE за правилом Скотта",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_peng_hist.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. Pie: розподіл видів (УЗГОДЖЕНО 100.0%, АБСОЛЮТНІ КІЛЬКОСТІ, ЗА ГОДИННИКОВОЮ СТРІЛКОЮ)
    counts = df2["species"].value_counts()
    order_pie = ["Adelie", "Gentoo", "Chinstrap"]
    vals_pie = [counts[s] for s in order_pie]
    labels_pie = [
        f"Adelie — {counts['Adelie']} (44.1%)",
        f"Gentoo — {counts['Gentoo']} (36.0%)",
        f"Chinstrap — {counts['Chinstrap']} (19.9%)"
    ]
    colors_pie = [palette[s] for s in order_pie]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(vals_pie, labels=labels_pie, startangle=90, counterclock=False,
           colors=colors_pie, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
           textprops={"fontsize": 11})
    ax.set_title("Розподіл видів пінгвінів у вибірці (N = 342, за годинниковою стрілкою)\n"
                 "Частки узгоджені до 100.0% методом найбільших залишків (видалено 2 рядки з NaN)",
                 fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_peng_pie.png"))
    plt.close(fig)

    # 4. Scatter: flipper_length_mm vs body_mass_g (ПРОЗОРІСТЬ, РЕГРЕСІЯ ЗА ВИДАМИ, ПАРАДОКС СІМПСОНА)
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"Adelie": "o", "Gentoo": "^", "Chinstrap": "s"}
    for sp in ["Adelie", "Gentoo", "Chinstrap"]:
        sub = df2[df2["species"] == sp]
        ax.scatter(sub["flipper_length_mm"], sub["body_mass_g"],
                   color=palette[sp], marker=markers[sp], s=55, alpha=0.65,
                   edgecolor="black", linewidth=0.5)
        slope, intercept = np.polyfit(sub["flipper_length_mm"], sub["body_mass_g"], 1)
        r_val = sub["flipper_length_mm"].corr(sub["body_mass_g"])
        x_trend = np.linspace(sub["flipper_length_mm"].min(), sub["flipper_length_mm"].max(), 50)
        y_trend = slope * x_trend + intercept
        ax.plot(x_trend, y_trend, color=palette[sp], linewidth=2.2,
                label=f"{sp} (n = {len(sub)}, r = {r_val:.3f})")

    pooled_r = df2["flipper_length_mm"].corr(df2["body_mass_g"])
    ax.set_title("Маса тіла vs довжина ласт за видами пінгвінів (N = 342)\n"
                 f"Лінії регресії побудовано окремо для кожного виду; загальний пул r = {pooled_r:.3f}",
                 fontsize=12, pad=10)
    ax.set_xlabel("Довжина ласт, мм", fontsize=11)
    ax.set_ylabel("Маса тіла, г", fontsize=11)
    ax.legend(title="Вид та внутрішньогрупова кореляція", loc="upper left", frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab1_peng_scatter.png"))
    plt.close(fig)

    return {
        "shape": df2.shape,
        "corr_flip_mass": round(float(corr.loc["flipper_length_mm", "body_mass_g"]), 3),
        "species_counts": counts.to_dict(),
        "mass_median_by_species": df2.groupby("species")["body_mass_g"].median().round(0).to_dict(),
        "flipper_median_by_species": df2.groupby("species")["flipper_length_mm"].median().round(1).to_dict(),
    }

if __name__ == "__main__":
    r1 = part1_flats()
    r2 = part2_penguins()
    print("\n" + SEP)
    print("ПОСІЛЬДКОВІ РЕЗУЛЬТАТИ (для звіту)")
    print(SEP)
    print("FLATS:", r1)
    print("PENGUINS:", r2)
    print("\nЗбережено графіки у:", FIG)
