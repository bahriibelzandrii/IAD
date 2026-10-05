# -*- coding: utf-8 -*-
"""
Lab 5: Кластерний аналіз даних у Python
- Частина 1: Метод K-середніх (KMeans) на синтетичних даних про фрукти (вага та ціна)
- Дослідження оптимального k методом ліктя та силуету
- Частина 2: Ієрархічна агломеративна кластеризація (stocks.csv, 50 спостережень)
- Кластеризація спостережень (дат) та кластеризація тікерів акцій
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print("=== Етап 1: K-середніх на синтетичних даних ===")
    np.random.seed(42)
    # 3 групи фруктів з різними характеристиками ваги і ціни
    # 1: легкі і дешеві (ягоди / цитрусові)
    # 2: середні за вагою та дорожчі
    # 3: важкі та помірні за ціною (дині, кавуни, ананаси)
    c1_weights = np.random.normal(loc=70, scale=12, size=40)
    c1_prices = np.random.normal(loc=3.5, scale=0.8, size=40)

    c2_weights = np.random.normal(loc=130, scale=15, size=35)
    c2_prices = np.random.normal(loc=8.5, scale=1.0, size=35)

    c3_weights = np.random.normal(loc=190, scale=18, size=45)
    c3_prices = np.random.normal(loc=4.5, scale=0.9, size=45)

    weights = np.concatenate([c1_weights, c2_weights, c3_weights])
    prices = np.concatenate([c1_prices, c2_prices, c3_prices])

    fruits_data = pd.DataFrame({"Вага": weights, "Ціна": prices})
    X_fruits = fruits_data[["Вага", "Ціна"]]

    # Навчання KMeans(n_clusters=3, random_state=42)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    fruits_data["Кластер"] = kmeans.fit_predict(X_fruits)
    centroids = kmeans.cluster_centers_

    # Графік кластерів KMeans
    plt.figure(figsize=(8, 6), dpi=150)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    cluster_names = ["Кластер 0 (Легкі / Низька ціна)",
                     "Кластер 1 (Середні / Висока ціна)",
                     "Кластер 2 (Важкі / Помірна ціна)"]
    for i in range(3):
        subset = fruits_data[fruits_data["Кластер"] == i]
        plt.scatter(
            subset["Вага"],
            subset["Ціна"],
            c=colors[i],
            label=f"Кластер {i}",
            alpha=0.75,
            edgecolors="k",
            s=55,
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        c="red",
        s=220,
        linewidths=2,
        edgecolors="black",
        label="Центроїди",
        zorder=10,
    )
    plt.title("Кластеризація фруктів методом K-середніх (k = 3)", fontsize=13, fontweight="bold")
    plt.xlabel("Вага (г)", fontsize=11)
    plt.ylabel("Ціна ($/кг)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    kmeans_fig_path = os.path.join(FIG_DIR, "kmeans_clusters.png")
    plt.savefig(kmeans_fig_path)
    plt.close()
    print(f"Збережено: {kmeans_fig_path}")

    # Дослідження оптимального k методом ліктя та силуету
    print("\n=== Етап 2: Метод ліктя та аналіз силуету ===")
    k_range = list(range(1, 11))
    wcss = []
    silhouette_scores = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_fruits)
        wcss.append(float(km.inertia_))
        if k >= 2:
            score = float(silhouette_score(X_fruits, labels))
            silhouette_scores[k] = score

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    color = "tab:blue"
    ax1.set_xlabel("Кількість кластерів (k)", fontsize=11)
    ax1.set_ylabel("Інерція WCSS", color=color, fontsize=11)
    line1 = ax1.plot(k_range, wcss, marker="o", color=color, linewidth=2, label="WCSS (Лікоть)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(k_range)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Коефіцієнт силуету (Silhouette)", color=color, fontsize=11)
    k_sil = list(silhouette_scores.keys())
    scores_sil = list(silhouette_scores.values())
    line2 = ax2.plot(k_sil, scores_sil, marker="s", color=color, linestyle="--", linewidth=2, label="Силует")
    ax2.tick_params(axis="y", labelcolor=color)

    # Об'єднана легенда
    lines = line1 + line2
    labels_leg = [l.get_label() for l in lines]
    ax1.legend(lines, labels_leg, loc="upper right")

    plt.title("Визначення оптимального k (WCSS та Silhouette Score)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    elbow_fig_path = os.path.join(FIG_DIR, "elbow_method.png")
    plt.savefig(elbow_fig_path)
    plt.close()
    print(f"Збережено: {elbow_fig_path}")

    # Етап 3: Ієрархічна кластеризація stocks.csv (спостереження)
    print("\n=== Етап 3: Ієрархічна кластеризація спостережень stocks.csv ===")
    stocks_path = os.path.join(BASE_DIR, "stocks.csv")
    df_stocks = pd.read_csv(stocks_path)
    df_sub = df_stocks.head(50)
    dates = df_sub["DATE"].astype(str).tolist()
    X_stocks = df_sub.drop(columns=["DATE"])

    # Дендрограма спостережень (дати)
    Z_obs = linkage(X_stocks, method="ward", metric="euclidean")

    plt.figure(figsize=(15, 8), dpi=150)
    plt.title("Дендрограма ієрархічної кластеризації спостережень котирувань акцій (Метод Уорда)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Дата / Номер спостереження", fontsize=11)
    plt.ylabel("Евклідова відстань", fontsize=11)
    dendrogram(Z_obs, labels=dates, leaf_rotation=90, leaf_font_size=8)
    plt.tight_layout()
    stocks_dendro_path = os.path.join(FIG_DIR, "stocks_dendrogram.png")
    plt.savefig(stocks_dendro_path)
    plt.close()
    print(f"Збережено: {stocks_dendro_path}")

    # Етап 4: Ієрархічна кластеризація тікерів акцій
    print("\n=== Етап 4: Ієрархічна кластеризація тікерів акцій ===")
    # Стандартизація часових рядів кожного тікера (щоб порівнювати динаміку форми, а не абсолютні ціни)
    scaler = StandardScaler()
    X_stocks_scaled = scaler.fit_transform(X_stocks)

    # Транспонуємо: рядки - компанії (тікери), стовпці - дні
    tickers = list(X_stocks.columns)
    Z_tickers = linkage(X_stocks_scaled.T, method="ward", metric="euclidean")

    plt.figure(figsize=(12, 7), dpi=150)
    plt.title("Дендрограма ієрархічної кластеризації тікерів акцій (Метод Уорда, стандартизовано)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Тікери компаній", fontsize=11)
    plt.ylabel("Евклідова відстань", fontsize=11)
    dendrogram(Z_tickers, labels=tickers, leaf_rotation=45, leaf_font_size=10)
    plt.tight_layout()
    tickers_dendro_path = os.path.join(FIG_DIR, "tickers_dendrogram.png")
    plt.savefig(tickers_dendro_path)
    plt.close()
    print(f"Збережено: {tickers_dendro_path}")

    # Етап 5: Збереження report.json
    report_data = {
        "kmeans": {
            "n_clusters": 3,
            "centroids": centroids.tolist(),
            "wcss": {int(k): round(v, 2) for k, v in zip(k_range, wcss)},
            "silhouette_scores": {int(k): round(v, 4) for k, v in silhouette_scores.items()},
            "best_k_silhouette": int(max(silhouette_scores, key=silhouette_scores.get)),
        },
        "stocks": {
            "n_samples": len(df_sub),
            "n_features": len(tickers),
            "tickers": tickers,
            "dates_range": [dates[0], dates[-1]],
            "linkage_method": "ward",
            "distance_metric": "euclidean"
        }
    }

    report_json_path = os.path.join(BASE_DIR, "report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Збережено: {report_json_path}")
    print("Усі етапи аналізу завершено успішно.")


if __name__ == "__main__":
    main()
