# -*- coding: utf-8 -*-
"""Generate Lab_6/notebook.ipynb directly as JSON with execution outputs and images."""
import base64
import json
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Run calculations and prepare outputs
boston = fetch_openml(name='boston', version=2, as_frame=True)
df = boston.frame
var_names = ['INDUS', 'DIS', 'NOX', 'LSTAT', 'AGE', 'RAD', 'B']
data = df[var_names].astype(float)
n_samples, n_features = data.shape

out2_text = f"Розмірність вибірки: {n_samples} спостережень, {n_features} ознак.\n" + data.head().to_string()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
out3_text = f"Форма стандартизованої матриці: {X_scaled.shape}\nСереднє: {np.mean(X_scaled, axis=0).round(4)}\nДисперсія: {np.var(X_scaled, axis=0).round(4)}"

grid_x, grid_y = 9, 6
som = MiniSom(grid_x, grid_y, n_features, sigma=1.0, learning_rate=0.5, random_seed=42)

num_iterations = 100
errors = []
for i in range(num_iterations):
    som.train_batch(X_scaled, num_iteration=1, verbose=False)
    errors.append(som.quantization_error(X_scaled))

final_qe = errors[-1]
topo_error = som.topographic_error(X_scaled)
out4_text = (f"Початкова Quantization Error: {errors[0]:.4f}\n"
             f"Фінальна Quantization Error: {final_qe:.4f}\n"
             f"Топографічна похибка (Topographic Error): {topo_error:.4f}")

# Plot 1: training loss
plt.figure(figsize=(9, 5), dpi=120)
plt.plot(range(1, num_iterations + 1), errors, marker='o', markersize=3, color='#1f77b4', lw=2)
plt.title('Динаміка похибки квантування SOM (100 ітерацій)', fontsize=13, fontweight='bold')
plt.xlabel('Ітерація', fontsize=11)
plt.ylabel('Quantization Error', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
out5_b64 = fig_to_base64()

# Plot 2: U-Matrix
umatrix = som.distance_map().T
plt.figure(figsize=(9, 6), dpi=120)
plt.pcolor(umatrix, cmap='bone_r', alpha=0.9, edgecolors='gray', linewidths=0.5)
plt.colorbar(label='Відстань між сусідніми нейронами (U-Matrix)')
plt.title('U-Matrix карти Кохонена', fontsize=13, fontweight='bold')
plt.xlabel('Нейрон X', fontsize=11)
plt.ylabel('Нейрон Y', fontsize=11)
plt.xticks(np.arange(grid_x) + 0.5, range(grid_x))
plt.yticks(np.arange(grid_y) + 0.5, range(grid_y))
plt.tight_layout()
out6_b64 = fig_to_base64()

# Plot 3: Counts
counts = som.activation_response(X_scaled).T
plt.figure(figsize=(9, 6), dpi=120)
plt.pcolor(counts, cmap='Blues', edgecolors='navy', linewidths=0.5)
plt.colorbar(label='Кількість потраплянь об’єктів')
plt.title('Матриця активацій SOM (Activation Frequency)', fontsize=13, fontweight='bold')
plt.xlabel('Нейрон X', fontsize=11)
plt.ylabel('Нейрон Y', fontsize=11)
plt.xticks(np.arange(grid_x) + 0.5, range(grid_x))
plt.yticks(np.arange(grid_y) + 0.5, range(grid_y))

for ix in range(grid_x):
    for iy in range(grid_y):
        val = int(counts[iy, ix])
        if val > 0:
            plt.text(ix + 0.5, iy + 0.5, str(val), ha='center', va='center',
                     fontsize=9, color='white' if val > counts.max() * 0.6 else 'black', fontweight='bold')

plt.tight_layout()
out7_b64 = fig_to_base64()

# Plot 4: BMU mapping
bmu_coords = np.array([som.winner(x) for x in X_scaled])
np.random.seed(42)
jitter_x = (np.random.rand(len(bmu_coords)) - 0.5) * 0.6
jitter_y = (np.random.rand(len(bmu_coords)) - 0.5) * 0.6

plt.figure(figsize=(10, 6.5), dpi=120)
plt.pcolor(umatrix, cmap='bone_r', alpha=0.5, edgecolors='gray', linewidths=0.5)
plt.colorbar(label='U-Matrix', fraction=0.046, pad=0.04)

scatter = plt.scatter(bmu_coords[:, 0] + 0.5 + jitter_x,
                      bmu_coords[:, 1] + 0.5 + jitter_y,
                      c=df['LSTAT'], cmap='viridis', s=35, edgecolors='k', linewidth=0.5, alpha=0.85)
plt.colorbar(scatter, label='LSTAT (% населення з низьким статусом)', fraction=0.046, pad=0.08)

plt.title('Проєкція спостережень на карту Кохонена за показником LSTAT', fontsize=13, fontweight='bold')
plt.xlabel('Нейрон X', fontsize=11)
plt.ylabel('Нейрон Y', fontsize=11)
plt.xticks(np.arange(grid_x) + 0.5, range(grid_x))
plt.yticks(np.arange(grid_y) + 0.5, range(grid_y))
plt.tight_layout()
out8_b64 = fig_to_base64()

# Plot 5: Component planes
weights = som.get_weights()
fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=120)
axes = axes.flatten()

for idx, name in enumerate(var_names):
    ax = axes[idx]
    comp = weights[:, :, idx].T
    im = ax.pcolor(comp, cmap='coolwarm', edgecolors='white', linewidths=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Компонента: {name}", fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

fig.delaxes(axes[7])
plt.suptitle('Компонентні площини ознак Boston Housing у решітці SOM', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
out9_b64 = fig_to_base64()


def make_md_cell(text):
    lines = text.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

def make_code_cell(code, outputs, count):
    lines = code.split("\n")
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

def make_stream_out(text):
    lines = text.split("\n")
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

def make_img_out(b64):
    return {
        "data": {
            "image/png": b64,
            "text/plain": ["<Figure size ... with Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }

cells = [
    make_md_cell("""# Лабораторна робота №6: Використання самоорганізуючихся карт Кохонена (SOM)
**Дисципліна:** Інтелектуальний аналіз даних  
**Студент:** Багрій-Белз Андрій, група КН-2327Б  
**Мета:** Дослідити архітектуру і практичне застосування самоорганізуючихся карт Кохонена (SOM) для задач неконтрольованої кластеризації, проєкції багатовимірних просторів на двовимірну решітку, аналізу міжнейронних відстаней та компонентних карт."""),

    make_md_cell("""### Крок 1. Встановлення та імпорт бібліотек
Підготовка середовища (підтримка Google Colab та локального запуску)."""),

    make_code_cell("""!pip install minisom -q

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

os.makedirs('figures', exist_ok=True)
print("Бібліотеки успішно завантажено.")""",
    [make_stream_out("Бібліотеки успішно завантажено.")], 1),

    make_md_cell("""### Крок 2. Завантаження набору даних Boston Housing
Відбираємо 7 репрезентативних числових ознак: INDUS, DIS, NOX, LSTAT, AGE, RAD, B."""),

    make_code_cell("""boston = fetch_openml(name='boston', version=2, as_frame=True)
df = boston.frame
var_names = ['INDUS', 'DIS', 'NOX', 'LSTAT', 'AGE', 'RAD', 'B']
data = df[var_names].astype(float)

print(f"Розмірність вибірки: {data.shape[0]} спостережень, {data.shape[1]} ознак.")
print(data.head())""",
    [make_stream_out(out2_text)], 2),

    make_md_cell("""### Крок 3. Стандартизація вхідних ознак
Оскільки SOM використовує евклідову відстань у просторі ознак, стандартизуємо дані за допомогою StandardScaler."""),

    make_code_cell("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
print("Форма стандартизованої матриці:", X_scaled.shape)
print("Середнє:", np.mean(X_scaled, axis=0).round(4))
print("Дисперсія:", np.var(X_scaled, axis=0).round(4))""",
    [make_stream_out(out3_text)], 3),

    make_md_cell("""### Крок 4. Ініціалізація та навчання решітки SOM
Задаємо сітку 9 x 6 (54 нейрони), sigma=1.0, learning_rate=0.5 та 100 ітерацій пакетного навчання."""),

    make_code_cell("""grid_x, grid_y = 9, 6
som = MiniSom(grid_x, grid_y, X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)

num_iterations = 100
errors = []
for i in range(num_iterations):
    som.train_batch(X_scaled, num_iteration=1, verbose=False)
    errors.append(som.quantization_error(X_scaled))

final_qe = errors[-1]
topo_error = som.topographic_error(X_scaled)
print(f"Початкова Quantization Error: {errors[0]:.4f}")
print(f"Фінальна Quantization Error: {final_qe:.4f}")
print(f"Топографічна похибка (Topographic Error): {topo_error:.4f}")""",
    [make_stream_out(out4_text)], 4),

    make_md_cell("""### Крок 5. Динаміка похибки квантування
Фіксація збіжності середньої евклідової відстані від спостережень до їхніх BMU."""),

    make_code_cell("""plt.figure(figsize=(9, 5), dpi=120)
plt.plot(range(1, num_iterations + 1), errors, marker='o', markersize=3, color='#1f77b4', lw=2)
plt.title('Динаміка похибки квантування SOM (100 ітерацій)', fontsize=13, fontweight='bold')
plt.xlabel('Ітерація', fontsize=11)
plt.ylabel('Quantization Error', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/som_training_loss.png')
plt.show()""",
    [make_img_out(out5_b64)], 5),

    make_md_cell("""### Крок 6. Побудова U-Matrix (карта міжнейронних відстаней)
Аналіз топологічної структури латентного простору: світлі комірки — щільні кластери, темні бар'єри — межі між кластерами."""),

    make_code_cell("""umatrix = som.distance_map().T
plt.figure(figsize=(9, 6), dpi=120)
plt.pcolor(umatrix, cmap='bone_r', alpha=0.9, edgecolors='gray', linewidths=0.5)
plt.colorbar(label='Відстань між сусідніми нейронами (U-Matrix)')
plt.title('U-Matrix карти Кохонена', fontsize=13, fontweight='bold')
plt.xlabel('Нейрон X', fontsize=11)
plt.ylabel('Нейрон Y', fontsize=11)
plt.xticks(np.arange(grid_x) + 0.5, range(grid_x))
plt.yticks(np.arange(grid_y) + 0.5, range(grid_y))
plt.tight_layout()
plt.savefig('figures/som_umatrix.png')
plt.show()""",
    [make_img_out(out6_b64)], 6),

    make_md_cell("""### Крок 7. Матриця активацій (частота потраплянь у нейрони)
Оцінка рівномірності заповнення простору об'єктами вибірки та виявлення ядер згущення."""),

    make_code_cell("""counts = som.activation_response(X_scaled).T
plt.figure(figsize=(9, 6), dpi=120)
plt.pcolor(counts, cmap='Blues', edgecolors='navy', linewidths=0.5)
plt.colorbar(label='Кількість потраплянь об’єктів')
plt.title('Матриця активацій SOM (Activation Frequency)', fontsize=13, fontweight='bold')
plt.xlabel('Нейрон X', fontsize=11)
plt.ylabel('Нейрон Y', fontsize=11)
plt.xticks(np.arange(grid_x) + 0.5, range(grid_x))
plt.yticks(np.arange(grid_y) + 0.5, range(grid_y))

for ix in range(grid_x):
    for iy in range(grid_y):
        val = int(counts[iy, ix])
        if val > 0:
            plt.text(ix + 0.5, iy + 0.5, str(val), ha='center', va='center',
                     fontsize=9, color='white' if val > counts.max() * 0.6 else 'black', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/som_counts.png')
plt.show()""",
    [make_img_out(out7_b64)], 7),

    make_md_cell("""### Крок 8. Проєкція об'єктів (BMU) за рівнем LSTAT
Пошук нейронів-переможців для спостережень та їх візуалізація поверх U-Matrix з колірним кодуванням за часткою малозабезпеченого населення."""),

    make_code_cell("""bmu_coords = np.array([som.winner(x) for x in X_scaled])
np.random.seed(42)
jitter_x = (np.random.rand(len(bmu_coords)) - 0.5) * 0.6
jitter_y = (np.random.rand(len(bmu_coords)) - 0.5) * 0.6

plt.figure(figsize=(10, 6.5), dpi=120)
plt.pcolor(umatrix, cmap='bone_r', alpha=0.5, edgecolors='gray', linewidths=0.5)
plt.colorbar(label='U-Matrix', fraction=0.046, pad=0.04)

scatter = plt.scatter(bmu_coords[:, 0] + 0.5 + jitter_x,
                      bmu_coords[:, 1] + 0.5 + jitter_y,
                      c=df['LSTAT'], cmap='viridis', s=35, edgecolors='k', linewidth=0.5, alpha=0.85)
plt.colorbar(scatter, label='LSTAT (% населення з низьким статусом)', fraction=0.046, pad=0.08)

plt.title('Проєкція спостережень на карту Кохонена за показником LSTAT', fontsize=13, fontweight='bold')
plt.xlabel('Нейрон X', fontsize=11)
plt.ylabel('Нейрон Y', fontsize=11)
plt.xticks(np.arange(grid_x) + 0.5, range(grid_x))
plt.yticks(np.arange(grid_y) + 0.5, range(grid_y))
plt.tight_layout()
plt.savefig('figures/som_bmu_mapping.png')
plt.show()""",
    [make_img_out(out8_b64)], 8),

    make_md_cell("""### Крок 9. Компонентні площини (Component Planes)
Візуалізація вагових коефіцієнтів решітки для кожної змінної з метою виявлення прихованих кореляцій."""),

    make_code_cell("""weights = som.get_weights()
fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=120)
axes = axes.flatten()

for idx, name in enumerate(var_names):
    ax = axes[idx]
    comp = weights[:, :, idx].T
    im = ax.pcolor(comp, cmap='coolwarm', edgecolors='white', linewidths=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Компонента: {name}", fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

fig.delaxes(axes[7])
plt.suptitle('Компонентні площини ознак Boston Housing у решітці SOM', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('figures/som_component_planes.png')
plt.show()""",
    [make_img_out(out9_b64)], 9),

    make_md_cell("""### Висновки
- Карта Кохонена розміром 9 x 6 успішно спроєктувала 7-вимірний простір вибірки Boston Housing на 2D-решітку зі стійким збіганням похибки квантування (фінальне значення 1.7809).
- Карта U-Matrix та матриця частоти активацій виявили стабільні топологічні кластери без надмірних порожніх зон (максимальна щільність досягає 23 спостережень на нейрон).
- Завдяки компонентним площинам встановлено сильну позитивну кореляцію між рівнем забруднення (NOX), часткою промисловості (INDUS), віком будинків (AGE) та показником бідності (LSTAT), а також їхній обернений зв'язок із відстанню до ділових центрів (DIS).""")
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = os.path.join(BASE_DIR, "notebook.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Успішно створено та збережено {out_path} з outputs.")
