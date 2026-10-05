# -*- coding: utf-8 -*-
"""
Lab 6: Self-Organizing Maps (SOM) analysis on Boston Housing dataset.
Generates:
  - figures/som_training_loss.png
  - figures/som_umatrix.png
  - figures/som_counts.png
  - figures/som_bmu_mapping.png
  - figures/som_component_planes.png
  - report.json
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Завантаження даних
print("[1/5] Завантаження Boston dataset...")
boston = fetch_openml(name='boston', version=2, as_frame=True)
df = boston.frame

var_names = ['INDUS', 'DIS', 'NOX', 'LSTAT', 'AGE', 'RAD', 'B']
data = df[var_names].astype(float)
n_samples, n_features = data.shape
print(f"  Кількість спостережень: {n_samples}, ознак: {n_features} ({var_names})")

# 2. Стандартизація ознак
print("[2/5] Стандартизація даних (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 3. Ініціалізація та навчання SOM
print("[3/5] Ініціалізація та навчання SOM...")
grid_x, grid_y = 9, 6
som = MiniSom(grid_x, grid_y, n_features, sigma=1.0, learning_rate=0.5, random_seed=42)

num_iterations = 100
errors = []
for i in range(num_iterations):
    som.train_batch(X_scaled, num_iteration=1, verbose=False)
    errors.append(som.quantization_error(X_scaled))

final_qe = float(errors[-1])
topo_error = float(som.topographic_error(X_scaled))
print(f"  Початкова QE: {errors[0]:.4f}, Фінальна QE: {final_qe:.4f}")
print(f"  Топографічна похибка (Topographic error): {topo_error:.4f}")

# 4. Побудова графіків
print("[4/5] Побудова графіків...")

# Графік 1: som_training_loss.png
fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
ax.plot(range(1, num_iterations + 1), errors, color='#1f77b4', lw=2, marker='o', markersize=3, label='Quantization Error')
ax.set_title('Динаміка похибки квантування SOM (100 ітерацій)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Ітерація', fontsize=12)
ax.set_ylabel('Quantization Error (середня відстань до BMU)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'som_training_loss.png'))
plt.close(fig)

# Графік 2: som_umatrix.png
umatrix = som.distance_map().T
fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
im = ax.pcolor(umatrix, cmap='bone_r', alpha=0.9, edgecolors='gray', linewidths=0.5)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Відстань між сусідніми нейронами (U-Matrix)', fontsize=11)
ax.set_title('U-Matrix карти Кохонена (розподіл меж та кластерних зон)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Нейрон X (0-8)', fontsize=11)
ax.set_ylabel('Нейрон Y (0-5)', fontsize=11)
ax.set_xticks(np.arange(grid_x) + 0.5)
ax.set_xticklabels(range(grid_x))
ax.set_yticks(np.arange(grid_y) + 0.5)
ax.set_yticklabels(range(grid_y))
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'som_umatrix.png'))
plt.close(fig)

# Графік 3: som_counts.png
counts = som.activation_response(X_scaled).T
fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
im = ax.pcolor(counts, cmap='Blues', edgecolors='navy', linewidths=0.5)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Кількість потраплянь об’єктів (Counts)', fontsize=11)
ax.set_title('Матриця активацій SOM (частота потраплянь об’єктів у нейрони)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Нейрон X (0-8)', fontsize=11)
ax.set_ylabel('Нейрон Y (0-5)', fontsize=11)
ax.set_xticks(np.arange(grid_x) + 0.5)
ax.set_xticklabels(range(grid_x))
ax.set_yticks(np.arange(grid_y) + 0.5)
ax.set_yticklabels(range(grid_y))
# Додамо текстові підписи кількості потраплянь у клітинки
for ix in range(grid_x):
    for iy in range(grid_y):
        val = int(counts[iy, ix])
        if val > 0:
            ax.text(ix + 0.5, iy + 0.5, str(val), ha='center', va='center',
                    fontsize=9, color='white' if val > counts.max() * 0.6 else 'black', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'som_counts.png'))
plt.close(fig)

# Графік 4: som_bmu_mapping.png
bmu_coords = np.array([som.winner(x) for x in X_scaled])
# Додаємо невеликий випадковий jitter для розрізнення точок в одному нейроні
np.random.seed(42)
jitter_x = (np.random.rand(len(bmu_coords)) - 0.5) * 0.6
jitter_y = (np.random.rand(len(bmu_coords)) - 0.5) * 0.6

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
im = ax.pcolor(umatrix, cmap='bone_r', alpha=0.5, edgecolors='gray', linewidths=0.5)
cbar1 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label('U-Matrix (структура відстаней)', fontsize=10)

scatter = ax.scatter(bmu_coords[:, 0] + 0.5 + jitter_x,
                     bmu_coords[:, 1] + 0.5 + jitter_y,
                     c=df['LSTAT'], cmap='viridis', s=35, edgecolors='k', linewidth=0.5, alpha=0.85)
cbar2 = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.08)
cbar2.set_label('LSTAT (% населення з низьким соц.-екон. статусом)', fontsize=10)

ax.set_title('Проєкція спостережень на карту Кохонена (BMU) з розфарбуванням за LSTAT', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Нейрон X (0-8)', fontsize=11)
ax.set_ylabel('Нейрон Y (0-5)', fontsize=11)
ax.set_xticks(np.arange(grid_x) + 0.5)
ax.set_xticklabels(range(grid_x))
ax.set_yticks(np.arange(grid_y) + 0.5)
ax.set_yticklabels(range(grid_y))
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'som_bmu_mapping.png'))
plt.close(fig)

# Графік 5: som_component_planes.png
weights = som.get_weights()
fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=200)
axes = axes.flatten()

for idx, name in enumerate(var_names):
    ax = axes[idx]
    comp = weights[:, :, idx].T
    im = ax.pcolor(comp, cmap='coolwarm', edgecolors='white', linewidths=0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Компонента: {name}", fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# Видаляємо останній 8-й підграфік (оскільки ознак 7)
fig.delaxes(axes[7])
plt.suptitle('Компонентні площини (Component Planes) ознак Boston Housing у просторі SOM', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'som_component_planes.png'))
plt.close(fig)

# 5. Збереження report.json
print("[5/5] Збереження report.json...")
report_data = {
    "grid_size": [grid_x, grid_y],
    "n_neurons": grid_x * grid_y,
    "features": var_names,
    "n_samples": n_samples,
    "iterations": num_iterations,
    "learning_rate": 0.5,
    "sigma": 1.0,
    "initial_quantization_error": float(errors[0]),
    "final_quantization_error": final_qe,
    "topographic_error": topo_error,
    "max_neuron_activations": int(counts.max()),
    "empty_neurons_count": int(np.sum(counts == 0)),
    "figures": {
        "training_loss": "figures/som_training_loss.png",
        "umatrix": "figures/som_umatrix.png",
        "counts": "figures/som_counts.png",
        "bmu_mapping": "figures/som_bmu_mapping.png",
        "component_planes": "figures/som_component_planes.png"
    }
}

with open(os.path.join(BASE_DIR, 'report.json'), 'w', encoding='utf-8') as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)

print("Розрахунки успішно завершено! Всі артефакти збережено.")
