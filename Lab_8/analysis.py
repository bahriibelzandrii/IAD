import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import apriori

# 1. Створення директорій для графіків та звітів
os.makedirs("figures", exist_ok=True)
os.makedirs("report", exist_ok=True)

# 2. Завантаження та перетворення транзакцій
df = pd.read_csv("Basket_data.csv", header=None)
transactions = []
for i in range(len(df)):
    row = [str(x).strip() for x in df.iloc[i].values if pd.notna(x) and str(x).strip() != '' and str(x).strip().lower() != 'nan']
    if row:
        transactions.append(row)

all_items = [item for sub in transactions for item in sub]
unique_items = sorted(list(set(all_items)))
print(f"Кількість транзакцій: {len(transactions)}")
print(f"Загальна кількість куплених товарів: {len(all_items)}")
print(f"Кількість унікальних найменувань товарів: {len(unique_items)}")

# 3. Видобування правил за допомогою Apriori
results = list(apriori.apriori(
    transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3.0,
    max_length=3
))
print(f"Знайдено записів правил (RelationRecord): {len(results)}")

# 4. Формування структурованого DataFrame
rules_list = []
for record in results:
    for stat in record.ordered_statistics:
        base = sorted(list(stat.items_base))
        add = sorted(list(stat.items_add))
        if not base or not add:
            continue
        base_str = ", ".join(base)
        add_str = ", ".join(add)
        rules_list.append({
            'rule': f"{base_str} -> {add_str}",
            'base': base_str,
            'add': add_str,
            'support': float(record.support),
            'confidence': float(stat.confidence),
            'lift': float(stat.lift)
        })

rules_df = pd.DataFrame(rules_list)
rules_df = rules_df.sort_values(by=['lift', 'confidence'], ascending=[False, False]).reset_index(drop=True)
print(f"Згенеровано спрямованих правил: {len(rules_df)}")
print("\nТоп-10 асоціативних правил за lift:")
print(rules_df[['rule', 'support', 'confidence', 'lift']].head(10).to_string())

# Збереження топ-20 правил у report.json
top20_rules = rules_df.head(20).to_dict(orient="records")
with open("report.json", "w", encoding="utf-8") as f:
    json.dump(top20_rules, f, ensure_ascii=False, indent=2)
print("\nЗбережено топ-20 правил у report.json")

# 5. Візуалізація 1: Спрямований мережевий граф асоціативних правил (Top-20)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(14, 10), dpi=200)

G = nx.DiGraph()
top_for_graph = rules_df.head(20)

for _, row in top_for_graph.iterrows():
    G.add_edge(row['base'], row['add'], weight=row['lift'], support=row['support'])

pos = nx.spring_layout(G, k=1.2, seed=42)

# Розміри та кольори вузлів
node_degrees = dict(G.degree())
node_sizes = [1800 + node_degrees[n] * 400 for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#6baed6', edgecolors='#2171b5', linewidths=1.8, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8.5, font_family='sans-serif', font_weight='bold', font_color='#111111', ax=ax)

edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
max_w = max(edge_weights)
min_w = min(edge_weights)
widths = [1.2 + 2.8 * ((w - min_w) / (max_w - min_w + 1e-6)) for w in edge_weights]

nx.draw_networkx_edges(
    G, pos,
    edge_color='#e6550d',
    width=widths,
    arrows=True,
    arrowsize=18,
    arrowstyle='-|>',
    connectionstyle='arc3,rad=0.08',
    node_size=node_sizes,
    ax=ax
)

edge_labels = {(u, v): f"L={d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='#a63603', ax=ax)

ax.set_title("Спрямований граф асоціативних правил (Top-20 за показником Lift)", fontsize=14, fontweight='bold', pad=15)
ax.axis('off')
plt.tight_layout()
fig.savefig("figures/apriori_rules_graph.png", bbox_inches='tight', dpi=200)
plt.close(fig)
print("Збережено figures/apriori_rules_graph.png")

# 6. Візуалізація 2: Кругова діаграма часток підтримки товарів / топ-правил
# Побудуємо для топ-7 правил + категорія 'Інші' (згідно Task.md)
top_pie_rules = rules_df.head(7).copy()
other_support = rules_df.iloc[7:]['support'].sum()
pie_labels = list(top_pie_rules['rule']) + ['Інші виявлені правила']
pie_values = list(top_pie_rules['support']) + [other_support]

fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
colors = ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#fc9272', '#fb6a4a', '#de2d26', '#d9d9d9']
wedges, texts, autotexts = ax.pie(
    pie_values,
    labels=None,
    autopct='%1.1f%%',
    pctdistance=0.8,
    startangle=140,
    colors=colors,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_weight('bold')

ax.legend(
    wedges, pie_labels,
    title="Асоціативні правила (Антецедент -> Консеквент)",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=8.5,
    title_fontsize=9.5
)
ax.set_title("Розподіл сумарної підтримки (Support) серед виявлених правил", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig("figures/apriori_pie_chart.png", bbox_inches='tight', dpi=200)
plt.close(fig)
print("Збережено figures/apriori_pie_chart.png")

# 7. Візуалізація 3: Стовпчикова діаграма підтримки та ліфту для топ-правил
top10_bar = rules_df.head(10).iloc[::-1] # реверс для красивого горизонтального графіку

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=200, sharey=True)

y_indices = np.arange(len(top10_bar))
rule_labels = top10_bar['rule'].tolist()

ax1.barh(y_indices, top10_bar['support'] * 100, color='#3182bd', height=0.65, edgecolor='#08519c')
ax1.set_yticks(y_indices)
ax1.set_yticklabels(rule_labels, fontsize=8.5)
ax1.set_xlabel("Підтримка, Support (%)", fontsize=10, fontweight='bold')
ax1.set_title("Рівень підтримки (Support)", fontsize=11, fontweight='bold')
for i, v in enumerate(top10_bar['support'] * 100):
    ax1.text(v + 0.01, i, f"{v:.2f}%", va='center', fontsize=8, color='#08519c', fontweight='bold')

ax2.barh(y_indices, top10_bar['lift'], color='#e6550d', height=0.65, edgecolor='#a63603')
ax2.set_xlabel("Підйом, Lift", fontsize=10, fontweight='bold')
ax2.set_title("Сила асоціативного зв'язку (Lift)", fontsize=11, fontweight='bold')
for i, v in enumerate(top10_bar['lift']):
    ax2.text(v + 0.05, i, f"{v:.2f}", va='center', fontsize=8, color='#a63603', fontweight='bold')

fig.suptitle("Порівняння рівнів Support та Lift для топ-10 асоціативних правил", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig("figures/apriori_support_bar.png", bbox_inches='tight', dpi=200)
plt.close(fig)
print("Збережено figures/apriori_support_bar.png")
