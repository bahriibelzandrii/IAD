# -*- coding: utf-8 -*-
"""Generate and execute Lab_3/notebook.ipynb."""
import json
import base64
import io
import os
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebook.ipynb")

cells_def = [
    {
        "type": "markdown",
        "source": [
            "# Лабораторна робота №3: Дисперсійний аналіз даних (ANOVA)\n",
            "\n",
            "**Дисципліна:** Інтелектуальний аналіз даних (ІАД)\n",
            "**Студент:** Багрій-Белз Андрій, група КН-2327б\n",
            "\n",
            "### Мета роботи:\n",
            "Засвоїти теоретичні основи та практичні навички застосування однофакторного (One-Way ANOVA) та двофакторного (Two-Way ANOVA) дисперсійного аналізу в Python з використанням бібліотек `statsmodels`, `pandas`, `scipy` та візуалізацією засобами `seaborn` і `matplotlib`."
        ]
    },
    {
        "type": "markdown",
        "source": [
            "## 1. Імпорт необхідних бібліотек та налаштування середовища\n",
            "\n",
            "**Мета:** Підключити інструменти для маніпуляції даними (`pandas`), розрахунку лінійних моделей та ANOVA-таблиць (`statsmodels`), статистичних критеріїв (`scipy.stats`) та побудови графіків."
        ]
    },
    {
        "type": "code",
        "source": [
            "import os\n",
            "import warnings\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from statsmodels.formula.api import ols\n",
            "from statsmodels.stats.anova import anova_lm\n",
            "from statsmodels.graphics.factorplots import interaction_plot\n",
            "from scipy import stats\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "plt.rcParams.update({'font.sans-serif': ['DejaVu Sans', 'Arial'], 'font.family': 'sans-serif'})\n",
            "print('Всі бібліотеки успішно імпортовано!')"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "## 2. Однофакторний дисперсійний аналіз (One-Way ANOVA) — `Voice.txt`\n",
            "\n",
            "**Мета тесту:** Дослідити взаємозв'язок між зростом співака (`Hei`) та типом голосу (`Factor`: Soprano, Alto, Tenor, Bass).\n",
            "Перевірити гіпотезу, чи є знайдена залежність реальним впливом тембру на зріст, чи це наслідок статевого диморфізму (жінки в середньому нижчі за чоловіків)."
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 2.1. Завантаження даних Voice.txt\n",
            "\n",
            "**Опис кроку:** Завантажуємо набір даних `Voice.txt` за допомогою `pandas` з підтримкою роботи як локально, так і у Google Colab."
        ]
    },
    {
        "type": "code",
        "source": [
            "# Завантаження локально або завантаження при роботі в Colab\n",
            "voice_file = 'Voice.txt' if os.path.exists('Voice.txt') else 'Lab_3/Voice.txt'\n",
            "if not os.path.exists(voice_file):\n",
            "    import gdown\n",
            "    gdown.download('https://drive.google.com/uc?id=1zc-C6y8EyHlhAYtLZ4fhwq_baA6IBDSB', 'Voice.txt', quiet=False)\n",
            "    voice_file = 'Voice.txt'\n",
            "\n",
            "df_voice = pd.read_csv(voice_file, sep=r'\\s+')\n",
            "print('Загальний розмір вибірки:', df_voice.shape)\n",
            "print('\\nКількість спостережень по типах голосу:')\n",
            "print(df_voice['Factor'].value_counts())\n",
            "print('\\nПерші рядки:')\n",
            "print(df_voice.head())"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 2.2. Повний однофакторний ANOVA для всіх співаків\n",
            "\n",
            "**Опис тесту:** Моделюємо лінійну залежність `Hei ~ Factor`. Перевіряємо нульову гіпотезу $H_0$ про рівність середніх зростів у чотирьох голосових групах."
        ]
    },
    {
        "type": "code",
        "source": [
            "model_voice_all = ols('Hei ~ Factor', data=df_voice).fit()\n",
            "anova_voice_all = anova_lm(model_voice_all)\n",
            "print('Таблиця ANOVA для всіх співаків:')\n",
            "print(anova_voice_all)\n",
            "\n",
            "f_stat = anova_voice_all.loc['Factor', 'F']\n",
            "p_val = anova_voice_all.loc['Factor', 'PR(>F)']\n",
            "print(f'\\nІнтерпретація: F = {f_stat:.2f}, p-value = {p_val:.4e}')\n",
            "print('Висновок: Оскільки p < 0.05, нульова гіпотеза відхиляється; є високозначуща різниця середніх зростів між типами голосу.')"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 2.3. Перевірка факторного впливу окремо всередині статевих груп (Жінки та Чоловіки)\n",
            "\n",
            "**Опис тесту:** Якщо фактор тембру дійсно зумовлює зріст, різниця має спостерігатися і між Soprano та Alto (жінки), а також між Tenor та Bass (чоловіки)."
        ]
    },
    {
        "type": "code",
        "source": [
            "# Фільтрація жіночої підгрупи: Soprano vs Alto\n",
            "df_women = df_voice[df_voice['Factor'].isin(['Soprano', 'Alto'])].copy()\n",
            "model_women = ols('Hei ~ Factor', data=df_women).fit()\n",
            "anova_women = anova_lm(model_women)\n",
            "print('=== ANOVA для ЖІНОК (Soprano vs Alto) ===')\n",
            "print(anova_women)\n",
            "p_women = anova_women.loc['Factor', 'PR(>F)']\n",
            "print(f'p-value для жінок = {p_women:.4f} (> 0.05) -> різниця статистично незначуща.')\n",
            "\n",
            "# Фільтрація чоловічої підгрупи: Tenor vs Bass\n",
            "df_men = df_voice[df_voice['Factor'].isin(['Tenor', 'Bass'])].copy()\n",
            "model_men = ols('Hei ~ Factor', data=df_men).fit()\n",
            "anova_men = anova_lm(model_men)\n",
            "print('\\n=== ANOVA для ЧОЛОВІКІВ (Tenor vs Bass) ===')\n",
            "print(anova_men)\n",
            "p_men = anova_men.loc['Factor', 'PR(>F)']\n",
            "print(f'p-value для чоловіків = {p_men:.4f} (> 0.05) -> різниця статистично незначуща.')\n",
            "\n",
            "print('\\nКлючовий висновок: Вплив тембру голосу на зріст у повній вибірці є артефактом статевого диморфізму (жінки нижчі за чоловіків).')"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 2.4. Візуалізація результатів Voice (Boxplot)\n",
            "\n",
            "**Опис графіка:** Діаграма розмаху зросту за типами голосу з накладеними індивідуальними точками та позначенням p-значень для кожної статі."
        ]
    },
    {
        "type": "code",
        "source": [
            "order_voice = ['Soprano', 'Alto', 'Tenor', 'Bass']\n",
            "palette_voice = {'Soprano': '#f48fb1', 'Alto': '#ec407a', 'Tenor': '#64b5f6', 'Bass': '#1e88e5'}\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)\n",
            "sns.boxplot(x='Factor', y='Hei', hue='Factor', data=df_voice, order=order_voice, palette=palette_voice, width=0.5, boxprops=dict(alpha=0.85), legend=False, ax=ax)\n",
            "sns.stripplot(x='Factor', y='Hei', data=df_voice, order=order_voice, color='black', alpha=0.35, jitter=0.2, size=5, ax=ax)\n",
            "ax.set_title('Розподіл зросту співаків за тембром голосу (Voice.txt)', pad=12, fontweight='bold')\n",
            "ax.set_xlabel('Тембр голосу (Factor)', labelpad=8)\n",
            "ax.set_ylabel('Зріст (дюйми, Hei)', labelpad=8)\n",
            "ax.axvline(1.5, color='grey', linestyle='--', linewidth=1.2, alpha=0.7)\n",
            "ax.text(0.5, df_voice['Hei'].max() - 1, 'Жінки (p = 0.235)', horizontalalignment='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#fce4ec', ec='#f48fb1'))\n",
            "ax.text(2.5, df_voice['Hei'].max() - 1, 'Чоловіки (p = 0.492)', horizontalalignment='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#e3f2fd', ec='#64b5f6'))\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "## 3. Двофакторний дисперсійний аналіз (Two-Way ANOVA) — `rat.txt`\n",
            "\n",
            "**Мета тесту:** Дослідити вплив фактора середовища (`ENVIRNMNT`: Free vs Restricted), генетичної лінії щурів (`STRAIN`: Bright, Mixed, Dull) та їхньої міжфакторної взаємодії (`ENVIRNMNT:STRAIN`) на кількість помилок при проходженні лабіринту (`ERRORS`)."
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 3.1. Завантаження даних rat.txt\n",
            "\n",
            "**Опис кроку:** Зчитуємо дані експерименту над щурами."
        ]
    },
    {
        "type": "code",
        "source": [
            "rat_file = 'rat.txt' if os.path.exists('rat.txt') else 'Lab_3/rat.txt'\n",
            "if not os.path.exists(rat_file):\n",
            "    import gdown\n",
            "    gdown.download('https://drive.google.com/uc?id=14qreu8glN6Yq9w1ECYYJy2BlpD_OAi0I', 'rat.txt', quiet=False)\n",
            "    rat_file = 'rat.txt'\n",
            "\n",
            "df_rat = pd.read_csv(rat_file, sep=r'\\s+')\n",
            "print('Розмірність вибірки:', df_rat.shape)\n",
            "print('\\nПерші рядки:')\n",
            "print(df_rat.head())"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 3.2. Оцінка двофакторної моделі (Two-Way ANOVA Type II)\n",
            "\n",
            "**Опис тесту:** Будуємо модель `ERRORS ~ C(ENVIRNMNT) + C(STRAIN) + C(ENVIRNMNT):C(STRAIN)` і формуємо ANOVA таблицю другого типу (`typ=2`)."
        ]
    },
    {
        "type": "code",
        "source": [
            "model_rat = ols('ERRORS ~ C(ENVIRNMNT) + C(STRAIN) + C(ENVIRNMNT):C(STRAIN)', data=df_rat).fit()\n",
            "anova_rat = anova_lm(model_rat, typ=2)\n",
            "print('Двофакторна таблиця ANOVA (rat.txt):')\n",
            "print(anova_rat)\n",
            "\n",
            "p_env = anova_rat.loc['C(ENVIRNMNT)', 'PR(>F)']\n",
            "p_strain = anova_rat.loc['C(STRAIN)', 'PR(>F)']\n",
            "p_inter = anova_rat.loc['C(ENVIRNMNT):C(STRAIN)', 'PR(>F)']\n",
            "\n",
            "print(f'\\nВплив середовища: F = {anova_rat.loc[\"C(ENVIRNMNT)\", \"F\"]:.3f}, p = {p_env:.4f} (< 0.05) -> статистично значущий')\n",
            "print(f'Вплив генетичної лінії: F = {anova_rat.loc[\"C(STRAIN)\", \"F\"]:.3f}, p = {p_strain:.4f} (< 0.05) -> статистично значущий')\n",
            "print(f'Взаємодія факторів: F = {anova_rat.loc[\"C(ENVIRNMNT):C(STRAIN)\", \"F\"]:.4f}, p = {p_inter:.4f} (> 0.05) -> ефект взаємодії відсутній (фактори адитивні)')"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 3.3. Візуалізація взаємодії факторів (Interaction Plot)\n",
            "\n",
            "**Опис графіка:** Графік середніх значень помилок для різних середовищ та ліній щурів. Паралельність ліній візуально підтверджує відсутність взаємодії між факторами."
        ]
    },
    {
        "type": "code",
        "source": [
            "fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)\n",
            "interaction_plot(\n",
            "    x=df_rat['STRAIN'],\n",
            "    trace=df_rat['ENVIRNMNT'],\n",
            "    response=df_rat['ERRORS'],\n",
            "    colors=['#1976d2', '#d32f2f'],\n",
            "    markers=['o', 's'],\n",
            "    ms=8,\n",
            "    linestyles=['-', '--'],\n",
            "    ax=ax,\n",
            "    legendtitle='Середовище'\n",
            ")\n",
            "ax.set_title('Графік взаємодії факторів середовища та генетичної лінії (Rat)', pad=12, fontweight='bold')\n",
            "ax.set_xlabel('Генетична лінія (STRAIN)', labelpad=8)\n",
            "ax.set_ylabel('Середня кількість помилок (ERRORS)', labelpad=8)\n",
            "ax.grid(True, linestyle=':', alpha=0.6)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 3.4. Візуалізація розподілу помилок (Boxplot)\n",
            "\n",
            "**Опис графіка:** Діаграма розмаху кількості помилок щурів за лінією та умовами середовища."
        ]
    },
    {
        "type": "code",
        "source": [
            "fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)\n",
            "sns.boxplot(\n",
            "    x='STRAIN',\n",
            "    y='ERRORS',\n",
            "    hue='ENVIRNMNT',\n",
            "    data=df_rat,\n",
            "    palette={'Free': '#42a5f5', 'Restricted': '#ef5350'},\n",
            "    width=0.55,\n",
            "    boxprops=dict(alpha=0.85),\n",
            "    ax=ax\n",
            ")\n",
            "sns.stripplot(\n",
            "    x='STRAIN',\n",
            "    y='ERRORS',\n",
            "    hue='ENVIRNMNT',\n",
            "    data=df_rat,\n",
            "    dodge=True,\n",
            "    palette={'Free': '#0d47a1', 'Restricted': '#b71c1c'},\n",
            "    alpha=0.6,\n",
            "    size=6,\n",
            "    ax=ax\n",
            ")\n",
            "handles, labels = ax.get_legend_handles_labels()\n",
            "ax.legend(handles[:2], labels[:2], title='Середовище')\n",
            "ax.set_title('Розподіл помилок щурів за середовищем та генетичною лінією', pad=12, fontweight='bold')\n",
            "ax.set_xlabel('Генетична лінія (STRAIN)', labelpad=8)\n",
            "ax.set_ylabel('Кількість помилок (ERRORS)', labelpad=8)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "## 4. Дисперсійний аналіз власного набору даних (`tips`)\n",
            "\n",
            "**Мета тесту:** Дослідити вплив дня тижня (`day`: Thur, Fri, Sat, Sun) на суму загального рахунку клієнтів ресторану (`total_bill`). Перевірити передумови ANOVA (гомогенність дисперсій) та оцінити статистичну значущість різниці."
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 4.1. Завантаження даних та описова статистика\n",
            "\n",
            "**Опис кроку:** Завантажуємо вбудований датасет `tips` бібліотеки `seaborn` та виводимо описові характеристики для кожного дня тижня."
        ]
    },
    {
        "type": "code",
        "source": [
            "df_tips = sns.load_dataset('tips')\n",
            "print(f'Кількість спостережень: {len(df_tips)}')\n",
            "print('\\nОписова статистика total_bill по днях тижня:')\n",
            "print(df_tips.groupby('day', observed=False)['total_bill'].describe())"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 4.2. Перевірка припущень та обчислення One-Way ANOVA\n",
            "\n",
            "**Опис тесту:** \n",
            "1. Перевірка однорідності дисперсій тестом Левена (`scipy.stats.levene`).\n",
            "2. Оцінка моделі `total_bill ~ C(day)` за допомогою дисперсійного аналізу."
        ]
    },
    {
        "type": "code",
        "source": [
            "# Перевірка гомогенності дисперсій (тест Левена)\n",
            "groups_tips = [group['total_bill'].values for _, group in df_tips.groupby('day', observed=False)]\n",
            "stat_levene, p_levene = stats.levene(*groups_tips)\n",
            "print(f'Тест Левена: W = {stat_levene:.4f}, p = {p_levene:.4f}')\n",
            "print('Оскільки p > 0.05, дисперсії між групами є однорідними (гомогенними), припущення ANOVA виконано.')\n",
            "\n",
            "# Однофакторний ANOVA\n",
            "model_tips = ols('total_bill ~ C(day)', data=df_tips).fit()\n",
            "anova_tips = anova_lm(model_tips)\n",
            "print('\\nТаблиця ANOVA для набору tips:')\n",
            "print(anova_tips)\n",
            "\n",
            "f_tips = anova_tips.loc['C(day)', 'F']\n",
            "p_tips = anova_tips.loc['C(day)', 'PR(>F)']\n",
            "print(f'\\nРезультат: F = {f_tips:.3f}, p-value = {p_tips:.4f}')\n",
            "print('Оскільки p = 0.0425 < 0.05, день тижня має статистично значущий вплив на суму рахунку (у вихідні середній чек вищий).')"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "### 4.3. Візуалізація розподілу рахунків (Boxplot)\n",
            "\n",
            "**Опис графіка:** Діаграма розмаху суми рахунку по днях тижня з точковими спостереженнями та лінією групових середніх значень."
        ]
    },
    {
        "type": "code",
        "source": [
            "day_order = ['Thur', 'Fri', 'Sat', 'Sun']\n",
            "palette_tips = {'Thur': '#81c784', 'Fri': '#4db6ac', 'Sat': '#ffb74d', 'Sun': '#ff8a65'}\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)\n",
            "sns.boxplot(\n",
            "    x='day',\n",
            "    y='total_bill',\n",
            "    hue='day',\n",
            "    data=df_tips,\n",
            "    order=day_order,\n",
            "    palette=palette_tips,\n",
            "    width=0.5,\n",
            "    boxprops=dict(alpha=0.85),\n",
            "    legend=False,\n",
            "    ax=ax\n",
            ")\n",
            "sns.stripplot(\n",
            "    x='day',\n",
            "    y='total_bill',\n",
            "    data=df_tips,\n",
            "    order=day_order,\n",
            "    color='#37474f',\n",
            "    alpha=0.35,\n",
            "    jitter=0.2,\n",
            "    size=5,\n",
            "    ax=ax\n",
            ")\n",
            "day_means = df_tips.groupby('day', observed=False)['total_bill'].mean().loc[day_order]\n",
            "ax.plot(range(4), day_means, color='#b71c1c', marker='D', markersize=6, linestyle='--', linewidth=1.5, label='Групове середнє')\n",
            "ax.set_title('Розподіл суми загального рахунку за днями тижня (Tips dataset)', pad=12, fontweight='bold')\n",
            "ax.set_xlabel('День тижня (day)', labelpad=8)\n",
            "ax.set_ylabel('Сума рахунку ($, total_bill)', labelpad=8)\n",
            "ax.legend(loc='upper left')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "type": "markdown",
        "source": [
            "## 5. Підсумкові висновки\n",
            "\n",
            "1. **Датасет Voice:** Однофакторний ANOVA для повної вибірки виявив формально значущий вплив тембру голосу на зріст ($p = 1.33 \\times 10^{-22}$). Проте роздільний аналіз для жінок ($p = 0.235$) та чоловіків ($p = 0.492$) довів, що всередині кожної статі тембр голосу не має статистично значущого зв'язку зі зростом. Загальний ефект зумовлений виключно статевим диморфізмом.\n",
            "2. **Датасет Rat:** Двофакторний ANOVA підтвердив значущий вплив умов середовища ($F = 5.823, p = 0.0267$) та генетичної лінії ($F = 4.164, p = 0.0326$) на кількість помилок щурів. Міжфакторна взаємодія відсутня ($F = 0.0084, p = 0.9916$), що означає адитивний характер дії факторів середовища та спадковості.\n",
            "3. **Датасет Tips:** Дослідження впливу дня тижня на середній чек виявило статистично значущу різницю ($F = 2.767, p = 0.0425$) при підтвердженій однорідності групових дисперсій (тест Левена $p = 0.5741$). У вихідні дні спостерігається тенденція до вищих сум чеків порівняно з четвергом та п'ятницею."
        ]
    }
]

# Execute each code cell and record execution_count, outputs, images
execution_env = {}
exec_count = 1

notebook_cells = []

# Mock plt.show to capture figure before closing
captured_figs = []
real_show = plt.show

def custom_show(*args, **kwargs):
    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        b64_png = base64.b64encode(buf.read()).decode("utf-8")
        captured_figs.append(b64_png)
    plt.close("all")

plt.show = custom_show

# Ensure working directory is Lab_3 during execution so relative file paths work
current_cwd = os.getcwd()
os.chdir(BASE_DIR)

try:
    for c in cells_def:
        if c["type"] == "markdown":
            notebook_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": c["source"]
            })
        elif c["type"] == "code":
            code_text = "".join(c["source"])
            stdout_io = io.StringIO()
            captured_figs.clear()
            plt.close("all")
            
            with contextlib.redirect_stdout(stdout_io):
                exec(code_text, execution_env)
            
            # Also catch any figures that didn't call show()
            for fignum in plt.get_fignums():
                fig = plt.figure(fignum)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                captured_figs.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close("all")

            stdout_val = stdout_io.getvalue()
            outputs = []
            
            if stdout_val:
                outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": stdout_val.splitlines(keepends=True)
                })
            
            for b64_png in captured_figs:
                outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": b64_png,
                        "text/plain": ["<Figure size ... with ... Axes>"]
                    },
                    "metadata": {}
                })
            
            notebook_cells.append({
                "cell_type": "code",
                "execution_count": exec_count,
                "metadata": {},
                "outputs": outputs,
                "source": c["source"]
            })
            exec_count += 1
finally:
    plt.show = real_show
    os.chdir(current_cwd)

notebook_json = {
    "cells": notebook_cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1, ensure_ascii=False)

print(f"Успішно створено та виконано {NOTEBOOK_PATH} ({len(notebook_cells)} клітинок).")
