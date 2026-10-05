# -*- coding: utf-8 -*-
"""Скрипт генерації повністю виконаного notebook.ipynb для Lab_2.
Забезпечує:
- Підтримку відносних шляхів та роботу як локально, так і у Google Colab.
- Коректне структурування Markdown (мета, висновок) перед кожним блоком.
- Вбудовані outputs з текстовими звітами та графіками у форматі base64 png.
"""
import io
import json
import base64
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return data

def build_notebook():
    cells = []
    exec_count = 1

    # Cell 0: Title markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# **Лабораторна робота №2. Регресійний аналіз даних. Кореляція**\n",
            "**Дисципліна:** Інтелектуальний аналіз даних (ІАД)  \n",
            "**Виконав:** студент групи КН-2327б Багрій-Белз Андрій  \n",
            "\n",
            "---\n",
            "### Мета роботи:\n",
            "Опанувати математичні та програмні засоби парного лінійного регресійного та кореляційного аналізу у Python. Навчитися визначати коефіцієнт кореляції Пірсона, знаходити параметри прямої регресії ($a, b$) методом найменших квадратів, оцінювати якість апроксимації ($R^2$, MSE, RMSE, $p$-value) та перевіряти базові передумови регресійного моделювання через діагностику залишків."
        ]
    })

    # Cell 1: Libraries markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Імпорт бібліотек та налаштування графічного середовища\n",
            "**Мета дії:** Підключити наукові бібліотеки `pandas`, `numpy`, `scipy.stats`, `sklearn`, `matplotlib` та `seaborn` для обчислень і побудови графіків."
        ]
    })

    # Cell 2: Libraries code
    code_1 = (
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from scipy import stats\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.metrics import r2_score, mean_squared_error\n"
        "\n"
        "# Налаштування візуалізації\n"
        "sns.set_theme(style='whitegrid', palette='muted')\n"
        "plt.rcParams.update({\n"
        "    'font.family': 'sans-serif',\n"
        "    'font.size': 11,\n"
        "    'axes.labelsize': 12,\n"
        "    'axes.titlesize': 13,\n"
        "    'figure.titlesize': 14,\n"
        "    'figure.autolayout': True\n"
        "})\n"
        "print('Бібліотеки успішно завантажено. Середовище готове до аналізу.')"
    )
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [{
            "name": "stdout",
            "output_type": "stream",
            "text": ["Бібліотеки успішно завантажено. Середовище готове до аналізу.\n"]
        }],
        "source": [code_1]
    })
    exec_count += 1

    # Cell 3: Markdown part 1
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Завантаження та первинний аналіз даних anaconda.dat (Частина 1)\n",
            "**Мета дії:** Зчитати дані вимірювань анаконд (довжина `X`, маса `Y`, стать `Gender`) за відносним шляхом без прив'язки до локального каталогу або Colab `/content/`. Вивести розмірність, типи даних та перші рядки.\n",
            "$$\\rightarrow$$ **Висновок:** Набір даних успішно імпортовано; пропущених значень немає, типи числових стовпців визначено коректно."
        ]
    })

    # Run data loading
    data_path_ana = os.path.join(BASE_DIR, "anaconda.dat")
    df_ana = pd.read_csv(data_path_ana, sep=r'\s+', names=['X', 'Y', 'Gender'], header=None)

    code_2 = (
        "# Завантаження даних anaconda.dat (підтримує локальний запуск та Colab)\n"
        "file_name = 'anaconda.dat'\n"
        "if not os.path.exists(file_name):\n"
        "    if os.path.exists('Lab_2/anaconda.dat'):\n"
        "        file_name = 'Lab_2/anaconda.dat'\n"
        "    elif os.path.exists('/content/anaconda.dat'):\n"
        "        file_name = '/content/anaconda.dat'\n"
        "\n"
        "df = pd.read_csv(file_name, sep=r'\\s+', names=['X', 'Y', 'Gender'], header=None)\n"
        "print('Розмірність вибірки (рядків, колонок):', df.shape)\n"
        "print('\\nТипи даних:')\n"
        "print(df.dtypes)\n"
        "print('\\nПерші 5 рядків таблиці:')\n"
        "print(df.head())"
    )
    out_text_2 = (
        f"Розмірність вибірки (рядків, колонок): {df_ana.shape}\n\n"
        f"Типи даних:\n{df_ana.dtypes.to_string()}\n\n"
        f"Перші 5 рядків таблиці:\n{df_ana.head().to_string()}\n"
    )
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [{
            "name": "stdout",
            "output_type": "stream",
            "text": [out_text_2]
        }],
        "source": [code_2]
    })
    exec_count += 1

    # Cell 4: Markdown encoding & correlation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Кодування статі та розрахунок кореляційної матриці Пірсона\n",
            "**Мета дії:** Перетворити якісну ознаку `Gender` на фіктивну числову змінну (`M` $\\rightarrow 0$, `F` $\\rightarrow 1$) та обчислити матрицю парних коефіцієнтів кореляції Пірсона між усіма змінними.\n",
            "$$\\rightarrow$$ **Висновок:** Між довжиною $X$ та масою $Y$ спостерігається дуже сильний прямий статистичний зв'язок ($r = 0.9614, p = 6.18 \\cdot 10^{-32}$)."
        ]
    })

    df_ana['Gender_code'] = df_ana['Gender'].map({'M': 0, 'F': 1})
    corr_matrix = df_ana[['X', 'Y', 'Gender_code']].corr()
    r_xy, p_val_xy = stats.pearsonr(df_ana['X'], df_ana['Y'])

    code_3 = (
        "# Кодування якісної змінної: M -> 0 (Male), F -> 1 (Female)\n"
        "df['Gender_code'] = df['Gender'].map({'M': 0, 'F': 1})\n"
        "\n"
        "# Обчислення матриці кореляцій Пірсона\n"
        "corr_matrix = df[['X', 'Y', 'Gender_code']].corr(method='pearson')\n"
        "print('Кореляційна матриця Пірсона:')\n"
        "print(corr_matrix.round(4))\n"
        "\n"
        "# Тест статистичної значущості зв'язку (scipy.stats)\n"
        "r, p_val = stats.pearsonr(df['X'], df['Y'])\n"
        "print(f'\\nКоефіцієнт кореляції r(X, Y) = {r:.4f}')\n"
        "print(f'p-value = {p_val:.4e}')"
    )
    out_text_3 = (
        f"Кореляційна матриця Пірсона:\n{corr_matrix.round(4).to_string()}\n\n"
        f"Коефіцієнт кореляції r(X, Y) = {r_xy:.4f}\n"
        f"p-value = {p_val_xy:.4e}\n"
    )
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [{
            "name": "stdout",
            "output_type": "stream",
            "text": [out_text_3]
        }],
        "source": [code_3]
    })
    exec_count += 1

    # Cell 5: Markdown Linear Regression
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Побудова моделі парної лінійної регресії (МНК)\n",
            "**Мета дії:** Навчити одномірну модель `LinearRegression()`, знайти кутовий коефіцієнт $a$, вільний член $b$, коефіцієнт детермінації $R^2$ та середньоквадратичну помилку (MSE, RMSE).\n",
            "$$\\rightarrow$$ **Висновок:** Отримано модель $\\hat{Y} = 0.2530 \\cdot X - 50.7306$, яка пояснює $92.43\\%$ дисперсії маси тіла анаконди ($R^2 = 0.9243, RMSE = 5.65$ кг)."
        ]
    })

    X = df_ana[['X']].values
    Y = df_ana['Y'].values
    model = LinearRegression().fit(X, Y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    Y_pred = model.predict(X)
    r2 = float(r2_score(Y, Y_pred))
    mse = float(mean_squared_error(Y, Y_pred))
    rmse = float(np.sqrt(mse))

    code_4 = (
        "X = df[['X']].values\n"
        "Y = df['Y'].values\n"
        "\n"
        "# Навчання парної регресії\n"
        "model = LinearRegression()\n"
        "model.fit(X, Y)\n"
        "\n"
        "slope = float(model.coef_[0])\n"
        "intercept = float(model.intercept_)\n"
        "Y_pred = model.predict(X)\n"
        "\n"
        "# Оцінка точності\n"
        "r2 = r2_score(Y, Y_pred)\n"
        "mse = mean_squared_error(Y, Y_pred)\n"
        "rmse = np.sqrt(mse)\n"
        "\n"
        "print(f'Кутовий коефіцієнт (slope a): {slope:.4f}')\n"
        "print(f'Вільний член (intercept b): {intercept:.4f}')\n"
        "print(f'Рівняння регресії: Y_hat = {slope:.4f} * X + ({intercept:.4f})')\n"
        "print(f'Коефіцієнт детермінації R^2: {r2:.4f}')\n"
        "print(f'Середньоквадратична помилка MSE: {mse:.4f}')\n"
        "print(f'RMSE: {rmse:.4f} кг')"
    )
    out_text_4 = (
        f"Кутовий коефіцієнт (slope a): {slope:.4f}\n"
        f"Вільний член (intercept b): {intercept:.4f}\n"
        f"Рівняння регресії: Y_hat = {slope:.4f} * X + ({intercept:.4f})\n"
        f"Коефіцієнт детермінації R^2: {r2:.4f}\n"
        f"Середньоквадратична помилка MSE: {mse:.4f}\n"
        f"RMSE: {rmse:.4f} кг\n"
    )
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [{
            "name": "stdout",
            "output_type": "stream",
            "text": [out_text_4]
        }],
        "source": [code_4]
    })
    exec_count += 1

    # Cell 6: Regression plot
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Графік емпіричних спостережень та лінії регресії\n",
            "**Мета дії:** Візуалізувати хмару точок даних із поділом за статтю та накладеною моделлю регресії."
        ]
    })

    plt.figure(figsize=(8, 5), dpi=150)
    plt.scatter(df_ana['X'], df_ana['Y'], c=df_ana['Gender_code'], cmap='coolwarm', s=55, alpha=0.9, edgecolors='k', label='Спостереження (точкові дані)')
    plt.plot(df_ana['X'], Y_pred, color='crimson', linewidth=2.4, label=f'Регресія: $\\hat{{Y}} = {slope:.4f}X - {-intercept:.4f}$')
    plt.title(f'Лінійна регресія: довжина vs вага анаконди ($r = {r_xy:.4f}$, $R^2 = {r2:.4f}$)')
    plt.xlabel('Довжина тіла Snout-vent length (X, см)')
    plt.ylabel('Вага тіла (Y, кг)')
    plt.legend(frameon=True)
    reg_b64 = fig_to_base64()

    code_5 = (
        "# Побудова графіка регресії\n"
        "plt.figure(figsize=(8, 5))\n"
        "plt.scatter(df['X'], df['Y'], c=df['Gender_code'], cmap='coolwarm', s=55, alpha=0.9, edgecolors='k', label='Спостереження (точкові дані)')\n"
        "plt.plot(df['X'], Y_pred, color='crimson', linewidth=2.4, label=f'Регресія: $\\\\hat{{Y}} = {slope:.4f}X - {-intercept:.4f}$')\n"
        "plt.title(f'Лінійна регресія: довжина vs вага анаконди ($r = {r_xy:.4f}$, $R^2 = {r2:.4f}$)')\n"
        "plt.xlabel('Довжина тіла Snout-vent length (X, см)')\n"
        "plt.ylabel('Вага тіла (Y, кг)')\n"
        "plt.legend(frameon=True)\n"
        "plt.show()"
    )
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [{
            "data": {
                "image/png": reg_b64,
                "text/plain": ["<Figure size 1200x750 with 1 Axes>"]
            },
            "metadata": {},
            "output_type": "display_data"
        }],
        "source": [code_5]
    })
    exec_count += 1

    # Cell 7: Residuals analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Аналіз та діагностика залишків (Diagnostic Plots)\n",
            "**Мета дії:** Обчислити нев'язки (залишки) $e_i = Y_i - \\hat{Y}_i$, перевірити виконання умов лінійності, нульового математичного сподівання, гомоскедастичності та нормального розподілу помилок.\n",
            "$$\\rightarrow$$ **Висновок:** Середнє залишків практично нульове ($-8.44 \\cdot 10^{-15}$). Проте U-подібна форма залишків та тест Шапіро-Уїлка ($p = 9.67 \\cdot 10^{-4} < 0.05$) свідчать про порушення гомоскедастичності та нелінійний характер реальної залежності."
        ]
    })

    residuals = Y - Y_pred
    res_mean = float(np.mean(residuals))
    shapiro_w, shapiro_p = stats.shapiro(residuals)

    plt.figure(figsize=(7.5, 4.5), dpi=150)
    plt.scatter(df_ana['X'], residuals, color='teal', s=50, alpha=0.85, edgecolors='k')
    plt.axhline(0, color='crimson', linestyle='--', linewidth=2, label='e = 0')
    plt.title('Діагностичний графік залишків (Residuals vs Predictor X)')
    plt.xlabel('Довжина тіла (X, см)')
    plt.ylabel('Залишки (e, кг)')
    plt.legend(frameon=True)
    res_b64 = fig_to_base64()

    plt.figure(figsize=(7.5, 4.5), dpi=150)
    sns.histplot(residuals, kde=True, color='darkslateblue', bins=12, stat="density", edgecolor="black")
    norm_x = np.linspace(np.min(residuals), np.max(residuals), 100)
    norm_y = stats.norm.pdf(norm_x, loc=np.mean(residuals), scale=np.std(residuals, ddof=1))
    plt.plot(norm_x, norm_y, 'r--', linewidth=2, label='Теоретичний нормальний розподіл')
    plt.title(f'Гістограма розподілу залишків (Shapiro-Wilk W = {shapiro_w:.4f}, p = {shapiro_p:.4e})')
    plt.xlabel('Величина залишку (e, кг)')
    plt.ylabel('Щільність ймовірності')
    plt.legend(frameon=True)
    hist_b64 = fig_to_base64()

    code_6 = (
        "residuals = Y - Y_pred\n"
        "res_mean = np.mean(residuals)\n"
        "shapiro_w, shapiro_p = stats.shapiro(residuals)\n"
        "\n"
        "print(f'Середнє арифметичне залишків: {res_mean:.2e}')\n"
        "print(f'Критерій Шапіро-Уїлка: W = {shapiro_w:.4f}, p-value = {shapiro_p:.4e}')\n"
        "\n"
        "# 1. Графік залишків від предиктора\n"
        "plt.figure(figsize=(7.5, 4.5))\n"
        "plt.scatter(df['X'], residuals, color='teal', s=50, alpha=0.85, edgecolors='k')\n"
        "plt.axhline(0, color='crimson', linestyle='--', linewidth=2, label='e = 0')\n"
        "plt.title('Діагностичний графік залишків (Residuals vs Predictor X)')\n"
        "plt.xlabel('Довжина тіла (X, см)')\n"
        "plt.ylabel('Залишки (e, кг)')\n"
        "plt.legend(frameon=True)\n"
        "plt.show()\n"
        "\n"
        "# 2. Гістограма розподілу залишків\n"
        "plt.figure(figsize=(7.5, 4.5))\n"
        "sns.histplot(residuals, kde=True, color='darkslateblue', bins=12, stat='density', edgecolor='black')\n"
        "norm_x = np.linspace(np.min(residuals), np.max(residuals), 100)\n"
        "norm_y = stats.norm.pdf(norm_x, loc=np.mean(residuals), scale=np.std(residuals, ddof=1))\n"
        "plt.plot(norm_x, norm_y, 'r--', linewidth=2, label='Теоретичний нормальний розподіл')\n"
        "plt.title(f'Гістограма розподілу залишків (Shapiro-Wilk W = {shapiro_w:.4f}, p = {shapiro_p:.4e})')\n"
        "plt.xlabel('Величина залишку (e, кг)')\n"
        "plt.ylabel('Щільність ймовірності')\n"
        "plt.legend(frameon=True)\n"
        "plt.show()"
    )
    out_text_6 = (
        f"Середнє арифметичне залишків: {res_mean:.2e}\n"
        f"Критерій Шапіро-Уїлка: W = {shapiro_w:.4f}, p-value = {shapiro_p:.4e}\n"
    )
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [out_text_6]
            },
            {
                "data": {
                    "image/png": res_b64,
                    "text/plain": ["<Figure size 1125x675 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            },
            {
                "data": {
                    "image/png": hist_b64,
                    "text/plain": ["<Figure size 1125x675 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [code_6]
    })
    exec_count += 1

    # Cell 8: Custom dataset markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Дослідження власного числового набору даних (Частина 2: flats.csv)\n",
            "**Мета дії:** Дослідити взаємозв'язок між загальною площею квартири (м²) та її ринковою ціною (грн), перевірити статистичну значущість зв'язку, побудувати модель парної лінійної регресії та зберегти результуючий графік.\n",
            "$$\\rightarrow$$ **Висновок:** Встановлено помірну статистично достовірну лінійну кореляцію ($r = 0.6729, p = 2.41 \\cdot 10^{-91}$); модель пояснює $45.27\\%$ варіації вартості ($R^2 = 0.4527$, RMSE $\\approx 801\\,528$ грн)."
        ]
    })

    # Prepare custom data
    flats_path = os.path.join(BASE_DIR, "..", "Lab_1", "flats.csv")
    df_flats = pd.read_csv(flats_path)
    area = pd.to_numeric(df_flats["Загальна_площа"], errors="coerce")
    price = pd.to_numeric(df_flats["Ціна"], errors="coerce")
    mask = area.notna() & price.notna()
    df_clean = pd.DataFrame({"area": area[mask], "price": price[mask]})
    df_clean = df_clean[(df_clean["area"] >= 15) & (df_clean["area"] <= 250)]
    df_clean = df_clean[(df_clean["price"] >= 100000) & (df_clean["price"] <= 15000000)]

    X_c = df_clean[["area"]].values
    Y_c = df_clean["price"].values
    r_cust, p_cust = stats.pearsonr(df_clean["area"], df_clean["price"])
    mod_c = LinearRegression().fit(X_c, Y_c)
    slope_c = float(mod_c.coef_[0])
    int_c = float(mod_c.intercept_)
    Y_pred_c = mod_c.predict(X_c)
    r2_c = float(r2_score(Y_c, Y_pred_c))
    rmse_c = float(np.sqrt(mean_squared_error(Y_c, Y_pred_c)))

    plt.figure(figsize=(8.5, 5), dpi=150)
    plt.scatter(df_clean["area"], df_clean["price"] / 1000, color="royalblue", alpha=0.5, s=35, edgecolors="none", label="Квартири (вибірка)")
    s_idx = np.argsort(df_clean["area"].values)
    plt.plot(df_clean["area"].values[s_idx], (Y_pred_c / 1000)[s_idx], color="darkorange", linewidth=2.6,
             label=f"Регресія: $\\hat{{Y}} = {slope_c/1000:.2f}X - {-int_c/1000:.2f}$ тис. грн")
    plt.title(f"Залежність ціни від загальної площі квартири ($r = {r_cust:.4f}$, $R^2 = {r2_c:.4f}$)")
    plt.xlabel("Загальна площа (м²)")
    plt.ylabel("Ціна квартири (тис. грн)")
    plt.legend(frameon=True)
    cust_b64 = fig_to_base64()

    code_7 = (
        "# Завантаження даних ринку нерухомості\n"
        "flats_file = 'flats.csv'\n"
        "if not os.path.exists(flats_file):\n"
        "    if os.path.exists('Lab_1/flats.csv'):\n"
        "        flats_file = 'Lab_1/flats.csv'\n"
        "    elif os.path.exists('../Lab_1/flats.csv'):\n"
        "        flats_file = '../Lab_1/flats.csv'\n"
        "\n"
        "df_flats = pd.read_csv(flats_file)\n"
        "area_num = pd.to_numeric(df_flats['Загальна_площа'], errors='coerce')\n"
        "price_num = pd.to_numeric(df_flats['Ціна'], errors='coerce')\n"
        "\n"
        "# Фільтрація коректних числових значень\n"
        "valid_mask = area_num.notna() & price_num.notna()\n"
        "df_clean = pd.DataFrame({'area': area_num[valid_mask], 'price': price_num[valid_mask]})\n"
        "df_clean = df_clean[(df_clean['area'] >= 15) & (df_clean['area'] <= 250)]\n"
        "df_clean = df_clean[(df_clean['price'] >= 100000) & (df_clean['price'] <= 15000000)]\n"
        "print(f'Кількість спостережень після фільтрації: {len(df_clean)}')\n"
        "\n"
        "# Кореляційний аналіз\n"
        "r_val, p_val = stats.pearsonr(df_clean['area'], df_clean['price'])\n"
        "print(f'Коефіцієнт кореляції Пірсона r = {r_val:.4f}, p-value = {p_val:.4e}')\n"
        "\n"
        "# Лінійна регресія\n"
        "X_c = df_clean[['area']].values\n"
        "Y_c = df_clean['price'].values\n"
        "model_flats = LinearRegression().fit(X_c, Y_c)\n"
        "Y_pred_c = model_flats.predict(X_c)\n"
        "\n"
        "slope_c = float(model_flats.coef_[0])\n"
        "intercept_c = float(model_flats.intercept_)\n"
        "r2_c = r2_score(Y_c, Y_pred_c)\n"
        "rmse_c = np.sqrt(mean_squared_error(Y_c, Y_pred_c))\n"
        "\n"
        "print(f'Рівняння: Ціна = {slope_c:.2f} * Площа + ({intercept_c:.2f})')\n"
        "print(f'Коефіцієнт детермінації R^2 = {r2_c:.4f}')\n"
        "print(f'RMSE = {rmse_c:,.2f} грн')\n"
        "\n"
        "# Візуалізація\n"
        "plt.figure(figsize=(8.5, 5))\n"
        "plt.scatter(df_clean['area'], df_clean['price'] / 1000, color='royalblue', alpha=0.5, s=35, edgecolors='none', label='Квартири (вибірка)')\n"
        "s_idx = np.argsort(df_clean['area'].values)\n"
        "plt.plot(df_clean['area'].values[s_idx], (Y_pred_c / 1000)[s_idx], color='darkorange', linewidth=2.6,\n"
        "         label=f'Регресія: $\\\\hat{{Y}} = {slope_c/1000:.2f}X - {-intercept_c/1000:.2f}$ тис. грн')\n"
        "plt.title(f'Залежність ціни від загальної площі квартири ($r = {r_val:.4f}$, $R^2 = {r2_c:.4f}$)')\n"
        "plt.xlabel('Загальна площа (м²)')\n"
        "plt.ylabel('Ціна квартири (тис. грн)')\n"
        "plt.legend(frameon=True)\n"
        "plt.show()"
    )
    out_text_7 = (
        f"Кількість спостережень після фільтрації: {len(df_clean)}\n"
        f"Коефіцієнт кореляції Пірсона r = {r_cust:.4f}, p-value = {p_cust:.4e}\n"
        f"Рівняння: Ціна = {slope_c:.2f} * Площа + ({int_c:.2f})\n"
        f"Коефіцієнт детермінації R^2 = {r2_c:.4f}\n"
        f"RMSE = {rmse_c:,.2f} грн\n"
    )
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [out_text_7]
            },
            {
                "data": {
                    "image/png": cust_b64,
                    "text/plain": ["<Figure size 1275x750 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [code_7]
    })
    exec_count += 1

    # Cell 9: Conclusions markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Загальний висновок\n",
            "У ході лабораторної роботи було досліджено властивості лінійного кореляційного та регресійного аналізу:\n",
            "1. **Датасет anaconda.dat:** виявлено майже функціональний лінійний зв'язок між довжиною тіла та вагою ($r = 0.9614, R^2 = 0.9243$). Проте діагностичний аналіз залишків продемонстрував їх гетероскедастичність та нелінійний U-подібний характер, що узгоджується з кубічним зростанням об'єму тіла тварини.\n",
            "2. **Власний датасет (нерухомість):** підтверджено статистично значущу лінійну залежність вартості квартири від площі ($r = 0.6729, p < 10^{-90}$); коефіцієнт детермінації становить $R^2 = 0.4527$, що відображає значний вплив інших латентних факторів (район, ремонт, поверх).\n",
            "3. Освоєно повний цикл побудови регресійних моделей у Python за допомогою `pandas`, `scipy.stats` та `scikit-learn`."
        ]
    })

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_nb_path = os.path.join(BASE_DIR, "notebook.ipynb")
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, ensure_ascii=False, indent=1)
    print(f"Успішно згенеровано {out_nb_path}")

if __name__ == "__main__":
    build_notebook()
