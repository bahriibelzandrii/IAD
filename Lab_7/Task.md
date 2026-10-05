# Task: Прогнозування з використанням часових рядів у Python (Лабораторна робота 7)

## 1. Контекст та вхідні файли

У директорії `Lab_7/` розміщені такі файли:
- **`kings.txt`** — часовий ряд тривалості життя / віку смерті 42 англійських королів.
- **`babyboom.txt`** — часовий ряд народжуваності / маси тіла новонароджених у часі.
- **`kings.R`** — референсний код мовою R із використанням бібліотек `TTR` та `forecast` (функції `SMA`, `decompose`, `nnetar`, `auto.arima`).
- **`lab_7.ipynb`** — референсний Jupyter Notebook з прогнозуванням засобами сучасної Python-бібліотеки `sktime` (датасет пасажиропотоку `load_airline`, поділ вибірки `temporal_train_test_split`, базова модель `NaiveForecaster`, візуалізація `plot_series`).
- **`лаб_7_python_Винарчик.pdf`** — методичні вказівки та студентський звіт щодо налаштування та запуску прогнозування часових рядів у Python.
- **`іад_лаб _р_7_2027.pdf`** — теоретичні відомості про структуру часових рядів (тренд, сезонність, стаціонарність, автокореляція ACF/PACF, моделі ковзного середнього, ARIMA).

---

## 2. Головна мета

Реалізувати повноцінний цикл дослідження та прогнозування часових рядів у Python, що охоплює:
1. **Первинний аналіз та згладжування ковзним середнім (Smoothing):**
   - Побудувати часовий ряд для `kings.txt` та згладити його простим ковзним середнім (Simple Moving Average, SMA з вікном $n=9$);
   - Дослідити ряд `babyboom.txt`, виконати згладжування ($n=11$) та провести аддитивну/мультиплікативну декомпозицію на тренд, сезонність і випадковий шум (`seasonal_decompose`).
2. **Моделювання та прогнозування часових рядів (Forecasting):**
   - Розділити часовий ряд на тренувальну та тестову вибірки з урахуванням часового порядку (`temporal_train_test_split`);
   - Застосувати базову модель `NaiveForecaster` (стратегії `"last"` або `"seasonal_last"`);
   - Застосувати просунуту статистичну модель (наприклад, `ARIMA` / `AutoARIMA` або експоненційне згладжування `ExponentialSmoothing`);
   - Здійснити прогноз на заданий горизонт (наприклад, 36 місяців для авіаперевезень або відповідний період для немовлят/королів);
   - Оцінити точність прогнозування за допомогою метрик: MAE, RMSE, MAPE;
   - Побудувати графіки суміщення реальних та прогнозних значень (`plot_series` або `matplotlib`).

---

## 3. Технологічний стек

- **Мова:** Python 3 (локальне виконання без хмарних сервісів).
- **Бібліотеки:**
  - `pandas` — часові індекси (`DatetimeIndex` / `PeriodIndex`), ресемплінг, ковзні вікна (`.rolling()`);
  - `numpy` — математичні обчислення;
  - `sktime` (`sktime.forecasting.naive.NaiveForecaster`, `sktime.forecasting.model_selection.temporal_train_test_split`, `sktime.utils.plotting.plot_series`, `sktime.performance_metrics.forecasting`) або `statsmodels` (`statsmodels.tsa.seasonal.seasonal_decompose`, `statsmodels.tsa.arima.model.ARIMA`, `statsmodels.tsa.holtwinters.ExponentialSmoothing`);
  - `matplotlib.pyplot` — графіки рядів, компонентів декомпозиції та прогнозних довірчих інтервалів.

---

## 4. Детальний покроковий план

### 4.1. Дослідження та згладжування рядів `kings.txt` та `babyboom.txt`
1. Зчитати дані:
   ```python
   kings = np.loadtxt('kings.txt')
   babyboom = np.loadtxt('babyboom.txt')
   ```
2. Згладжування ряду `kings.txt` ковзним середнім:
   ```python
   kings_series = pd.Series(kings)
   kings_sma = kings_series.rolling(window=9, center=True).mean()
   ```
   Побудувати графік порівняння початкового ряду та лінії тренду: `figures/kings_sma.png`.
3. Декомпозиція ряду `babyboom.txt`:
   - Сформувати ряд з періодичністю (наприклад, щомісячна частота `freq=12`);
   - Застосувати `seasonal_decompose(babyboom_series, model='additive', period=12)`;
   - Побудувати та зберегти 4 складові графіка (Observed, Trend, Seasonal, Residual): `figures/babyboom_decomposition.png`.

### 4.2. Побудова прогнозної моделі (Airline Passengers або Babyboom)
1. Завантажити часовий ряд (через `sktime.datasets.load_airline` або підготувати `babyboom.txt` з часовим індексом `PeriodIndex`).
2. Розділити на тренувальну та тестову частини:
   ```python
   from sktime.forecasting.model_selection import temporal_train_test_split
   y_train, y_test = temporal_train_test_split(y, test_size=36)
   ```
3. Візуалізувати початкові дані:
   ```python
   from sktime.utils.plotting import plot_series
   plot_series(y_train, y_test, labels=["Train", "Test"])
   plt.title("Розподіл на Train та Test")
   plt.savefig("figures/split_series.png")
   ```

### 4.3. Навчання моделей та генерація прогнозу
1. **Базова наївна модель (Baseline):**
   ```python
   from sktime.forecasting.naive import NaiveForecaster
   forecaster_naive = NaiveForecaster(strategy="last")
   forecaster_naive.fit(y_train)
   fh = np.arange(1, len(y_test) + 1)
   y_pred_naive = forecaster_naive.predict(fh)
   ```
2. **Сезонна/Трендова модель (ARIMA або ExponentialSmoothing):**
   ```python
   from statsmodels.tsa.holtwinters import ExponentialSmoothing
   model_hw = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
   y_pred_hw = model_hw.forecast(len(y_test))
   ```
3. Побудувати підсумковий порівняльний графік реальних даних і прогнозів:
   - Зберегти: `figures/forecast_comparison.png`.

### 4.4. Оцінка точності
1. Розрахувати показники:
   - $MAE = \frac{1}{n} \sum |y - \hat{y}|$
   - $RMSE = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$
   - $MAPE = \frac{100\%}{n} \sum \left|\frac{y - \hat{y}}{y}\right|$
2. Порівняти якість наївного прогнозу та сезонної моделі в табличному вигляді.

---

## 5. Генерація офіційного звіту (.docx) за допомогою скіла report-maker

Після виконання розрахунків та побудови графіків необхідно згенерувати офіційний друкований звіт у форматі Word:
- **Інструмент:** зареєстрований скіл **`report-maker`** (скрипти в `~/.omp/agent/managed-skills/report-maker/` або `build_report.py`).
- **Вихідний файл:** `report/ІАД_КН-2327Б_Багрій-Белз_ЛР-7.docx`.
- **Метадані титульної сторінки:**
  - Дисципліна: `Інтелектуальний аналіз даних` (код: `ІАД`);
  - Тема: `Прогнозування з використанням часових рядів`;
  - Лабораторна робота №: `7`;
  - Студент: `Багрія-Белз Андрій` (у родовому відмінку титульного блоку: `Багрія-Белза Андрія`);
  - Група: `КН-2327Б`.
- **Вимоги до тексту та стилю (КРИТИЧНО):**
  - Орієнтуватися на оформлення `C:\Programing\University\IAD\Lab_1\report\ІАД_КН-2327Б_Багрій-Белз_ЛР-1.docx`, **але тексту має бути значно менше (без води як у ЛР-1)**.
  - Тільки **строго необхідний фактаж**:
    1. Мета роботи (1 речення);
    2. Завдання 1 (Згладжування та декомпозиція): графік ковзного середнього `kings_sma.png` та графік декомпозиції `babyboom_decomposition.png` $\rightarrow$ стислий висновок щодо виділених компонентів (тренд, сезонність);
    3. Завдання 2 (Прогнозування): графіки поділу ряду `split_series.png` та суміщеного прогнозу `forecast_comparison.png` $\rightarrow$ зведена таблиця метрик точності (MAE, RMSE, MAPE) для наївного підходу та моделі часових рядів;
    4. Висновок: 1 лаконічний абзац (2–4 речення) від першої особи минулого часу.
- **Графічні матеріали:** усі графіки центровані, чіткі, займають усю ширину тексту (14.5–16 см).

---

## 6. Структура файлів результату

- `analysis.py` — автономний Python-скрипт.
- `notebook.ipynb` — готовий блокнот для **Google Colab** та локального використання:
  - Перед кожним блоком (згладжування королів, декомпозиція народжуваності, поділ часового ряду, навчання прогнозних моделей, оцінка похибок) містяться структуровані, лаконічні **Markdown-комірки** (мета $\rightarrow$ код $\rightarrow$ результат $\rightarrow$ інтерпретація).
  - Усі комірки виконані (графіки рядів та прогнози збережені у виводах).
  - Шляхи до `kings.txt` та `babyboom.txt` відносні.
  - Додано рядок для встановлення потрібних бібліотек у Colab за потреби (`!pip install sktime statsmodels`).
- `report/ІАД_КН-2327Б_Багрій-Белз_ЛР-7.docx` — готовий офіційний звіт через `report-maker` з мінімальним змістовним текстом та графіками.
- `build_report.py` або `report.json` — скрипт/конфігурація побудови звіту.
- `figures/` — збережені графіки:
  - `kings_sma.png`
  - `babyboom_decomposition.png`
  - `split_series.png`
  - `forecast_comparison.png`
- `report.json` — збережені метрики якості прогнозу (MAE, RMSE, MAPE для кожної моделі).

---

## 7. Критерії готовності (Definition of Done)

- [ ] Успішно зчитано `kings.txt` та `babyboom.txt`.
- [ ] Побудовано згладжений графік ковзного середнього `kings_sma.png`.
- [ ] Здійснено сезонну декомпозицію часового ряду на компоненти `babyboom_decomposition.png`.
- [ ] Реалізовано хронологічний поділ вибірки `temporal_train_test_split`.
- [ ] Навчено модель прогнозування (`NaiveForecaster` та додатково трендову/сезонну модель).
- [ ] Отримано прогноз на заданий горизонт `fh` та побудовано графік `forecast_comparison.png`.
- [ ] Розраховано числові метрики точності (MAE, RMSE, MAPE).
- [ ] Сформовано виконаний `notebook.ipynb` із логічними та короткими описами у Markdown-комірках і сумісністю з Google Colab.
- [ ] Згенеровано офіційний звіт `report/ІАД_КН-2327Б_Багрій-Белз_ЛР-7.docx` за допомогою скіла `report-maker` з усіма графіками та строго необхідним текстом (без зайвої води).
- [ ] Надано аналітичні висновки щодо поведінки часового ряду, наявності тренду та сезонних коливань.
