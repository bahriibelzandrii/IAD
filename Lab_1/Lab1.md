# Task: Exploratory Data Analysis (EDA) and Visualization in Python (Labs 1-2)

## 1. Source Files & Context

Use the following attached files and datasets as the main reference:

- [ІАД_лаб_роб_1.pdf](ІАД_лаб_роб_1.pdf) & [іад_лаб_1-2_oldddd.pdf](іад_лаб_1-2_oldddd.pdf)  
  Contain the laboratory guidelines for initial data analysis, statistical characteristics, and data visualization. (Note: The original PDFs contain examples in R, but **you must implement everything strictly in Python**).
- [flats.csv](flats.csv)  
  The mandatory test dataset containing real estate data (City, Rooms, Area, Price).
- **Custom Kaggle Dataset**  
  A secondary dataset chosen from the provided list of 15 recommended Kaggle datasets.

---

## 2. Main Goal

Perform comprehensive **Exploratory Data Analysis (EDA)** and **Data Visualization** using Python. 

The project must consist of two main parts:
1. **Part 1:** Analyze the provided test dataset (`flats.csv`), answer specific control questions, and build basic visualizations.
2. **Part 2:** Download, clean, and analyze a **custom dataset from Kaggle**, demonstrating advanced visualization skills, including a **mandatory Boxplot**.

---

## 3. Technology Stack

You must use **Python** exclusively. 

Required libraries:
- `pandas` (for data manipulation and analysis)
- `numpy` (for numerical operations)
- `matplotlib.pyplot` (for basic plotting)
- `seaborn` (for advanced statistical plotting, especially **boxplots**)

*Recommended environment:* Jupyter Notebook (`.ipynb`) or Google Colab, as it allows mixing code, visualizations, and markdown explanations.

---

## 4. Part 1: Test Dataset Analysis (`flats.csv`)

Load the `flats.csv` file and perform the following steps. 

### 4.1. Data Cleaning (Crucial Step)
The `flats.csv` file contains dirty data that must be cleaned before analysis:
- The `Загальна_площа` (Area) column contains strings with commas as decimal separators (e.g., `"42,99"`) and scientific notation (e.g., `"8e+05"`). You must replace commas with dots and convert the column to `float`.
- The `Ціна` (Price) column may also contain scientific notation. Convert it to `float` or `int`.
- Drop rows with critical missing values if any.

### 4.2. Basic Statistics & Control Questions
Write Python code to answer the following questions from the laboratory guidelines. Print the answers clearly in the notebook output:

1. What are the dimensions (number of rows and columns) of the dataframe?
2. Display the first 6 rows, the first 15 rows, and the last 6 rows.
3. Display the column names.
4. How many variables (columns) are in the dataset?
5. How many unique cities are in the dataset?
6. **Analytical question:** Are all of them actually cities? *(Hint: Check for "Києво-Святошинський", which is a district, not a city. Filter it out or note it in your analysis).*
7. How many 3-room apartments are sold in the city of Odesa (Одеса)?
8. What is the **median** area of a 1-room apartment in the city of Lviv (Львів)?

### 4.3. Basic Visualization (`flats.csv`)
Create the following plots for the `flats.csv` dataset:
1. **Bar Chart:** Number of apartments per city (or per number of rooms).
2. **Histogram:** Distribution of apartment prices.
3. **Scatter Plot:** Relationship between Area (X-axis) and Price (Y-axis).

---

## 5. Part 2: Custom Kaggle Dataset Analysis

Choose **ONE** dataset from the recommended list below (or propose a similar one).

### 5.1. Recommended Kaggle Datasets
1. Global Coffee Health Dataset
2. Lifestyle and Sleep Patterns
3. Shop Shoes
4. Restaurant Sales Data
5. Homelessness and Shelter Data
6. World Happiness Report
7. Weather Data
8. Salary Dataset (Simple Linear Regression)
9. Volcanic Eruptions
10. Software Engineer Salaries TR 2025
11. IMDB Dataset of Top 1000 Movies and TV Shows
12. Urban Air Quality and Climate Dataset
13. Penguin Species Dataset
14. Flood Disaster Data
15. Top Rated Movies

### 5.2. EDA Requirements for Custom Dataset
1. **Load & Inspect:** Use `df.head()`, `df.info()`, `df.describe()`.
2. **Data Cleaning:** Handle missing values (`df.isnull().sum()`), drop duplicates, and convert data types if necessary.
3. **Statistical Summary:** Calculate mean, median, standard deviation, and correlations for key numeric columns.

### 5.3. Mandatory Visualization Requirements
You must generate at least **four different types of charts** for the custom dataset. 

**⚠️ CRITICAL REQUIREMENT:** One of the charts **MUST be a Boxplot** (using `seaborn.boxplot` or `matplotlib.pyplot.boxplot`). 

The 4 required chart types are:
1. **Scatter Plot:** To show the correlation between two continuous variables (e.g., Salary vs. Years of Experience, or Budget vs. IMDB Rating).
2. **Pie Chart:** To show the proportion of categorical data (e.g., Top 5 movie genres, or distribution of coffee types). *Note: Limit to top 5-7 categories and group the rest as "Other" to keep it readable.*
3. **Histogram / KDE:** To show the distribution of a single numeric variable (e.g., distribution of salaries, or earthquake magnitudes).
4. **BOXPLOT (Mandatory):** To show data distribution, quartiles, and **outliers** (e.g., Price distribution by category, or Sleep hours by lifestyle group). 

---

## 6. Code Structure & Best Practices

- Use clear variable names (e.g., `df_flats`, `df_kaggle`).
- Add comments explaining *why* a specific transformation or plot is used.
- Ensure all plots have:
  - A clear **Title** (`plt.title()`)
  - Labeled **X and Y axes** (`plt.xlabel()`, `plt.ylabel()`)
  - A **Legend** (if applicable)
  - Proper formatting (e.g., rotating X-axis labels if they overlap: `plt.xticks(rotation=45)`).
- Use `seaborn` for the Boxplot and Scatter plots to make them visually appealing and statistically informative.

---

## 7. Report / Notebook Requirements

The final deliverable should be a Jupyter Notebook (`.ipynb`) or a well-structured Python script with saved `.png` images. If a formal report (PDF/Word) is required by the university, it must include:

1. **Title & Objective:** State the purpose of the work (EDA and Visualization in Python).
2. **Part 1 Results:** 
   - Code snippets for data cleaning.
   - Text answers to the 8 control questions regarding `flats.csv`.
   - Screenshots of the 3 plots (Bar, Histogram, Scatter).
3. **Part 2 Results:**
   - Name and link to the chosen Kaggle dataset.
   - Brief description of the dataset (what it represents).
   - Data cleaning steps taken.
   - **Screenshots of all 4 mandatory plots (Scatter, Pie, Histogram, Boxplot).**
   - A brief analytical conclusion for each plot (e.g., "The boxplot reveals significant outliers in the software engineer salaries in Istanbul...").
4. **General Conclusion:** Summary of what was learned about the datasets and the Python visualization tools.

---

## 8. Definition of Done

The task is complete when:
- [ ] `flats.csv` is loaded, cleaned (commas to dots, string to float), and analyzed.
- [ ] All 8 specific control questions for `flats.csv` are answered using Python code.
- [ ] 3 basic plots are generated for `flats.csv`.
- [ ] A custom dataset is downloaded from Kaggle and loaded into Python.
- [ ] Missing values and data types in the custom dataset are handled.
- [ ] A **Scatter plot** is created for the custom dataset.
- [ ] A **Pie chart** is created for the custom dataset.
- [ ] A **Histogram** is created for the custom dataset.
- [ ] A **Boxplot** is created for the custom dataset (showing outliers/quartiles).
- [ ] All plots have titles, axis labels, and are visually readable.
- [ ] The final Notebook/Report contains code, visual outputs, and analytical conclusions.

---

## 9. Important Notes for the Agent

- **Do not use R.** The original PDFs contain R code (`ggplot2`, `dplyr`), but the student's requirement is strictly **Python** (`pandas`, `matplotlib`, `seaborn`).
- **Boxplot is non-negotiable.** It is specifically requested by the instructor to demonstrate the ability to visualize statistical dispersion and outliers.
- **Data Cleaning is part of the grade.** The `flats.csv` file is intentionally messy (e.g., `"42,99"` instead of `42.99`). The agent must write a robust cleaning function (e.g., `df['col'] = df['col'].str.replace(',', '.').astype(float)`) before attempting to calculate medians or plot.