# ML Project File Structure — Detailed Breakdown

---

## Project Data Flow

```
Files (1, 2, 3)          Files (4, 5)             Files (6, 9)              Files (7, 8, 10)           Files (11, 12)
Data Ingestion      →    Target Variable     →    Feature Preparation  →    Training &          →    Evaluation &
& Understanding          Analysis                                            Experiments               Delivery
```

---

## File-by-File Breakdown

---

### 1. `data_ingestion.py` — Data Loading & Merging

| | |
|---|---|
| **Inputs** | Raw CSV files for Milestone 1 & Milestone 2 |

**Operations:**
- Read and merge files (e.g., merging the 7 E-Learning project files)
- Programmatically detect the problem type from the target variable:
  - Continuous values → **Regression**
  - Categories (Low, Medium, High) → **Classification**
  - Or both
- Calculate null counts and duplicates per column

**Outputs:** Merged DataFrame, problem type array, nulls/duplicates dictionary

---

### 2. `data_cleaning.py` — Data Cleaning

| | |
|---|---|
| **Inputs** | Merged DataFrame from File 1 |

**Operations:**
- Remove duplicate rows
- Handle missing values via imputation (`SimpleImputer` — mean, median, or most frequent)
  - > ⚠️ **Note:** Deleting rows due to missing values is **NOT allowed**
- Handle misleading values and outliers

**Outputs:** Clean DataFrame ready for analysis

---

### 3. `eda.py` — Exploratory Data Analysis

| | |
|---|---|
| **Inputs** | Clean DataFrame |

**Operations:**
- Plot histograms to understand distributions
- Generate a correlation matrix to reveal feature relationships
- Analyze relationships between independent variables

**Outputs:** Plots and insight reports

---

### 4. `target_analysis_reg.py` — Regression Target Analysis

| | |
|---|---|
| **Inputs** | Clean DataFrame for Milestone 1 |

**Operations:**
- Inspect the distribution of the continuous target variable (e.g., `popularity` or `score`)
- If distribution is skewed → apply **log transformation** to approximate a normal distribution *(as recommended in the textbook)*

**Outputs:** Regression-ready target vector `y_reg`

---

### 5. `target_analysis_clf.py` — Classification Target Analysis

| | |
|---|---|
| **Inputs** | Clean DataFrame for Milestone 2 |

**Operations:**
- Check for **class imbalance** in the target variable (e.g., `GamePopularity`: Low, Medium, High)
- Convert text categories to numeric labels using **LabelEncoder**

**Outputs:** Classification-ready numeric target vector `y_clf`

---

### 6. `preprocessing.py` — Feature Preprocessing

| | |
|---|---|
| **Inputs** | Feature matrix `X` |

**Operations:**
- Apply **StandardScaler / MinMaxScaler** to numeric columns
- Apply **One-Hot Encoding** to categorical columns
- Perform **Feature Engineering** to create new derived features
- Build a `ColumnTransformer` and `Pipeline` combining all steps

> 🔑 **Critical:** The Pipeline must encapsulate all preprocessing steps for clean deployment.

**Outputs:** Fitted Pipeline object, preprocessed feature matrix `X_preprocessed`

---

### 7. `regression_models.py` — Regression Models

| | |
|---|---|
| **Inputs** | `X_preprocessed` and `y_reg` |

**Operations:**
- Split data into train/test sets
- Train **at least 2 different models** *(Milestone 1 requirement)*
- Apply **hyperparameter tuning**

**Outputs:** Trained regression models

---

### 8. `classification_models.py` — Classification Models

| | |
|---|---|
| **Inputs** | `X_preprocessed` and `y_clf` |

**Operations:**
- Split data **80% train / 20% test** *(Milestone 2 requirement)*
- Train **at least 3 different models**
- Tune **2 hyperparameters** per model, testing **3 different values** each while keeping others fixed

**Outputs:** Trained classification models

---

### 9. `feature_selection.py` — Feature Selection

| | |
|---|---|
| **Inputs** | `X_preprocessed` |

**Operations:**
- Identify the most important features using `SelectFromModel` or correlation analysis
- Drop low-importance features *(applied **after** preprocessing, as required)*

**Outputs:** List of top-selected features for training

---

### 10. `experiment_tracker.py` — Experiment Comparison

| | |
|---|---|
| **Inputs** | Models + data in two versions: **with** Feature Engineering and **without** |

**Operations:**
- Train models under both conditions
- Compare results to determine whether feature engineering and selection improved performance

**Outputs:** Performance comparison report

---

### 11. `evaluation.py` — Results & Performance Analysis

| | |
|---|---|
| **Inputs** | Final models, test data (`X_test`, `y_test`) |

**Operations:**
- **Regression:** Compute MSE and R²
- **Classification:** Compute Accuracy
- **Milestone 2 requirement:** Generate **3 bar charts** comparing models on:
  1. Accuracy
  2. Total Training Time
  3. Total Testing Time

**Outputs:** Final performance reports and visualizations

---

### 12. `model_saving_and_inference.py` — Model Saving & Inference Script

| | |
|---|---|
| **Inputs** | Winning trained models, preprocessing Pipeline object |

**Operations:**
- Save models and preprocessing steps using `pickle` or `joblib`
- Create a standalone **test script** for the practical exam that:
  - Loads the saved model
  - Accepts a new CSV of **unseen data**
  - Applies preprocessing
  - Outputs predictions **without re-fitting** (no `.fit()` allowed)

**Outputs:** `.pkl` files + `test_script.py`

---

## Summary Table

| # | File | Purpose | Key Output |
|---|------|---------|------------|
| 1 | `data_ingestion.py` | Load & merge raw data | Merged DataFrame |
| 2 | `data_cleaning.py` | Remove noise & impute | Clean DataFrame |
| 3 | `eda.py` | Visualize & explore | Plots & insights |
| 4 | `target_analysis_reg.py` | Prepare regression target | `y_reg` |
| 5 | `target_analysis_clf.py` | Prepare classification target | `y_clf` |
| 6 | `preprocessing.py` | Scale, encode, engineer | Pipeline + `X_preprocessed` |
| 7 | `regression_models.py` | Train regression models | Trained models |
| 8 | `classification_models.py` | Train classification models | Trained models |
| 9 | `feature_selection.py` | Select best features | Feature list |
| 10 | `experiment_tracker.py` | Compare experiments | Comparison report |
| 11 | `evaluation.py` | Evaluate & visualize | Metrics + bar charts |
| 12 | `model_saving_and_inference.py` | Save & deploy | `.pkl` + `test_script.py` |