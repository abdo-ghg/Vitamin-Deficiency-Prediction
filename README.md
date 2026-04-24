# 🧬 Vitamin Deficiency Prediction

> **A production-ready machine learning pipeline** that predicts the severity of vitamin deficiency in individuals based on demographic data, lifestyle factors, dietary nutrient intake, and clinical symptoms — packaged as a REST API for real-world integration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack & Architecture](#-tech-stack--architecture)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Results](#-model-results)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Usage / Running the App](#-usage--running-the-app)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔬 Project Overview

Vitamin deficiencies affect over **2 billion people worldwide** and are frequently underdiagnosed due to the subtlety of early symptoms and the cost of clinical lab work. This project addresses that gap by building a **regression-based machine learning system** capable of estimating a patient's vitamin deficiency severity score (continuous, 0–1 scale) from a rich, multi-domain feature set.

**What it solves:**

- Enables early screening without expensive blood tests.
- Provides a continuous deficiency score rather than a binary diagnosis — capturing the full spectrum of deficiency severity.
- Exposes predictions via a REST API, making it trivially integratable into any EHR system, mobile app, or clinical dashboard.

**Input features span four domains:**

| Domain | Examples |
|---|---|
| Demographics | Age, Gender, BMI |
| Lifestyle | Smoking, Alcohol, Exercise, Diet type, Sun exposure |
| Nutrient Intake | Vitamins A, C, D, E, B12, Folate, Calcium, Iron (all as % of RDA) |
| Clinical | Symptom count and symptom list (bone pain, fatigue, night blindness, etc.) |

---

## 🛠️ Tech Stack & Architecture

### Libraries & Frameworks

| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime |
| **Data Layer** | Pandas, NumPy | Data ingestion, transformation, and feature engineering |
| **Visualisation** | Matplotlib, Seaborn | EDA plots (histograms, scatter, box, heatmaps, count plots) |
| **ML Models** | scikit-learn, XGBoost | Regression models and preprocessing pipelines |
| **API Server** | Flask | Serves the trained model artifact as a REST endpoint |
| **Serialisation** | joblib | Persists the scikit-learn `Pipeline` object to disk |
| **Notebook** | Jupyter | Interactive end-to-end research workflow |

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                          │
│                                                              │
│  data.csv ──► Pandas ──► Feature Engineering                 │
│                               │                              │
│                    ┌──────────▼──────────┐                   │
│                    │ scikit-learn Pipeline│                   │
│                    │  ├─ Imputation       │                   │
│                    │  ├─ Encoding         │                   │
│                    │  ├─ PowerTransformer │                   │
│                    │  ├─ StandardScaler   │                   │
│                    │  └─ XGBRegressor     │                   │
│                    └──────────┬──────────┘                   │
│                               │                              │
│                         pipeline.pkl  (joblib)               │
└───────────────────────────────┼──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                     INFERENCE PHASE                           │
│                                                              │
│   JSON Request ──► Flask /predict ──► pipeline.predict()     │
│                                            │                 │
│                                       JSON Response          │
│                                  { "predictions": 0.73 }     │
└──────────────────────────────────────────────────────────────┘
```

**Key design decision:** The entire preprocessing chain (imputation → encoding → power transformation → scaling) is encapsulated inside a single scikit-learn `Pipeline` object. The Flask API performs **zero preprocessing** — it only passes raw JSON payloads directly to `pipeline.predict()`, eliminating any risk of training–serving skew.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | `Data/data.csv` |
| **Records** | 3,500 patient samples |
| **Features** | 21 columns (20 input + 1 target) |
| **Documentation** | `Data/Vitamin_Deficiency_Dataset_Report.pdf` |
| **Missing values** | `alcohol_consumption` (~31.7 % missing), `symptoms_list` (~32.9 % missing) |

### Full Feature Reference

```
Demographics
  age                         int64    — Patient age (18–84)
  gender                      object   — Male / Female
  bmi                         float64  — Body Mass Index

Lifestyle
  smoking_status              object   — Never / Former / Current
  alcohol_consumption         object   — None / Moderate / Heavy  (31.7 % NaN)
  exercise_level              object   — Sedentary / Light / Moderate / Active
  diet_type                   object   — Omnivore / Vegetarian / Vegan / Pescatarian
  sun_exposure                object   — Low / Moderate / High

Socioeconomic
  income_level                object   — Low / Mid / High
  latitude_region             object   — Low / Mid / High

Nutrient Intake (% of Recommended Daily Allowance)
  vitamin_a_percent_rda       float64
  vitamin_c_percent_rda       float64
  vitamin_d_percent_rda       float64
  vitamin_e_percent_rda       float64
  vitamin_b12_percent_rda     float64
  folate_percent_rda          float64
  calcium_percent_rda         float64
  iron_percent_rda            float64

Clinical
  symptoms_count              int64    — Number of active symptoms (0–8)
  symptoms_list               object   — Semicolon-delimited symptom strings (32.9 % NaN)

Target
  vitamin_deficiency          float64  — Continuous deficiency score [0, 1.89]
```

### Key EDA Findings

- **Strongest symptom correlates** with deficiency score:
  - `sym_numbness_tingling` (r = 0.538)
  - `sym_bone_pain` (r = 0.532)
  - `sym_muscle_weakness` (r = 0.483)
  - `sym_memory_problems` (r = 0.478)
- **Statistically significant categorical features** (F-regression, p < 0.05):
  - `symptoms_list` (F = 1951.87)
  - `diet_type` (F = 435.18)
- **Top numeric predictors**: `vitamin_b12_percent_rda`, `iron_percent_rda`, `symptoms_count`, `vitamin_d_percent_rda`

---

## 🔄 ML Pipeline

The notebook (`vitamin_deficiency_prediction.ipynb`) implements a rigorous, reproducible pipeline:

### 1. Import Libraries
```python
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                   StandardScaler, PowerTransformer,
                                   MultiLabelBinarizer)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
from xgboost import XGBRegressor
```

### 2. Feature Engineering
- **BMI binning**: `Underweight / Normal / Overweight / Obese`
- **Age grouping**: `Youth (<30) / Adults (30–50) / Seniors (>50)`
- **Symptom multi-hot encoding**: 9 binary `sym_*` columns extracted from semicolon-delimited `symptoms_list`

### 3. Preprocessing
| Step | Method |
|---|---|
| Missing values | `alcohol_consumption` → `"Missing"` fill; `symptoms_list` → `"No Symptoms"` fill |
| Label encoding | `gender`, `diet_type`, `smoking_status`, `alcohol_consumption`, `symptoms_list`, `latitude_region` |
| Ordinal encoding | `exercise_level`, `income_level`, `sun_exposure`, `bmi_category`, `age_group` |
| Outlier clipping | IQR-based clipping fitted on training set only (no data leakage) |
| Power transformation | Yeo-Johnson applied to skewed numeric features |
| Scaling | `StandardScaler` (zero-mean, unit-variance) |

### 4. Models Evaluated
- `LinearRegression`
- `RandomForestRegressor(random_state=42)`
- `XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.01, subsample=0.8)`
- `SVR(kernel='rbf', C=0.2, epsilon=0.05)`

### 5. Validation
- Hold-out split: 80 % train / 20 % test
- 5-Fold cross-validation on all models

---

## 📈 Model Results

### Hold-out Test Set (all 30 features)

| Model | R² (Test) | MSE (Test) | MSE (Train) |
|---|---|---|---|
| Linear Regression | 0.8493 | 0.0288 | 0.0265 |
| Random Forest | 0.8623 | 0.0263 | 0.0034 |
| **XGBoost** | **0.8709** | **0.0247** | **0.0195** |
| SVR | 0.8434 | 0.0299 | 0.0171 |

### 5-Fold Cross-Validation (all 30 features)

| Model | Mean R² | Std R² | Mean MSE |
|---|---|---|---|
| Linear Regression | 0.8616 | ±0.0097 | 0.0273 |
| Random Forest | 0.8764 | ±0.0116 | 0.0244 |
| **XGBoost** | **0.8851** | **±0.0105** | **0.0227** |
| SVR | 0.8586 | ±0.0122 | 0.0279 |

> **Winner: XGBoost** — highest R² and lowest MSE in both hold-out and cross-validation experiments, with consistent variance across folds.

### Hold-out Test Set (selected features only)

After feature selection (top 4 numeric + top 4 symptom binary columns):

| Model | R² (Test) | MSE (Test) |
|---|---|---|
| Linear Regression | 0.8490 | 0.0289 |
| Random Forest | 0.8589 | 0.0270 |
| XGBoost | 0.8703 | 0.0248 |
| SVR | 0.8681 | 0.0252 |

*Feature selection causes a negligible accuracy drop while reducing dimensionality by ~73 %.*

---

## ✅ Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| pip | latest |
| Git | any recent version |

> **Optional for notebook**: Jupyter Lab / Jupyter Notebook (`pip install jupyterlab`)

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/abdo-ghg/Vitamin-Deficiency-Predction.git
cd Vitamin-Deficiency-Predction

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Train the model and generate the pipeline artifact
#    (Required before starting the API server)
python train.py
```

> `train.py` reads `Data/data.csv`, fits the full preprocessing + XGBoost pipeline, evaluates it on a held-out test set, and writes `pipeline.pkl` to the project root.

---

## 🚀 Usage / Running the App

### Option A — Jupyter Notebook (Research / EDA)

```bash
jupyter notebook vitamin_deficiency_prediction.ipynb
# or
jupyter lab vitamin_deficiency_prediction.ipynb
```

Run all cells top-to-bottom. The notebook includes:
- Full EDA (30+ visualisations)
- Preprocessing pipeline construction
- Multi-model training and cross-validation
- Feature importance and selection analysis

### Option B — Flask REST API (Production / Integration)

```bash
# Make sure pipeline.pkl exists first (run train.py)
python app.py
```

The server starts on `http://0.0.0.0:5000`.

---

## 🌐 API Endpoints

### `GET /health`

Liveness probe — confirms the server is running and the pipeline is loaded.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /predict`

Accepts a raw patient record (single object **or** a batch array) and returns the predicted vitamin deficiency score(s).

**Request Headers**
```
Content-Type: application/json
```

**Request Body — Single Record**
```json
{
  "age": 35,
  "bmi": 22.5,
  "serum_level": 18.0,
  "sun_exposure_hrs": 1.5,
  "gender": "Female",
  "primary_symptom": "bone_pain",
  "diet_type": "Vegan"
}
```

**Request Body — Batch (array)**
```json
[
  { "age": 35, "bmi": 22.5, "serum_level": 18.0, ... },
  { "age": 50, "bmi": 28.1, "serum_level": 42.0, ... }
]
```

**Success Response `200 OK`**
```json
{
  "predictions": 0.73,
  "num_samples": 1
}
```
*For a batch request `predictions` is an array of floats.*

**Error Responses**

| HTTP Code | Cause | Body (example) |
|---|---|---|
| `415` | `Content-Type` is not `application/json` | `{ "error": "Request Content-Type must be application/json" }` |
| `422` | One or more required fields are absent | `{ "error": "Missing required fields", "missing": ["gender"], "required": [...] }` |
| `500` | Internal pipeline error | `{ "error": "Prediction failed", "detail": "<traceback>" }` |

**Required Fields for `/predict`**

```
age               — numeric (float/int)
bmi               — numeric (float)
serum_level       — numeric (float)
sun_exposure_hrs  — numeric (float)
gender            — string  ("Male" | "Female" | "Other")
primary_symptom   — string  ("bone_pain" | "fatigue" | "none" | ...)
diet_type         — string  ("Omnivore" | "Vegetarian" | "Vegan" | "Pescatarian")
```

**Example — cURL**
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "age": 35,
           "bmi": 22.5,
           "serum_level": 18.0,
           "sun_exposure_hrs": 1.5,
           "gender": "Female",
           "primary_symptom": "bone_pain",
           "diet_type": "Vegan"
         }'
```

**Example — Python `requests`**
```python
import requests

payload = {
    "age": 35, "bmi": 22.5, "serum_level": 18.0,
    "sun_exposure_hrs": 1.5, "gender": "Female",
    "primary_symptom": "bone_pain", "diet_type": "Vegan"
}
resp = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(resp.json())
# → {"predictions": 0.73, "num_samples": 1}
```

---

## 📁 Project Structure

```
Vitamin-Deficiency-Predction/
│
├── Data/
│   ├── data.csv                              # 3,500-record dataset (21 features)
│   └── Vitamin_Deficiency_Dataset_Report.pdf # Full dataset documentation
│
├── Project Description/
│   └── [2026] ML Projects Milestone 1.pdf   # Academic milestone spec
│
├── vitamin_deficiency_prediction.ipynb       # End-to-end research notebook
│   ├── § 1  Import Libraries
│   ├── § 2  Load Dataset
│   ├── § 3  Data Understanding
│   ├── § 4  (Dataset info / dtypes)
│   ├── § 5  Data Preprocessing
│   │     ├── 5.1 Feature Engineering
│   │     ├── 5.2 Handling Missing Values
│   │     ├── 5.3 Encoding
│   │     ├── 5.4 Train / Test Split
│   │     ├── 5.5 Outlier Handling
│   │     ├── 5.6 Feature Transformation
│   │     └── 5.7 Feature Scaling
│   ├── § 6  Model Training
│   ├── § 7  Cross-Validation
│   ├── § 8  EDA (Visualisations)
│   │     ├── 8.1 Numeric Distributions
│   │     ├── 8.2 Scatter / Box Plots vs Target
│   │     ├── 8.3 Categorical Column Analysis
│   │     ├── 8.4 Correlation Heatmap
│   │     └── 8.5 Symptoms Analysis
│   └── § 9  Feature Selection + Re-evaluation
│
├── train.py                                  # Standalone training script → pipeline.pkl
├── app.py                                    # Flask REST API server
├── pipeline.pkl                              # Serialised pipeline (generated by train.py)
│
├── requirements.txt                          # Python dependencies
├── .gitignore
└── README.md                                 # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with a descriptive message
   ```bash
   git commit -m "feat: add SHAP explainability to API response"
   ```
4. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a **Pull Request** targeting `main`

### Development Conventions
- Follow [PEP 8](https://peps.python.org/pep-0008/) style.
- Add docstrings to any new public functions.
- Ensure `train.py` can be run end-to-end without errors before submitting a PR.

---

## 📄 License

This project is developed for **educational and academic purposes** as part of the 2026 ML Projects curriculum.

---

<div align="center">

**Built with ❤️ using Python, scikit-learn, XGBoost & Flask**

</div>
