# 🧬 Vitamin Deficiency Prediction

A machine learning project that predicts the likelihood of vitamin deficiency in individuals based on demographic data, lifestyle factors, dietary intake, and clinical symptoms.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Vitamin deficiencies affect billions of people worldwide and can lead to serious health complications if left undetected. This project builds a predictive model that estimates the degree of vitamin deficiency using patient health profiles, enabling early detection and intervention.

The model takes into account a wide range of features — from basic demographics (age, gender, BMI) to lifestyle habits (smoking, alcohol, exercise, diet) and nutrient intake levels (vitamins A, C, D, E, B12, folate, calcium, iron) — to produce a continuous deficiency score.

---

## 📊 Dataset

- **Source:** `Data/data.csv`  
- **Records:** 3,500 samples  
- **Features:** 21 columns  
- **Documentation:** See `Data/Vitamin_Deficiency_Dataset_Report.pdf` for detailed dataset documentation.

### Feature Overview

| Category | Features |
|---|---|
| **Demographics** | `age`, `gender`, `bmi` |
| **Lifestyle** | `smoking_status`, `alcohol_consumption`, `exercise_level`, `diet_type`, `sun_exposure` |
| **Socioeconomic** | `income_level`, `latitude_region` |
| **Nutrient Intake (% RDA)** | `vitamin_a`, `vitamin_c`, `vitamin_d`, `vitamin_e`, `vitamin_b12`, `folate`, `calcium`, `iron` |
| **Clinical** | `symptoms_count`, `symptoms_list` |
| **Target** | `vitamin_deficiency` (continuous, 0–1 scale) |

### Sample Symptoms

Symptoms include: `bone_pain`, `dry_skin`, `night_blindness`, `numbness_tingling`, `memory_problems`, and more.

---

## 🔄 Project Workflow

The notebook (`vitamin_deficiency_prediction.ipynb`) follows a structured ML pipeline:

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
2. **Load Dataset** — Read and inspect the CSV data
3. **Data Understanding** — Explore data types, distributions, and summary statistics
4. **Exploratory Data Analysis (EDA)** — Visualize feature relationships and patterns
5. **Preprocessing** — Handle missing values, encode categorical variables, feature engineering
6. **Train/Test Split & Scaling** — Split data and apply `StandardScaler`
7. **Model Training** — Train models including `RandomForestClassifier` and `LinearRegression`
8. **Evaluation** — Assess performance with accuracy, classification report, and confusion matrix

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.12** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | Machine learning (models, preprocessing, evaluation) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/abdo-ghg/Vitamin-Deficiency-Predction.git
cd Vitamin-Deficiency-Predction

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Launch the notebook
jupyter notebook vitamin_deficiency_prediction.ipynb
```

---

## 📁 Project Structure

```
Vitamin-Deficiency-Predction/
├── Data/
│   ├── data.csv                              # Dataset (3,500 records × 21 features)
│   └── Vitamin_Deficiency_Dataset_Report.pdf # Dataset documentation
├── Project Description/
│   └── [2026] ML Projects Milestone 1.pdf    # Project requirements & milestones
├── vitamin_deficiency_prediction.ipynb        # Main Jupyter notebook
└── README.md                                 # This file
```

---

## 📈 Results

> _Results will be updated once the full pipeline (EDA → Modeling → Evaluation) is complete._

Key metrics to be reported:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1)
- **Confusion Matrix**

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is for educational and academic purposes.
