# 💳 Credit Scoring Model — Production-Grade Machine Learning Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CodeAlpha](https://img.shields.io/badge/CodeAlpha-Internship-orange.svg)](https://www.codealpha.tech/)
[![Framework: LightGBM](https://img.shields.io/badge/Framework-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
This repository contains an enterprise-grade Machine Learning solution designed to classify customer creditworthiness into three distinct categories (`Good`, `Standard`, `Poor`). Developed for **CodeAlpha Machine Learning Internship (Task 1)**, the project follows a production-ready architecture: clean data engineering, domain-driven feature extraction, high-performance gradient boosting models (`LightGBM`, `XGBoost`), model interpretability, and automated serialization.

---

## 🛠 Project Architecture & File Structure

```
CodeAlpha_CreditScoringModel/
│
├── 📊 Credit_Score_EDA_Granular_EN.ipynb            # Publication-grade Exploratory Data Analysis
├── 🚀 Credit_Score_Machine_Learning_Granular_EN.ipynb # Production ML Pipeline & Model Benchmarking
├── 📦 credit_scoring_best_lgbm.pkl                       # Serialized Champion Model (LightGBM)
├── 🏷 target_label_encoder.pkl                           # Target Label Encoder Artifact
├── 📄 final_credit_score_predictions.csv                 # Generated Test Set Predictions
└── 📝 README.md                                         # Project Documentation
```

---

## 🔬 Exploratory Data Analysis (EDA)
The EDA pipeline (`Credit_Score_EDA_Granular_EN.ipynb`) performs statistical and diagnostic profiling:
* **Target Distribution:** Evaluates class balance (`Standard`: 53.1%, `Poor`: 29.0%, `Good`: 17.8%).
* **Data Sanitization & Regex Utilities:** Cleans noisy trailing symbols (`_`) and converts textual history (`X Years and Y Months`) into numerical months.
* **Missing Value Auditing:** Quantifies column missingness percentages.
* **Bivariate Financial Profiling:** Identifies core credit drivers (`Annual_Income`, `Outstanding_Debt`, `Interest_Rate`) across credit ratings using high-DPI box plots.
* **High-Resolution Multicollinearity Heatmap:** Maps feature correlation to prevent redundancy.

---

## 💡 Financial Domain Feature Engineering
To maximize predictive performance, 8 domain-specific financial ratios were constructed:

| Engineered Feature | Mathematical Formula / Definition |
| :--- | :--- |
| **Debt-to-Income Ratio (DTI)** | `Outstanding_Debt / ((Annual_Income / 12) + 1e-5)` |
| **EMI-to-Salary Ratio** | `Total_EMI_per_month / (Monthly_Inhand_Salary + 1e-5)` |
| **Total Monthly Outflow** | `Total_EMI_per_month + Amount_invested_monthly` |
| **Net Saving Ratio** | `Monthly_Balance / (Monthly_Inhand_Salary + 1e-5)` |
| **Debt per Loan** | `Outstanding_Debt / (Num_of_Loan + 1)` |
| **Delinquency Index per Bank** | `Num_of_Delayed_Payment / (Num_Bank_Accounts + 1)` |
| **Utilization per Credit Card** | `Credit_Utilization_Ratio / (Num_Credit_Card + 1)` |
| **Inquiries per Bank Account** | `Num_Credit_Inquiries / (Num_Bank_Accounts + 1)` |

---

## 🤖 Model Benchmarking & Evaluation

Four ensemble classifiers were evaluated on a stratified holdout split (80% Train / 20% Validation):

| Model Algorithm | Accuracy | Macro F1-Score | ROC-AUC (OvR) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM Classifier** | **82.4%** | **0.8120** | **0.9150** | **Champion Model** 🏆 |
| **XGBoost Classifier** | 81.8% | 0.8055 | 0.9090 | Candidate |
| **Random Forest Classifier** | 79.5% | 0.7780 | 0.8870 | Benchmark |
| **Extra Trees Classifier** | 78.2% | 0.7620 | 0.8730 | Benchmark |

---

## ⚡ Quick Start & Execution

### 1. Clone Repository
```bash
git clone https://github.com/YourUsername/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel
```

### 2. Environment Setup & Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib
```

### 3. Run Pipeline
```bash
jupyter notebook Credit_Score_Machine_Learning_Granular_EN.ipynb
```

---

## 👤 Author
* **Developer:** Ziad Ali Fathy Mohamed
* **Program:** CodeAlpha Machine Learning Internship
* **Repository Name:** `CodeAlpha_CreditScoringModel`
