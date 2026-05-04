# 💰 Income Prediction Using Machine Learning

A machine learning project to predict whether an individual's annual income exceeds **$50,000** using the 1994 US Census Bureau dataset.

> **AMS 580 Project** — Stony Brook University

---

## 📌 Abstract

This project develops and compares multiple classification models to predict binary income categories (≤$50K vs >$50K) using demographic and employment features. The pipeline covers data preprocessing, feature engineering, class imbalance handling, model development, and ensemble methods.

---

## 📁 Project Structure

```
Income-Prediction-using-Machine-Learning/
├── data/
│   ├── train.csv
│   └── test.csv
├── results/
│   ├── model_comparison.png
│   ├── xgboost_feature_importance.png
│   ├── random_forest_feature_importance.png
│   └── confusion_matrices/
├── income_prediction.R
├── requirements.R
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | 1994 US Census Bureau (Kohavi & Becker) |
| Training set | 32,561 entries |
| Test set | 16,281 entries |
| Target variable | Income (≤$50K or >$50K) |
| Features | 14 predictor variables |

### Key Features

| Category | Feature | Type |
|----------|---------|------|
| Demographic | Age, Race, Sex, Native Country | Mixed |
| Education | Education level, Education-num | Categorical/Numeric |
| Employment | Workclass, Occupation, Hours-per-week | Categorical/Numeric |
| Financial | Capital gain, Capital loss | Continuous |
| Personal | Marital status, Relationship | Categorical |

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Trimmed whitespace and converted character variables to factors
- Combined train/test for consistent preprocessing
- Dummy encoding of categorical variables using `dummyVars`

### 2. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique)
- Used K=5 nearest neighbors to generate synthetic samples for the minority class (>$50K)

### 3. Feature Selection
- Removed near-zero variance features to reduce complexity and overfitting

---

## 🤖 Models Developed

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline model using `glm()` with binomial family |
| Random Forest | 300 decision trees, mtry = sqrt(predictors) |
| Neural Network | 5 hidden nodes, weight decay = 0.01, max 300 iterations |
| XGBoost | 200 rounds, eta = 0.05, max depth = 8 |
| LightGBM | 100 rounds, learning rate = 0.1, binary objective |
| Meta-Model (XGB + RF) | Stacked ensemble with logistic regression meta-learner |
| Meta-Model (XGB + RF + LGBM) | Three-model stacked ensemble |
| Meta-Model (XGB + LGBM) | Two-model stacked ensemble |

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|-------------|
| Logistic Regression | 0.8079 | 0.8131 | 0.8062 |
| Random Forest | 0.8604 | 0.6288 | 0.9320 |
| Neural Network | 0.8021 | 0.8406 | 0.7898 |
| XGBoost | 0.8690 | 0.6684 | 0.9326 |
| LightGBM | 0.8704 | 0.6677 | 0.9341 |
| **Meta-Model (XGB + RF)** | **0.8716** | **0.6537** | **0.9387** |
| Meta-Model (XGB + RF + LGBM) | 0.8680 | 0.6473 | 0.9367 |
| Meta-Model (XGB + LGBM) | 0.8690 | 0.6492 | 0.9387 |

### Key Findings
- **Best model:** Meta-Model (Stacked XGBoost + Random Forest) with **87.16% accuracy**
- Random Forest and XGBoost achieved the best specificity (0.93) among individual models
- Neural Network had the lowest overall performance
- All models performed better at identifying low-income individuals (higher specificity than sensitivity)

### Feature Importance

**Top features across models:**
- `education-num` — strongest predictor (education level closely tied to income)
- `capital-gain` — investment income highly significant
- `age` — job experience impacts salary
- `marital-status` (Married-civ-spouse) — highly significant
- `hours-per-week` — weekly working hours

> 📁 See the `results/` folder for feature importance plots and confusion matrices.

### Ensemble Methods Summary

| Ensemble Type | Technique | Result |
|---------------|-----------|--------|
| Bagging | Random Forest (300 trees) | Reduced variance, 0.86 accuracy |
| Boosting | XGBoost + LightGBM | Best individual accuracy ~0.87 |
| Stacking | XGB + RF meta-model | **Best overall: 0.8716** |

---

## 🏆 Final Model

**Meta-Model: Stacked XGBoost + Random Forest**

Chosen because:
1. Highest accuracy (87.16%) among all models
2. Best balance between sensitivity (0.65) and specificity (0.94)
3. Combines XGBoost's gradient boosting with Random Forest's variance reduction
4. Better computational efficiency than the three-model ensemble

---

## 🔮 Practical Applications

- **Economic Policy** — Understanding income determinants
- **Targeted Marketing** — Identifying high-income individuals
- **Financial Services** — Credit eligibility assessment
- **Socioeconomic Research** — Studying income inequality
- **Human Resources** — Compensation and talent acquisition

---

## ⚠️ Limitations

- Dataset from 1994 may not reflect current economic conditions
- Binary classification only (≤$50K vs >$50K)
- Missing features like geographic location and industry sector

---

## 🔭 Future Work

- Additional feature engineering (polynomial features, interaction terms)
- Deep learning architectures
- Time-series analysis for income trends
- Fairness and bias analysis across demographic groups

---

## 🚀 Getting Started

```r
# Install required packages
source("requirements.R")

# Run the full pipeline
source("income_prediction.R")
```
