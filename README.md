# Business Revenue Regression

An end-to-end machine learning project to predict annual business revenue using operational, demographic, and digital marketing features. This repository covers the complete data science lifecycle, from exploratory data analysis and feature engineering to multi-model benchmarking, hyperparameter tuning, model evaluation, and artifact serialization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Architecture](#dataset-architecture)
- [Project Structure](#project-structure)
- [Methodology and Pipeline](#methodology-and-pipeline)
  - [1. Data Inspection and Cleaning](#1-data-inspection-and-cleaning)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Outlier Treatment](#3-outlier-treatment)
  - [4. Feature Engineering and Selection](#4-feature-engineering-and-selection)
  - [5. Preprocessing Pipeline](#5-preprocessing-pipeline)
  - [6. Model Benchmarking](#6-model-benchmarking)
  - [7. Hyperparameter Tuning](#7-hyperparameter-tuning)
  - [8. Model Evaluation and Diagnostics](#8-model-evaluation-and-diagnostics)
- [Benchmark Results](#benchmark-results)
- [Key Business Insights and Feature Drivers](#key-business-insights-and-feature-drivers)
- [Model Artifacts](#model-artifacts)
- [Installation and Setup](#installation-and-setup)
- [Inference Guide](#inference-guide)
- [License](#license)

---

## Project Overview

Accurate revenue forecasting is critical for operational resource allocation, budgeting, and commercial valuation. This project implements a robust supervised regression framework to predict `annual_revenue` across diverse enterprise profiles.

The workflow evaluates ten distinct regression algorithms ranging from regularized linear models to ensemble and gradient-boosted decision trees, establishing a production-ready scoring pipeline persisted with reusable metadata.

---

## Problem Statement

Given a set of operational metrics (team size, founder experience), marketing spend (monthly ad spend, website traffic), business categorization, regional location, and customer satisfaction metrics, the objective is to build an accurate predictive regression model that minimizes forecasting error and generalizes effectively to unseen business records.

---

## Dataset Architecture

The dataset consists of 1,200 commercial entity records with 13 features:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Integer | Founder or business maturity age (years) |
| `experience_years` | Integer | Years of industry experience |
| `experience_proxy` | Float | Latent proxy score for operational experience |
| `monthly_ad_spend` | Float | Monthly budget allocated to digital advertising |
| `website_visits` | Integer | Aggregate monthly website traffic |
| `team_size` | Integer | Total full-time employee headcount |
| `customer_rating` | Float | Average customer satisfaction score (scale 1.0 - 5.0) |
| `region` | Categorical | Operating region (Cairo, Alexandria, Delta, Upper Egypt) |
| `business_type` | Categorical | Sector category (Ecommerce, Retail, Services, Manufacturing) |
| `subscription_plan` | Categorical | Service tier (Basic, Standard, Premium) |
| `noise_feature_1` | Float | Synthetic uninformative random distribution |
| `noise_feature_2` | Float | Synthetic uninformative random distribution |
| `annual_revenue` | Float | Target variable: Annual business revenue |

---

## Project Structure

```text
Business Revenue Regression/
|-- Dataset/
|   `-- dataset.csv
|-- Model/
|   |-- artifacts/
|   |   |-- best_model_pipeline.joblib
|   |   |-- preprocessor.joblib
|   |   |-- model_metadata.json
|   |   |-- feature_info.json
|   |   `-- feature_importances.csv
|   `-- business_revenue_regression.ipynb
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

---

## Methodology and Pipeline

### 1. Data Inspection and Cleaning
- Verified completeness across all records (zero null values detected).
- Validated uniqueness (no duplicate observations found).
- Ensured appropriate data type formatting for all numerical and categorical fields.

### 2. Exploratory Data Analysis
- Assessed distribution symmetry for the continuous target variable (`annual_revenue`).
- Evaluated Pearson correlation coefficients between predictors and the target.
- Verified absence of severe multimodality across operational variables.

### 3. Outlier Treatment
- Identified distribution tails using the Interquartile Range (IQR) method:
  `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`
- Applied winsorization (clipping) to retain full sample size while mitigating extreme leverage points.

### 4. Feature Engineering and Selection
- Removed synthetic uninformative features (`noise_feature_1`, `noise_feature_2`) based on statistically insignificant correlation (|r| < 0.05).
- Created interaction and ratio features:
  - `ad_spend_per_visit`: Effective customer acquisition cost efficiency indicator (`monthly_ad_spend / (website_visits + 1)`).
  - `revenue_potential`: Composite scale metric (`experience_years * team_size`).
  - `experience_rating`: Domain competence interaction (`experience_years * customer_rating`).

### 5. Preprocessing Pipeline
Integrated within a scikit-learn `ColumnTransformer`:
- Numerical variables: `StandardScaler` (zero-mean, unit-variance standardization).
- Categorical variables: `OneHotEncoder(drop='first', sparse_output=False)` to prevent the dummy variable trap.
- Train-test split: 80% training (960 rows) and 20% test (240 rows) stratified with a fixed random seed (42).

### 6. Model Benchmarking
Ten regression algorithms were evaluated using 5-fold cross-validation on the training split:
- Ordinary Least Squares (Linear Regression)
- L2 Regularized Regression (Ridge)
- L1 Regularized Regression (Lasso)
- Combined Regularization (ElasticNet)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Extreme Gradient Boosting (XGBoost)
- Light Gradient Boosting Machine (LightGBM)

### 7. Hyperparameter Tuning
Exhaustive parameter search via `GridSearchCV` across 5-fold cross-validation for top candidate models, optimizing for R-squared (R2).

### 8. Model Evaluation and Diagnostics
- Evaluated against held-out test data across standard regression metrics: R2, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
- Comprehensive residual diagnostics: residual vs. predicted scatter, residual frequency distribution, and Quantile-Quantile (Q-Q) normality plots.

---

## Benchmark Results

### 5-Fold Cross-Validation Baseline Comparison (Training Split)

| Model | CV R2 Mean | CV R2 Std | CV MAE | CV RMSE |
| :--- | :---: | :---: | :---: | :---: |
| Ridge Regression | 0.8533 | 0.0164 | 14,824 | 18,963 |
| Lasso Regression | 0.8530 | 0.0162 | 14,842 | 18,979 |
| Linear Regression | 0.8530 | 0.0162 | 14,842 | 18,980 |
| Gradient Boosting | 0.8179 | 0.0176 | 16,539 | 21,142 |
| LightGBM | 0.8072 | 0.0185 | 17,058 | 21,753 |
| Random Forest | 0.7778 | 0.0216 | 18,152 | 23,347 |
| XGBoost | 0.7757 | 0.0236 | 18,201 | 23,450 |
| ElasticNet | 0.7723 | 0.0229 | 18,542 | 23,643 |
| AdaBoost | 0.7478 | 0.0104 | 19,408 | 24,890 |
| Decision Tree | 0.5407 | 0.0330 | 26,283 | 33,540 |

### Final Test Set Evaluation (Held-Out Test Set)

| Metric | Training Set | Test Set |
| :--- | :---: | :---: |
| R2 Score | 0.8595 | 0.8400 |
| Mean Absolute Error (MAE) | 14,485.50 | 16,210.28 |
| Root Mean Squared Error (RMSE) | 18,598.17 | 20,031.96 |
| Mean Absolute Percentage Error (MAPE) | 6.59% | 7.46% |

The regularized Ridge model demonstrated superior generalizability without signs of overfitting, maintaining an average prediction error of approximately 7.46% on unseen records.

---

## Key Business Insights and Feature Drivers

Based on regularized linear coefficients, the primary drivers influencing annual revenue are:

1. `team_size` (+29,628): Organizational capacity and employee headcount serve as the strongest direct operational indicator of revenue output.
2. `subscription_plan_Premium` (+27,168): Businesses utilizing premium commercial tiers show a substantial revenue premium relative to baseline plans.
3. `experience_years` (+22,069): Founder and leadership operational tenure strongly correlates with commercial maturity and scale.
4. `business_type_Retail` (+15,385): Retail operations demonstrate elevated top-line sales figures in comparison with services and e-commerce baselines.
5. `monthly_ad_spend` (+9,170): Marketing investment functions as a statistically verifiable revenue accelerator when supported by appropriate web traffic conversion.

---

## Model Artifacts

All training metadata, feature specifications, and serialized models are stored under `Model/artifacts/`:

- `best_model_pipeline.joblib`: Complete trained pipeline comprising the `ColumnTransformer` preprocessor and the optimized `Ridge` estimator.
- `preprocessor.joblib`: Serialized preprocessing pipeline for isolated feature transformations.
- `model_metadata.json`: Comprehensive model metrics, hyperparameters, dataset specifications, and benchmark tables.
- `feature_info.json`: Explicit mapping of input numeric, categorical, and encoded feature columns.
- `feature_importances.csv`: Complete ranking of feature importance coefficients.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NumiKun/Business-Revenue-Regression.git
cd Business-Revenue-Regression
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## Inference Guide

To load the trained pipeline and generate revenue predictions on new input records:

```python
import joblib
import pandas as pd

# Load the serialized production pipeline
pipeline = joblib.load('Model/artifacts/best_model_pipeline.joblib')

# Define new business observations
new_data = pd.DataFrame({
    'age': [35, 48],
    'experience_years': [12, 22],
    'experience_proxy': [11.8, 21.5],
    'monthly_ad_spend': [25000.0, 45000.0],
    'website_visits': [65000, 120000],
    'team_size': [45, 95],
    'customer_rating': [4.2, 4.8],
    'region': ['Cairo', 'Alexandria'],
    'business_type': ['Ecommerce', 'Retail'],
    'subscription_plan': ['Standard', 'Premium'],
    'ad_spend_per_visit': [25000.0 / 65001, 45000.0 / 120001],
    'revenue_potential': [12 * 45, 22 * 95],
    'experience_rating': [12 * 4.2, 22 * 4.8]
})

# Generate annual revenue predictions
predictions = pipeline.predict(new_data)

for idx, pred in enumerate(predictions):
    print(f"Company {idx + 1} Projected Annual Revenue: ${pred:,.2f}")
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.
