# Data Pre-processing

## Data Quality, Feature Engineering, Leakage Assessment & Reproducible Machine Learning Pipeline


---

## 1. Project Overview

This project develops a reproducible machine learning workflow for a lending institution, with particular emphasis on **data quality, feature reliability, leakage prevention, fairness, and reproducibility**.

The objective is not simply to maximise predictive performance, but to develop a machine learning solution that can be reliably evaluated, reproduced, audited, and eventually deployed within a credit-risk environment.

The project covers four major components:

1. **Data Quality Audit:** identification and remediation of the ten most significant data-quality problems in the dataset.
2. **Feature Engineering:** development and documentation of at least 15 engineered features across a minimum of four feature families.
3. **Leakage Assessment:**  identification, quantification, and remediation of two intentionally planted sources of data leakage.
4. **Reproducible ML Pipeline:** development of a single Scikit-learn pipeline containing preprocessing, feature transformation, and model training, evaluated using group-aware cross-validation.

---

## 2. Project Objectives

The project aims to:

* Assess the quality and reliability of the available lending data.
* Identify missingness, invalid values, duplicates, outliers, inconsistencies, and other data-quality issues.
* Determine appropriate missing-data mechanisms where applicable.
* Apply evidence-based data remediation strategies.
* Engineer predictive features while maintaining clear data lineage.
* Ensure engineered features use only information available at the appropriate point in time.
* Detect and remove sources of target and temporal leakage.
* Quantify the effect of leakage on model performance.
* Build an end-to-end Scikit-learn pipeline.
* Evaluate model performance using group-aware cross-validation.
* Produce a model artefact that can be independently reloaded and executed.
* Establish a reproducible workflow suitable for credit-risk modelling.

---

## 3. Repository Structure

```text
.
├── README.md
│
├── notebooks/
│   └── Lab_2_Data_Preparation_and_Feature_Engineering.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── pipeline.py
│
├── models/
│   └── final_pipeline.joblib
│
├── reports/
│   └── Assignment_1_Technical_Report.pdf
│
└── requirements.txt
```

> **Note:** Directory and file names should be adjusted to match the actual submitted repository structure.

---

# 4. Methodology

## 4.1 Data Quality Audit

The dataset was subjected to a comprehensive quality assessment before modelling.

The audit considered:

* Missing values
* Missingness proportions
* Duplicate observations
* Invalid or impossible values
* Outliers
* Data-type inconsistencies
* Cardinality problems
* Category inconsistencies
* Temporal inconsistencies
* Suspicious distributions
* Potential leakage-related variables

The ten columns with the most significant quality concerns were then selected for detailed investigation.

For each problematic variable, the analysis documents:

| Component   | Description                         |
| ----------- | ----------------------------------- |
| Issue       | Identified data-quality problem     |
| Evidence    | Statistical or diagnostic evidence  |
| Missingness | MCAR, MAR, or MNAR where applicable |
| Remediation | Selected treatment                  |
| Rationale   | Why the treatment is appropriate    |

All remediation decisions are based on observed evidence rather than assumptions.

---

## 4.2 Missing-Data Assessment

Missing values were assessed using both descriptive statistics and relationships with other variables.

Where applicable, missingness was classified as:

* **MCAR:  Missing Completely At Random:** missingness is unrelated to observed or unobserved information.
* **MAR:  Missing At Random:** missingness can be explained by observed variables.
* **MNAR:  Missing Not At Random:** missingness is potentially related to the unobserved value itself or another unobserved mechanism.

The classification is treated as an analytical judgement supported by evidence rather than as something that can be established solely from the missing-value percentage.

---

# 5. Feature Engineering

A minimum of 15 engineered features are developed across at least four feature families.

The feature dictionary records the complete lineage of each feature.

Each feature is documented using:

| Field              | Description                                 |
| ------------------ | ------------------------------------------- |
| Feature name       | Name used in the modelling dataset          |
| Definition         | What the feature represents                 |
| Feature family     | Category of engineered feature              |
| Computation window | Period used to calculate the feature        |
| Upstream fields    | Source variables                            |
| Availability time  | When each source variable becomes available |
| Fairness note      | Potential bias or ethical concern           |

## Feature Lineage

A key principle of the feature-engineering process is **temporal validity**.

For every engineered variable:

```text
Raw Source Field
       ↓
Data Cleaning
       ↓
Feature Transformation
       ↓
Engineered Feature
       ↓
Model Input
```

No feature should incorporate information that would not have been available at the prediction time.

This is particularly important in credit-risk modelling because variables generated after a lending decision may artificially inflate model performance.

---

# 6. Data Leakage Assessment

Two intentionally planted sources of data leakage are investigated.

For each leakage source, the analysis documents:

1. Where the leakage occurs.
2. How the leakage enters the modelling process.
3. Which recognised leakage mechanism it represents.
4. Model performance before leakage removal.
5. Model performance after leakage removal.
6. The resulting change in ROC-AUC.
7. The recommended governance or development control.

The analysis compares:

```text
Model with Leakage
        ↓
Observed ROC-AUC
        ↓
Leakage Identified
        ↓
Leakage Removed
        ↓
Re-trained Model
        ↓
Leakage-free ROC-AUC
```

The difference between the two models demonstrates the practical impact of leakage on offline model evaluation.

---

# 7. Leakage Prevention Principles

The project follows several controls to reduce the possibility of leakage:

### 7.1 Temporal availability

Features must only use information available before or at the prediction point.

### 7.2 Train/test separation

Information from validation or test observations must not influence preprocessing or model fitting.

### 7.3 Pipeline-based preprocessing

Transformations such as:

* Imputation
* Scaling
* Encoding
* Feature selection
* Other learned transformations

are performed inside the Scikit-learn pipeline.

### 7.4 Cross-validation isolation

Preprocessing is fitted independently within each training fold rather than being fitted once on the complete dataset.

### 7.5 Group-aware validation

Where observations belong to the same entity/group, groups are kept together during cross-validation to reduce overly optimistic estimates caused by related observations appearing in both training and validation folds.

---

# 8. Reproducible Machine Learning Pipeline

The final modelling workflow is implemented as a single Scikit-learn `Pipeline`.

Conceptually:

```text
Raw Dataset
     │
     ▼
Data Selection
     │
     ▼
Column-specific Preprocessing
     │
     ├── Numerical Features
     │       ├── Missing-value treatment
     │       └── Scaling / transformation
     │
     └── Categorical Features
             ├── Missing-value treatment
             └── Encoding
     │
     ▼
Machine Learning Model
     │
     ▼
Predicted Probability
     │
     ▼
ROC-AUC
```

The pipeline ensures that preprocessing and model fitting form one reproducible object.

This prevents preprocessing operations performed before cross-validation from contaminating validation folds.

---

# 9. Model Evaluation

Model performance is evaluated using **group-aware cross-validation**.

The primary evaluation metric is:

**ROC-AUC (Receiver Operating Characteristic 
Area Under the Curve)**

The final reported result is:

```text
Cross-validated ROC-AUC (leak-free, group-aware, 5-fold): 0.6274 (+/- 0.0038)
Cross-validated ROC-AUC (with two planted leaks included): 1.0000 (+/- 0.0000)
Offline AUC inflation attributable to leakage: +0.3747
```

The final model performance should be interpreted alongside the data-quality and leakage findings rather than considered in isolation.

A slightly lower ROC-AUC after leakage removal is preferable to an artificially high score produced by information that would not be available during real-world prediction.

---

# 10. Model Persistence

The final fitted pipeline is saved using `joblib`.

Example:

```python
import joblib

joblib.dump(final_pipeline, "models/final_pipeline.joblib")
```

The saved object contains the preprocessing and modelling workflow required to generate predictions.

It can subsequently be loaded using:

```python
import joblib

pipeline = joblib.load("models/final_pipeline.joblib")
```

and used for prediction:

```python
predictions = pipeline.predict(X_new)
probabilities = pipeline.predict_proba(X_new)[:, 1]
```

The objective is for the marker to be able to load and execute the saved pipeline **without manually reproducing preprocessing steps**.

---

# 11. Reproducibility

The analysis is designed to be reproducible from the submitted notebook, source code, data, and environment specification.

The workflow should therefore document:

* Python version
* Major package versions
* Random seeds
* Dataset structure
* Feature-engineering logic
* Cross-validation strategy
* Model configuration
* Pipeline configuration

Required dependencies are recorded in:

```text
requirements.txt
```

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Conda environments:

```bash
conda create -n dsa8401 python=3.11
conda activate dsa8401
pip install -r requirements.txt
```

---

# 12. Running the Project

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/Lab_2_Data_Preparation_and_Feature_Engineering.ipynb
```

Run the notebook sequentially from the first cell to the final cell.

The notebook should generate or load the required processed data and produce the final model pipeline.

---

# 13. Technical Design Principles

The project follows these principles:

### Evidence over assumptions

Data-quality decisions are supported by statistical evidence, diagnostic plots, validation checks, or other observable characteristics of the dataset.

### Reproducibility over manual processing

Preprocessing required for modelling is incorporated into the machine learning pipeline wherever appropriate.

### Temporal validity

A feature is considered valid only when its source information would have been available at the time the prediction was required.

### Leakage prevention

Validation information, future information, and target-derived information must not influence model training.

### Group-aware validation

Where observations are related by a common entity, group-aware splitting is used to obtain a more realistic estimate of generalisation performance.

### Responsible credit modelling

Feature engineering considers potential fairness implications, especially where variables may indirectly encode socioeconomic status, demographic characteristics, geography, or other sensitive characteristics.

---

# 14. Key Deliverables

The completed submission contains:

* [x] Data-quality audit
* [x] Identification of ten major data-quality issues
* [x] Missing-data classification where applicable
* [x] Evidence-based remediation strategies
* [x] Feature dictionary
* [x] At least 15 engineered features
* [x] At least four feature families
* [x] Feature lineage and availability times
* [x] Fairness considerations
* [x] Leakage assessment
* [x] Leakage impact comparison
* [x] Leakage prevention recommendations
* [x] End-to-end Scikit-learn pipeline
* [x] Group-aware cross-validation
* [x] Cross-validated ROC-AUC
* [x] Persisted `joblib` model
* [x] Reproducibility instructions

---

# 15. Final Model Record

| Item                          | Result                          |
| ----------------------------- | -------------------------------- |
| Model                         | `LogisticRegression` inside a `ColumnTransformer` + `Pipeline` |
| Validation strategy           | `GroupKFold` (5 folds, grouped by resolved `customer_id`) |
| Primary metric                | ROC-AUC                         |
| Final CV ROC-AUC (leak-free)  | 0.6274 (+/- 0.0038)             |
| CV ROC-AUC with leaks (reference only) | 1.0000 (+/- 0.0000)      |
| Number of engineered features | 20 engineered + 5 lightly-transformed raw = 25 total model inputs (22 numerical, 3 categorical) |
| Feature families              | 5 (RFM/behavioural, cyclical/temporal, monetary transforms, data-quality flags, velocity/diversity) |
| Leakage sources removed       | 2 (`manual_review_score`, `settlement_status`) |
| Saved pipeline                | `models/final_pipeline.joblib` |
| Random seed                   | No stochastic model component (`GroupKFold` is deterministic; `LogisticRegression` uses the deterministic `lbfgs` solver); `random_state` values 1-7 in the notebook are used only for illustrative `.sample()` calls |

---

## 16. Conclusion

This project demonstrates a complete data-preparation and machine-learning workflow designed for a credit-risk setting. The emphasis is placed on **data quality, temporal validity, leakage prevention, fairness, reproducibility, and reliable model evaluation**.

The final model should therefore be assessed not only by its ROC-AUC, but also by whether its predictions are based on information that would genuinely be available at decision time and whether the complete modelling process can be independently reproduced.

> **Core principle:** A reliable credit-risk model is not simply a model with high predictive performance; it is a model whose performance can be trusted, explained, reproduced, and governed.
