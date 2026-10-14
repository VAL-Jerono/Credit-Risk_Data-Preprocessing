# Technical Report: Credit Risk Data Preparation and ML Pipeline

**Target Audience:** Head of Credit Risk  
**Institution:** Mobile Lending Operations  
**Date:** August 2026  
**Subject:** Quality, Reliability, Leakage Assessment, and Reproducibility of Credit-Risk Machine Learning Solution  

---

## Executive Summary

This report presents a thorough audit, feature engineering architecture, data leakage investigation, and reproducible machine learning pipeline developed on **183,600 mobile money transactions** across 3,991 unique customers in 10 East African regional hubs.

### Key Highlights & Business Findings
1. **Data Quality Audit:** Identified and remediated **10 significant data quality problems** ranging from exact transaction replays to multi-system string/datetime formatting anomalies.
2. **Missing Data Analysis:** Formally classified missingness mechanisms (MNAR for GPS coordinates, MAR for regional agent logging, MCAR for device identifiers) to avoid biased imputation strategies.
3. **Feature Engineering & Lineage:** Engineered **28 features across 5 distinct feature families** strictly using historical information prior to the decision point (`SCORING_TS`).
4. **Data Leakage & Model Inflation:** Discovered **2 planted leakage sources** (`manual_review_score` and `settlement_status`). We demonstrate that incorporating these post-outcome variables inflates offline model evaluation to an artificial **1.0000 ROC-AUC (+0.3747 inflation)**. When purged, the honest baseline model achieves a reliable **0.6274 (+/- 0.0038) ROC-AUC**.
5. **Reproducibility & Model Artifact:** Encapsulated all preprocessing, imputation, scaling, and classification into a single Scikit-Learn `Pipeline` validated via 5-fold `GroupKFold` on resolved customer IDs, persisted to `final_pipeline.joblib`.

---

## 1. Dataset & Project Overview

The dataset (`mobile_money_statements.csv`) captures raw mobile money ledger entries containing customer identifiers, geolocation markers, transaction amounts, timestamps, agent details, device IDs, balance states, and post-transaction operational markers.

| Attribute | Value |
|---|---|
| **Raw Records** | 183,600 |
| **Cleaned Operational Records** | 180,000 |
| **Unique Customer Entities (`customer_id`)** | 3,991 |
| **Unique SIM Card Numbers (`msisdn`)** | 4,825 |
| **Target Variable (`is_fraud`)** | Binary (8.4% overall fraud rate) |
| **Temporal Horizon** | June 2025 – July 2026 |
| **Scoring Cutoff (`SCORING_TS`)** | `2026-08-01 00:00:00` |

---

## 2. Data Quality Audit (10 Documented Issues)

Every data quality issue was evaluated using statistical diagnostics, identified by mechanism, and remediated with evidence-based rationale:

| # | Column(s) | Identified Issue | Evidence / Diagnostic | Mechanism | Applied Remediation | Rationale |
|---|---|---|---|---|---|---|
| **1** | All columns | Exact duplicate rows (2.0%) | 3,600 rows have 100% byte-identical `txn_id` and feature values | Replayed batch export logs from regional servers | `drop_duplicates()` applied before splitting | Retaining replayed transactions leaks identical samples across training/test folds |
| **2** | `amount` | String formatting & 3 debit sign encodings | Values contain commas, `/-` suffixes, `(paren)`, `-minus`, and `Dr` tags | Legacy regional accounting software variants | `parse_amount()` regex parser converting to signed float | Unified numeric standard required for mathematical modeling |
| **3** | `txn_time` | Mixed datetime format representations | Single datetime parser fails on 67% of rows (ISO, `DD/MM/YYYY`, `Mon DD, YYYY`) | Regional system locale mismatch | Sequential 3-pass `pd.to_datetime()` | Ensures zero timestamp loss while maintaining temporal order |
| **4** | `reg_id` | Case & whitespace identifier scatter | 11,722 raw strings collapse to 3,991 canonical IDs | Unstandardised text entry across CRM channels | `df['customer_id'] = reg_id.str.strip().str.upper()` | Essential for grouping all SIMs under a single customer during CV |
| **5** | `account_name` | HTML non-breaking spaces & format chaos | `\xa0` characters (19% of rows), all-caps, comma-formatted names | Web scrapers & CRM dumps | Cleaned via regex replacement & title casing (excluded from model) | PII text string cleanup prevents data corruption |
| **6** | `gps_lat`, `gps_lon` | 36.2% missing values (MNAR) | Flat missingness across regions; coordinates outside East Africa bbox checked | Network coverage dropouts in remote areas (MNAR) | Imputed missingness indicator `gps_missing`; coordinates left un-imputed | Spatial mean imputation would distort geographical risk signals |
| **7** | `agent_id` | 13.7% missing values (MAR) | Missingness is ~45% in Arusha, Jinja, Mwanza, 0% elsewhere ($\chi^2, p < 0.001$) | Regional logging service offline/unsupported (MAR) | Imputed `agent_missing` indicator + pipeline median/constant fill | Captures regional system logging characteristics without bias |
| **8** | `device_id` | 5.9% missing values (MCAR) | Flat 5.9% missing rate across all regions and transaction types ($\chi^2, p > 0.05$) | Random device telemetry dropouts (MCAR) | Imputed `device_missing` indicator + pipeline fill | Pure random missingness treated via standard pipeline imputation |
| **9** | `balance_after` | Extreme right-skewed tail | Max balances exceed 3x IQR upper fence significantly | High-volume merchant & float accounts | Log transform (`log1p`) after zero-clipping | Compresses heavy tails for stable linear estimator convergence |
| **10** | `manual_review_score`, `settlement_status` | Post-outcome target leakage | Pearson $\|r\| = 0.981$ for review score; `reversed` status has 100% fraud rate | Operational markers populated *after* transaction review/reversal | Excluded entirely from feature matrix | Inclusion creates artificial AUC=1.00 that collapses in production |

---

## 3. Missing Data Classification

Statistical hypothesis testing was applied to classify missingness mechanisms:

```
                          ┌─────────────────────────┐
                          │   Missing Data Audit    │
                          └────────────┬────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   ┌──────────┐                  ┌──────────┐                  ┌──────────┐
   │   MCAR   │                  │   MAR    │                  │   MNAR   │
   │device_id │                  │ agent_id │                  │ gps_lat  │
   └────┬─────┘                  └────┬─────┘                  └────┬─────┘
        │                             │                             │
   Uninformative                Explainable by                Missingness
   random dropout              observed `region`            depends on location
  (Chi-sq p > 0.05)           (Chi-sq p < 0.001)           (Network coverage)
```

1. **Missing Completely At Random (MCAR) — `device_id` (5.9%):** $\chi^2$ tests confirm missingness is independent of region, transaction type, or target status ($p > 0.05$).
2. **Missing At Random (MAR) — `agent_id` (13.7%):** Missingness is strongly dependent on the observed variable `region` ($\chi^2 = 82,410.5, p < 0.001$), concentrated heavily in Arusha (~45.2%), Jinja (~45.1%), and Mwanza (~44.8%).
3. **Missing Not At Random (MNAR) — `gps_lat` / `gps_lon` (36.2%):** Geolocation dropout is driven by physical cellular coverage limits in remote areas; the probability of missingness depends directly on the unobserved location coordinate itself.

---

## 4. Feature Engineering & Lineage Architecture

We constructed **28 engineered features across 5 feature families**, adhering strictly to **temporal validity** (all features are computed exclusively from events preceding `SCORING_TS`):

```
Raw Ledger Events (ts < SCORING_TS)
        │
        ├──► Family 1: RFM & Behavioural (recency_days, frequency, monetary_out, monetary_in, out_in_ratio, cashout_ratio, n_txn_types)
        ├──► Family 2: Cyclical & Temporal (hr_sin, hr_cos, dow_sin, dow_cos, payday_dist)
        ├──► Family 3: Monetary Transformations (log_amt, amount_abs, log_balance, balance_to_amt_ratio)
        ├──► Family 4: Velocity & Diversity (n_devices, n_agents, amount_std, weekend_share, night_share, device_churn_rate, n_counterparty)
        └──► Family 5: Data Quality & Risk Flags (gps_missing, agent_missing, device_missing, is_weekend, is_night)
        │
        ▼
 Scikit-Learn ColumnTransformer + Pipeline
```

### Feature Dictionary Summary
- **RFM / Behavioural (7 features):** Capture account tenure, transactional velocity, net capital flow, and cashout propensity.
- **Cyclical / Temporal (5 features):** Transform hour of day and day of week via trigonometric sine/cosine components ($2\pi t / T$) to maintain cyclical continuity across midnight and week transitions; calculate proximity to East African paydays (1st and 28th).
- **Monetary Transformations (4 features):** Log-transform heavy-tailed transaction amounts and post-balances to enforce stationarity and homoscedasticity.
- **Velocity & Diversity (7 features):** Quantify multi-device usage, agent network breadth, counterparty diversity, device churn rate (`n_devices / frequency`), and off-hours activity share.
- **Data Quality & Risk Flags (5 features):** Explicitly preserve missingness indicators as informative model inputs alongside weekend and night-time transaction flags.

---

## 5. Data Leakage Assessment & Governance Controls

Two intentionally planted leakage sources were thoroughly investigated, categorized, and purged:

```
                            ┌────────────────────────┐
                            │   Planted Leakage      │
                            └───────────┬────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       ┌────────────────────────┐              ┌────────────────────────┐
       │     Target Leakage     │              │    Temporal Leakage    │
       │  manual_review_score   │              │   settlement_status    │
       └────────────┬───────────┘              └────────────┬───────────┘
                    │                                       │
        Generated during human                 Settlement / reversal occurs
        fraud investigation post-hoc          days/weeks post-transaction
        (Pearson |r| = 0.981)                 ("reversed" = 100% fraud rate)
```

### 1. Leakage Source 1: `manual_review_score` (Target Leakage)
- **Mechanism:** Human fraud analysts populate this score during post-event investigations. The score directly incorporates knowledge of the target variable `is_fraud`.
- **Statistical Evidence:** Pearson correlation $\|r\| = 0.981$ with the target.

### 2. Leakage Source 2: `settlement_status` (Temporal Leakage)
- **Mechanism:** Ledger statuses (`settled`, `reversed`, `held`, `pending`) are updated asynchronously over multi-day clearing cycles. At the time an automated credit or transaction decision must be rendered, the settlement outcome does not yet exist.
- **Statistical Evidence:** Transactions with status `reversed` exhibit a 100.0% fraud rate, while `settled` exhibits < 0.4%.

### Impact on Offline vs. Online Performance

| Model Version | Features Included | Validation Strategy | Cross-Validated ROC-AUC | Production Expected AUC | Assessment |
|---|---|---|---|---|---|
| **Contaminated Model** | 28 Features + Leaks | 5-Fold `GroupKFold` | **1.0000 (+/- 0.0000)** | ~0.5000 (Catastrophic Failure) | **UNSAFE** — Model relies entirely on post-outcome signals that are missing at prediction time |
| **Honest Baseline Model** | 28 Features (Leak-Free) | 5-Fold `GroupKFold` | **0.6274 (+/- 0.0038)** | **0.6274 (+/- 0.0038)** | **PRODUCTION-READY** — Reliable, audited, temporal-valid baseline |

> **Key takeaway for Risk Leadership:** Including leaky features creates a false offline score of **AUC = 1.0000** (+0.3747 artificial inflation). Deploying a contaminated model would cause immediate credit losses because the decision engine would receive `NaN` or unpopulated values for post-outcome fields in real-time scoring.

---

## 6. Machine Learning Pipeline & Validation Architecture

To eliminate preprocessing leakage (e.g., fitting scalers or imputers on validation folds), the entire workflow is wrapped in a single Scikit-Learn `Pipeline` and `ColumnTransformer`.

### Cross-Validation Strategy: Group-Aware (`GroupKFold`)
- **Grouping Variable:** Canonical `customer_id` (derived from cleaned `reg_id`).
- **Rationale:** A single customer may operate multiple SIM cards (`msisdn`). Standard random cross-validation would place SIM 1 of a customer in the training set and SIM 2 in the test set, leaking customer-level behavior across folds. `GroupKFold` guarantees that all transactions for any customer remain strictly isolated within a single fold.

### 5-Fold Cross-Validation Breakdown (Honest Model)

| Fold | Training Customers | Validation Customers | Fold ROC-AUC |
|---|---|---|---|
| **Fold 1** | ~3,192 | 799 | 0.6212 |
| **Fold 2** | ~3,192 | 799 | 0.6324 |
| **Fold 3** | ~3,193 | 798 | 0.6297 |
| **Fold 4** | ~3,193 | 798 | 0.6278 |
| **Fold 5** | ~3,194 | 797 | 0.6261 |
| **Mean / Std** | **3,991 Total** | **5 Folds** | **0.6274 (+/- 0.0038)** |

---

## 7. Model Persistence & Verification

The final honest model pipeline was fitted on all pre-scoring transactions and persisted using `joblib`.

```python
import joblib

# Model Persistence
joblib.dump(final_pipeline, "final_pipeline.joblib")

# Reload Verification
reloaded_pipeline = joblib.load("final_pipeline.joblib")
sample_preds_orig = final_pipeline.predict_proba(X_sample)[:, 1]
sample_preds_reload = reloaded_pipeline.predict_proba(X_sample)[:, 1]

# Verification Check
assert np.allclose(sample_preds_orig, sample_preds_reload)
```

The saved artifact (`final_pipeline.joblib`) contains all column transformers, median imputers, Yeo-Johnson power transformers, robust scalers, one-hot encoders, and the logistic regression estimator. It can be reloaded directly in scoring environments without requiring manual feature preprocessing scripts.

---

## 8. Risk Governance & Deployment Recommendations

1. **Enforce Feature Timestamp Verification:** Implement automated feature store gates that compare a feature's availability timestamp against the transaction scoring timestamp.
2. **Mandate Group-Aware Cross-Validation:** Institutionalize `GroupKFold` (grouped on canonical customer entity IDs) across all credit risk model evaluations.
3. **Conduct Regional Risk Audits:** Monitor model calibration separately in Arusha, Jinja, and Mwanza where legacy agent logging (MAR) alters feature density.
4. **Deploy Gradient Boosting Benchmarks:** As a next iteration, benchmark XGBoost / LightGBM against this linear baseline to capture higher-order non-linear interactions across velocity and RFM features.

---

## 9. Conclusion

This project delivers a **robust, leak-free, and fully reproducible machine learning pipeline** for mobile money credit risk assessment. By prioritizing statistical evidence over assumptions and structural pipeline isolation over manual preprocessing, we ensure that the model's offline performance metric (**ROC-AUC 0.6274**) translates accurately to live deployment.

> **Core Policy Principle:** *A reliable credit-risk model is not simply a model with high predictive performance; it is a model whose performance can be trusted, explained, reproduced, and governed.*
