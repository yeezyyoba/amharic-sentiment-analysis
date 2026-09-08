# Ethiopian Fintech Analytics Platform
## Final Project Report

**Author**: Eyob Nebyou  
**Institution**: Addis Ababa University, CNCS  
**Date**: August 2026  
**GitHub**: https://github.com/yeezyyoba/ethiopian-fintech-analytics

---

## 1. Executive Summary

This project builds an end-to-end credit risk prediction system for the Ethiopian
digital finance ecosystem. Using 150,000 real customer records, we developed a
machine learning pipeline that predicts loan default probability with an AUC-ROC
of 0.8842 and a Recall of 0.8284 — meaning the model correctly identifies 82.8%
of all actual defaulters.

The system combines rigorous data science methodology with business-facing
deliverables: a SHAP explainability analysis, an interactive Streamlit dashboard,
and a REST API for real-time inference.

---

## 2. Problem Statement

Digital financial services in Ethiopia — mobile banking, microloans, mobile money —
are growing rapidly. Two critical challenges face these platforms:

1. **Credit Risk**: How likely is a customer to default on a loan?
2. **Fraud Detection**: Is a transaction fraudulent?

Without predictive models, loan officers rely on manual assessment — slow,
inconsistent, and unable to scale. This project automates that assessment.

---

## 3. Dataset

| Property | Detail |
|---|---|
| Name | Give Me Some Credit |
| Source | Kaggle |
| Rows | 150,000 customers |
| Features | 11 raw features |
| Target | SeriousDlqin2yrs (binary default label) |
| Default rate | 6.68% |
| Class imbalance | 14:1 (non-default : default) |

---

## 4. Methodology

### 4.1 Exploratory Data Analysis
- Identified 14:1 class imbalance requiring SMOTE
- Found `RevolvingUtilizationOfUnsecuredLines` as top correlated feature (0.278)
- Confirmed statistical significance with t-tests (p=0.0000 for age and income)
- Identified under-25s as highest default risk group (11.73% rate)

### 4.2 Feature Engineering
Built 8 domain-specific features:

| Feature | Description |
|---|---|
| `delinquency_score` | Weighted sum of past-due events (90+ days × 3) |
| `debt_to_income` | Monthly debt payment relative to income |
| `is_young_borrower` | Flag for age < 30 |
| `is_senior_borrower` | Flag for age ≥ 60 |
| `high_utilization` | Flag for credit utilization > 80% |
| `total_past_due` | Sum of all delinquency counts |
| `has_delinquency` | Binary flag for any delinquency |
| `has_dependents` | Binary flag for dependents > 0 |

**Key finding**: `has_delinquency` (correlation 0.3144) became the strongest
predictor — outperforming the original `RevolvingUtilizationOfUnsecuredLines`.

### 4.3 Class Imbalance — SMOTE
Applied SMOTE (Synthetic Minority Oversampling Technique) to fix the 14:1
imbalance. Dataset grew from 149,954 to 279,862 samples (1:1 balance).
Training used SMOTE-balanced data; evaluation used the original distribution.

### 4.4 SQL Analysis
Built an in-memory SQLite database to explore business insights:
- Combined delinquency + high utilization → 36.82% default rate
- Default rate by age group confirmed EDA findings

---

## 5. Model Results

| Model | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.8500 | 0.3160 | 0.1990 | 0.7677 |
| Random Forest | 0.8714 | 0.4356 | 0.3949 | 0.4855 |
| LightGBM | 0.8727 | 0.3692 | 0.5710 | 0.2728 |
| **XGBoost** | **0.8842** | 0.3326 | 0.2081 | **0.8284** |

**Best model: XGBoost**

XGBoost achieved the highest AUC-ROC (0.8842) and the best Recall (0.8284).
In credit risk, missing a defaulter (false negative) costs more than a false
alarm — so Recall is the priority metric. Only 1,720 defaulters were missed
out of 10,023.

---

## 6. Model Explainability — SHAP

SHAP (SHapley Additive exPlanations) was used to explain why the model
makes each prediction.

**Global feature importance (SHAP ranking):**
1. `has_delinquency` — SHAP value 1.4215 (dominant predictor)
2. `high_utilization` — SHAP value 0.5738
3. `delinquency_score`
4. `RevolvingUtilizationOfUnsecuredLines`
5. `age`

**Business translation — three questions the model asks:**
1. *Has this customer missed payments before?* (`has_delinquency`, `delinquency_score`)
2. *Are they overextended on credit?* (`high_utilization`, `RevolvingUtilization`)
3. *Do they have capacity to repay?* (`MonthlyIncome`, `debt_to_income`, `age`)

This matches exactly what experienced loan officers assess manually — the model
has learned real credit risk intuition from data.

---

## 7. Deployment

### Streamlit Dashboard
Interactive dashboard with:
- Credit risk overview and KPI cards
- Default rate by customer segment
- SHAP feature importance visualization
- Individual customer risk scoring

### FastAPI REST Endpoint
```
POST /predict
Input:  Customer financial features (JSON)
Output: Default probability, risk tier, top risk factors
```

---

## 8. Limitations

- Dataset is US-based (Give Me Some Credit) — Ethiopian-specific data would
  improve relevance
- Model evaluated on historical data — performance on future data may differ
- Label noise: some neutral labels may be misclassified
- Fairness analysis shows higher false positive rates for young borrowers

---

## 9. Conclusion

This project demonstrates a complete, production-ready credit risk pipeline —
from raw data through feature engineering, model training, explainability, and
deployment. The XGBoost model achieves AUC-ROC of 0.8842 with 82.8% recall
on defaulters, and SHAP analysis reveals that past delinquency history is by
far the most powerful signal.

The methodology, codebase, and findings are directly applicable to Ethiopian
fintech platforms seeking to automate credit decisioning at scale.

---

## 10. References

- Kaggle: Give Me Some Credit Dataset
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions (SHAP)
- Chawla, N. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
