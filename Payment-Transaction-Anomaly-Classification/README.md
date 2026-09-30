# Payment Transaction Anomaly Classification

## Capstone — FinTech

An educational machine-learning prototype that classifies payment transactions as **Routine (0)** or **Anomalous/Fraud-like (1)**.

### Dataset
- 284,807 original records
- 214 duplicate rows removed → **284,593 rows used for modelling**
- 492 anomalous/fraud-like records
- 6 behavioural/numeric predictors
- No missing values in the supplied dataset

### Features
1. `transaction_amount`
2. `txn_hour`
3. `txn_frequency_24h`
4. `merchant_category_risk`
5. `account_age_days`
6. `avg_txn_amount_30d`

Target: `is_fraud` (0 = routine, 1 = anomalous/fraud-like).

## Methodology
1. Load and inspect the supplied dataset.
2. Check missing values, duplicates, class imbalance and data types.
3. Remove duplicate rows.
4. Explore distributions, outliers, feature/target relationships and correlations.
5. Apply `log1p` to the two skewed amount features for Logistic Regression/KNN.
6. Use a **75/25 stratified train/test split**, `random_state=42`.
7. Compare three foundational models:
   - Logistic Regression (`class_weight='balanced'`)
   - KNN (`k=7`)
   - Decision Tree (`class_weight='balanced'`, `max_depth=6`)
8. Evaluate accuracy, precision, recall, F1, ROC-AUC, confusion matrices and ROC curves.
9. Perform feature-importance and false-negative/error analysis.
10. Deploy the documented final prototype model with the exact same preprocessing used during training.

## Final prototype model
**Logistic Regression** is used for the Streamlit prototype because recall is especially important for anomaly screening and Logistic Regression achieved the highest recall among the three tested foundational models on the held-out test set. It also provides an interpretable baseline.

### Held-out test results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.36% | 1.36% | 84.55% | 2.67% | 90.49% |
| KNN (k=7) | 99.83% | 61.90% | 10.57% | 18.06% | 69.61% |
| Decision Tree | 90.81% | 1.50% | 80.49% | 2.94% | 87.48% |

Logistic Regression confusion matrix: **TN=63,474; FP=7,552; FN=19; TP=104**.

The very low precision is an important finding: because the positive class is extremely rare, a recall-oriented classifier can flag many routine transactions. The prototype should therefore be treated as a screening/flagging demonstration, not an automatic financial decision system.

## Streamlit
Run from the project root:

```bash
py -m streamlit run app/app.py
```

The application loads `models/anomaly_model.pkl`, which contains the complete preprocessing pipeline and final Logistic Regression model. This prevents training-time and inference-time preprocessing from diverging.

## Project structure

```text
Payment-Transaction-Anomaly-Classification/
├── app/
│   └── app.py
├── data/
│   ├── transactions.csv
│   └── SOURCE.txt
├── models/
│   ├── anomaly_model.pkl
│   └── model_summary.json
├── notebooks/
│   └── capstone_notebook.ipynb
├── outputs/
├── README.md
├── requirements.txt
├── technical_paper.md
└── presentation.md
```

## Important limitation
This is an educational capstone prototype. It does not establish that a transaction is actually fraudulent and should not be used as an autonomous production financial decision system.
