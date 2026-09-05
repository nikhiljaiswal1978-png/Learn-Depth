# Loan Approval Screening — Short Report

## 1. Problem

The task is to build a **binary classification** model that predicts `target` (1 = positive/risk/event class, 0 = negative/non-event class) for loan applicants, using six numeric predictors: `income_monthly`, `credit_score`, `debt_to_income`, `employment_years`, `loan_amount`, and `prior_defaults`. The required algorithm is **Logistic Regression**, evaluated against a simple baseline.

## 2. Data

The dataset contains **1,000 rows** and 6 predictors + target, all numeric (no categorical fields). Key checks:

| Check | Result |
|---|---|
| Missing values | None in any column |
| Duplicate rows | None |
| Class balance | Perfectly balanced — 500 / 500 (50% / 50%) |
| Negative or out-of-range values | None found |

`income_monthly` and `loan_amount` are right-skewed (max values several times the mean/median), typical of financial data. `debt_to_income` spans a range (~11 to ~1095) inconsistent with a conventional 0–1 or 0–100% ratio; it is used as-is as a continuous numeric feature, with the units caveat noted in Limitations.

Correlations with `target` (Pearson):

| Feature | Correlation with target |
|---|---|
| income_monthly | +0.238 |
| loan_amount | +0.215 |
| debt_to_income | +0.182 |
| employment_years | −0.193 |
| prior_defaults | −0.199 |
| credit_score | −0.211 |

No pair of predictors showed severe multicollinearity, so all six were retained.

![Feature distributions](fig_distributions.png)
![Correlation matrix](fig_corr.png)

## 3. Preprocessing

- No imputation needed (no missing values).
- No deduplication needed (no duplicates).
- No categorical encoding needed (all predictors numeric).
- **Standardization**: all six features were standardized (z-score) using `StandardScaler`. This aids solver convergence and makes coefficients comparable in "per 1 standard deviation" units.
- **Leakage control**: the train/test split was performed *first*; the scaler was `fit` only on the training partition and then applied (`transform`) to the test partition, so no test-set information influenced preprocessing statistics.

## 4. Train/Test Split

An **80/20 split**, stratified on `target`, with `random_state=42` for reproducibility:

- Train: 800 rows (50% / 50% class balance preserved)
- Test: 200 rows (50% / 50% class balance preserved)

## 5. Model

**Logistic Regression** (scikit-learn), configuration:
- `solver="lbfgs"`
- Default ridge-style (L2) regularization, `C=1.0` (no hyperparameter search performed — appropriate for this baseline-scope task)
- `max_iter=1000`
- `random_state=42`
- Trained on the standardized training features only

A **stratified Dummy Classifier** was used as the simple baseline, justified by the exactly-balanced classes (it gives a true ~50% reference point to confirm the Logistic Regression model is learning real signal).

## 6. Results

| Metric | Logistic Regression | Baseline (Dummy) |
|---|---|---|
| Accuracy | **0.705** | 0.500 |
| Precision | **0.688** | 0.500 |
| Recall | **0.750** | 0.510 |
| F1-score | **0.718** | 0.505 |
| ROC-AUC | **0.782** | 0.500 |

The model clearly outperforms the baseline across every metric, confirming genuine predictive signal in the six features.

**Confusion matrix** (test set, threshold = 0.5):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 66 (TN) | 34 (FP) |
| **Actual 1** | 25 (FN) | 75 (TP) |

![Confusion matrix](fig_confusion_matrix.png)
![ROC curve](fig_roc_curve.png)

**Interpretation:**
- **False Positives (34):** applicants incorrectly predicted as class 1 — i.e., wrongly flagged into the positive/event/risk group.
- **False Negatives (25):** applicants incorrectly predicted as class 0 — i.e., a true positive/event case was missed. If `target=1` represents credit risk, this is typically the costlier error in a lending context (a risky applicant slips through as "safe").
- Recall (0.75) exceeds precision (0.688), meaning the model is somewhat more inclined to catch actual positives at the cost of some false alarms — a reasonable default posture for a risk-screening tool, though the threshold should ultimately be tuned to the institution's real cost trade-off rather than left at 0.5.

## 7. Coefficient Interpretation

Because features are standardized, each coefficient is a **log-odds change per 1 standard deviation** increase in that feature; `exp(coefficient)` is the corresponding **odds ratio**.

| Feature | Coefficient (log-odds / 1 SD) | Odds Ratio (per 1 SD) |
|---|---|---|
| loan_amount | +0.594 | 1.81 |
| income_monthly | +0.585 | 1.80 |
| debt_to_income | +0.434 | 1.54 |
| prior_defaults | −0.516 | 0.60 |
| credit_score | −0.525 | 0.59 |
| employment_years | −0.609 | 0.54 |

![Coefficients](fig_coefficients.png)

**Reading this:** a one-SD increase in `loan_amount` is associated with roughly an 81% increase in the odds of `target=1`, holding other features constant; a one-SD increase in `employment_years` is associated with roughly a 46% decrease in those odds, and so on for each feature.

**Important caveat:** in a typical credit-risk setting we would expect `prior_defaults` and `credit_score` to be strong *positive* risk indicators (more defaults / lower credit score → higher risk). Here both are negatively associated with `target=1`, which only makes intuitive sense if `target=1` denotes "approved / favorable outcome" rather than "high risk." This directionality likely reflects how the dataset was synthetically generated rather than a realistic underwriting process, and the true business meaning of `target=1` should be confirmed with domain stakeholders before acting on these coefficients.

## 8. Error Analysis

59 of 200 test cases (29.5%) were misclassified. About half of these had predicted probabilities within 0.15 of the 0.5 decision boundary (median distance ≈0.14) — consistent with genuinely ambiguous cases. The remainder were misclassified with higher confidence, indicating the linear decision boundary does not fully separate the classes; this is consistent with the visible overlap in per-feature distributions between the two classes.

## 9. Limitations

- Logistic Regression assumes a **linear log-odds relationship**; non-linear effects or interactions (e.g., risk rising sharply past a debt-to-income threshold) are not captured.
- The counter-intuitive coefficient directions for `credit_score` and `prior_defaults` suggest this may be a synthetic/stylized dataset, limiting how far conclusions transfer to real lending data.
- No cross-validation was performed — metrics reflect a single 80/20 split.
- Hyperparameters (`C`) and the decision threshold were left at defaults, not tuned against a validation set or real business costs.
- `debt_to_income`'s scale doesn't match a conventional ratio; its exact real-world units should be confirmed upstream.

## 10. Possible Improvements

- Stratified k-fold cross-validation for more stable metric estimates.
- A modest hyperparameter search over `C` / penalty type, selected via cross-validated ROC-AUC.
- Feature engineering (e.g., a loan-to-income ratio) and/or a non-linear model (Random Forest, Gradient Boosting) as a comparison-only benchmark to check whether linearity is limiting performance.
- Threshold tuning based on the real relative cost of false positives vs. false negatives.
- Validation of feature definitions/units against the original data source.

## 11. Conclusion

The Logistic Regression model — trained with leakage-safe preprocessing on a stratified 80/20 split — achieves **70.5% accuracy, 0.688 precision, 0.750 recall, 0.718 F1, and 0.782 ROC-AUC**, clearly beating a balanced baseline. It is a transparent, auditable first-pass screening model whose coefficients can be explained directly in terms of standard-deviation effects and odds ratios. However, the unexpected coefficient directions for `credit_score` and `prior_defaults`, the single-split evaluation, and the untuned threshold mean it should be treated as a **baseline / proof-of-concept**, not a production lending decision system, without further validation against real-world data and business-defined costs.
