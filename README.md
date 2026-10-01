# Loan default prediction

A default-risk model for 20,000 loans, with calibrated probabilities, portfolio loss estimates (EL/CECL), a RAROC pricing optimizer and a fair-lending audit built on top of it.

The full analysis is in `loan_default_eda_model.ipynb`. This file summarizes it.

## About this project

This is a personal project built from what was learned during a Summer 2025 internship in the credit field at Navy Federal Credit Union. The credit-risk framing (PD, LGD, EAD, expected loss, CECL, RAROC pricing, fair-lending screening and model risk management) comes from that experience and from public regulatory guidance.

It is not affiliated with, endorsed by, or representative of Navy Federal Credit Union. No Navy Federal data, models, systems or internal methods were used. All assumptions (LGD by loan purpose, cost of funds, opex, capital multiplier) are illustrative.

## Dataset credit

The data is the [Loan Prediction Dataset 2025](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025/data) on Kaggle, published by nabihazahid under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The data is not included in this repository. Download it from the Kaggle page and save it as `loan_dataset_20000.csv` in the project root before running the notebooks.

## License

The code, notebooks, models and write-up are released under the [MIT License](LICENSE). The dataset is not part of this repository and keeps its original Apache 2.0 license, as described above.

## Contents

- [About this project](#about-this-project)
- [Dataset credit](#dataset-credit)
- [License](#license)
- [Project layout](#project-layout)
- [Running the notebooks](#running-the-notebooks)
- [Using the saved models](#using-the-saved-models)
- [Data and target](#data-and-target)
- [EDA findings](#eda-findings)
- [Modeling approach](#modeling-approach)
- [Results](#results)
- [Monotonic constraints](#monotonic-constraints)
- [Cross-validation](#cross-validation)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Probability calibration](#probability-calibration)
- [Portfolio loss: EL and CECL](#portfolio-loss-el-and-cecl)
- [Feature importance](#feature-importance)
- [Pricing optimizer](#pricing-optimizer)
- [The grade_subgrade question](#the-grade_subgrade-question)
- [Fair lending audit](#fair-lending-audit)
- [Model risk management](#model-risk-management)

## Project layout

```
loan_default_eda_model.ipynb   EDA and training notebook (executed, plots and tables embedded)
loan_dataset_20000.csv         Source data, not tracked in git (download from Kaggle, see Dataset credit)
cleaning_system/               Outlier and anomaly detection used to vet the data before modeling
                               (outlier_detector.py, detection.ipynb, config.yaml, anomaly_report.txt;
                               flagged_output.csv is generated and not tracked in git)
models/                        Saved models and metadata
requirements.txt               Package versions the models were fit under
model_explanation.txt          Plain-text version of this write-up
```

## Running the notebooks

1. Install the packages in `requirements.txt`.
2. Download the dataset from Kaggle (see [Dataset credit](#dataset-credit)) and save it as `loan_dataset_20000.csv` in the project root.
3. Run `loan_default_eda_model.ipynb` from the project root. It reads `loan_dataset_20000.csv` and writes the saved models to `models/`.
4. Optional: run the outlier check from inside `cleaning_system/`, either as a script or through the notebook. Both read the CSV and `config.yaml`, and regenerate `flagged_output.csv` and `anomaly_report.txt`.

```
cd cleaning_system
python outlier_detector.py ../loan_dataset_20000.csv config.yaml
```

`detection.ipynb` holds the same code, with the paths set to `../loan_dataset_20000.csv` and `config.yaml`.

The notebooks use relative paths, so start Jupyter from the folders above or the files will not be found.

## Using the saved models

Both models are full scikit-learn pipelines (preprocessing, monotonic XGBoost, isotonic calibration) saved with joblib.

| File | Includes `grade_subgrade` | Use |
|---|---|---|
| `models/xgboost_monotonic_calibrated.joblib` | yes | Scoring or explaining loans that are already graded or approved |
| `models/xgboost_monotonic_calibrated_preunderwriting.joblib` | no | Pre-approval and application-time scoring |

`models/metadata.json` lists the feature columns, use case, and test-set ROC-AUC and Brier score for each model.

```python
import joblib
import pandas as pd

model = joblib.load("models/xgboost_monotonic_calibrated_preunderwriting.joblib")
applicants = pd.DataFrame([...])            # columns listed in models/metadata.json
pd_default = model.predict_proba(applicants)[:, 1]
```

The versions in `requirements.txt` are needed to load the files without version mismatch errors (pandas 2.3.2, numpy 2.3.2, scikit-learn 1.7.1, xgboost 3.0.5, fairlearn 0.14.0, matplotlib 3.10.6, seaborn 0.13.2, scipy 1.16.1, joblib 1.5.2). Reloading the main model in a fresh Python process reproduces the notebook's PD for a sample applicant exactly.

## Data and target

The raw target is `loan_paid_back` (1 = paid back, 0 = defaulted). The notebook flips it to `default = 1 - loan_paid_back`, so 1 always means the bad outcome, and every model predicts that label.

The class split is 80% paid back and 20% defaulted. That is real, not a data error, so it is handled with class weighting.

Nothing is missing, so no imputation was needed.

Numeric features (15): `age`, `annual_income`, `monthly_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`, `loan_term`, `installment`, `num_of_open_accounts`, `total_credit_limit`, `current_balance`, `delinquency_history`, `public_records`, `num_of_delinquencies`

Categorical features (6): `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`

## EDA findings

- `employment_status` is the strongest single driver. Unemployed borrowers default 82% of the time, students 59%, self-employed and employed about 11%, retired about 0.5%.
- `grade_subgrade` is a lender-assigned credit grade (A1 best, F5 worst). It lines up almost perfectly with default rate: A grades default around 4-8%, F grades around 30-35%.
- Among the numeric fields, `debt_to_income_ratio` (correlation +0.22) and `credit_score` (-0.20) are the strongest on their own. Income, loan amount, credit limit and balance have almost no linear correlation with default.
- The outlier pass in `cleaning_system/` flagged 1,455 rows (7.3%) on statistical grounds, mostly from the skewed income and balance columns. It found no domain-rule violations and no cross-field inconsistencies. The flagged rows are unusual but not broken, so none were dropped.

## Modeling approach

The split is 80% train and 20% test, stratified on the target so both keep the 80/20 class ratio.

Preprocessing is a single scikit-learn `ColumnTransformer`, fit on the training data only so nothing leaks from the test set:

- Numeric columns go through `StandardScaler`.
- Categorical columns go through `OneHotEncoder`. The first category is dropped to avoid redundant dummies, and unknown categories at test time are ignored.

Three models were trained, each as a `Pipeline` of preprocessing plus classifier:

1. Logistic regression. A linear baseline with `class_weight="balanced"`. It is the easiest to read, since the coefficients show the direction and size of each effect.
2. Random forest. 400 trees, `max_depth=8`, `min_samples_leaf=20`, `class_weight="balanced"`. The shallow trees and leaf floor limit overfitting on 16k training rows.
3. XGBoost. 400 trees, `max_depth=4`, `learning_rate=0.05`, `subsample` and `colsample_bytree` at 0.8. `scale_pos_weight` is the majority/minority ratio in the training set, which is XGBoost's version of class weighting.

## Results

Held-out test set, 4,000 rows.

| Model | ROC-AUC | PR-AUC | Recall (default) |
|---|---|---|---|
| Logistic regression | 0.886 | 0.772 | 0.73 |
| Random forest | 0.882 | 0.769 | 0.69 |
| XGBoost | 0.892 | 0.787 | 0.70 |

ROC-AUC measures how well the model ranks defaulters above non-defaulters (0.5 is random, 1.0 is perfect). PR-AUC is more informative than ROC-AUC when classes are imbalanced. Recall is the share of actual defaulters caught at a 0.5 threshold.

All three catch roughly 70-73% of defaulters, at the cost of some false positives (good borrowers who get flagged). Moving the threshold trades recall against precision. A bank that prioritizes catching defaulters would push recall up, and one that wants to avoid turning away good borrowers would push precision up.

XGBoost scored highest on both ROC-AUC and PR-AUC, so it is the model used from here on.

## Monotonic constraints

None of the three models above has a monotonic constraint. A tree model can learn that risk dips at some high debt-to-income value purely from noise, even though finance says the relationship should run one way.

A fourth model, `xgboost_monotonic`, was trained with XGBoost's `monotone_constraints`. Each constrained numeric feature can now move predicted default probability in only one direction. The one-hot categorical columns are unordered, so they get constraint 0.

| Constraint | Features | Reasoning |
|---|---|---|
| +1 (raising it never lowers risk) | `debt_to_income_ratio` | less room to repay |
| | `loan_amount` | bigger exposure |
| | `interest_rate` | a higher rate reflects and adds to risk |
| | `installment` | heavier payment burden |
| | `delinquency_history`, `num_of_delinquencies` | more past delinquencies |
| | `public_records` | more adverse records |
| -1 (raising it never raises risk) | `credit_score` | a higher score means lower risk |
| | `annual_income`, `monthly_income` | more ability to repay |
| 0 (direction unclear) | `age` | risk is not monotonic across life stages |
| | `loan_term` | tangled up with pricing |
| | `num_of_open_accounts` | access on one side, exposure on the other |
| | `total_credit_limit` | cushion on one side, exposure on the other |
| | `current_balance` | ambiguous without scaling by limit or income |
| | all one-hot categorical columns | unordered |

To verify, each constrained feature was swept across its observed range with everything else held at the median or mode. Predicted default probability moved only in the allowed direction for all 10 constrained features. Accuracy did not suffer:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| XGBoost (unconstrained) | 0.8918 | 0.7865 |
| XGBoost (monotonic) | 0.8932 | 0.7891 |

The model now follows the financial rules by construction, not just as a tendency in the data, at no cost in accuracy. That matters for a model that would drive or explain real lending decisions, since an auditor can point to a guaranteed direction.

## Cross-validation

Every number above comes from one 80/20 split. To see how much it moves, 5-fold stratified cross-validation was run on the training set with the monotonic XGBoost pipeline. The test set stayed untouched.

| Metric | Mean +/- std | Folds |
|---|---|---|
| ROC-AUC | 0.8890 +/- 0.0046 | 0.8882, 0.8909, 0.8960, 0.8819, 0.8879 |
| PR-AUC | 0.7818 +/- 0.0091 | 0.7865, 0.7799, 0.7935, 0.7660, 0.7831 |

The held-out test result (ROC-AUC 0.8932, PR-AUC 0.7891) is within about one fold-to-fold standard deviation of the CV mean, so it is not a lucky split.

## Hyperparameter tuning

The XGBoost settings (`n_estimators=400`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`) were picked by hand. A `RandomizedSearchCV` (25 candidates, 3-fold CV, scored on ROC-AUC) checked whether they are near the best. The monotone constraints stayed fixed, since they are a domain requirement and not something to tune.

- Best CV ROC-AUC in the search: 0.8926
- Best params: `n_estimators=442`, `max_depth=5`, `learning_rate=0.019`, `subsample=0.855`, `colsample_bytree=0.732`, `min_child_weight=8`, `reg_lambda=3.78`

| Model | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| Tuned | 0.8963 | 0.7932 |
| Hand-picked | 0.8932 | 0.7891 |

Decision: keep the hand-picked hyperparameters. The tuned model is about 0.3 points better, less than the 0.46-point fold-to-fold spread, so the difference is within noise. Switching would mean redoing calibration, EL/CECL, pricing and the fair-lending audit for a gain that cannot be told apart from sampling variation. The tuned values are recorded here for anyone who wants to revisit this with a bigger search or more data.

## Probability calibration

Everything downstream (EL/CECL, pricing, approval decisions) uses the predicted probability directly as PD. That only works if the probabilities are calibrated: of all loans predicted at 30%, about 30% should default. XGBoost trained with `scale_pos_weight` gives this up, because reweighting the loss pushes predicted probabilities away from true frequencies.

A 10-bin calibration curve on the test set showed the raw model was fine at the extremes and badly overconfident in the middle:

| Mean predicted PD | Observed default rate | Gap |
|---|---|---|
| 0.507 | 0.190 | -0.317 |
| 0.683 | 0.347 | -0.335 |

The raw Brier score was 0.1131. The fix was `CalibratedClassifierCV` (isotonic, 5-fold) around the same monotonic XGBoost pipeline. Isotonic regression is a non-decreasing map, so it cannot change how applicants are ranked. Only the translation from raw score to probability changes.

| | Before | After |
|---|---|---|
| Brier score | 0.1131 | 0.0786 |
| ROC-AUC | 0.8932 | 0.8933 |
| Largest calibration gap | 33.5 points | about 1.5 points |

All later sections use this calibrated PD.

Caveat: isotonic regression fits a step function, so it can flatten a low-risk region to exactly PD = 0.0. About 4,146 of 20,000 loans (21%) land there. If smoother low-end probabilities matter more than the best calibration fit, Platt scaling (`method="sigmoid"`) is a less flexible alternative worth trying.

## Portfolio loss: EL and CECL

Expected loss: `EL = PD x LGD x EAD`

CECL (basic form): `ACL = Amortized Cost Basis x Estimated Lifetime Loss Rate`

| Term | Source |
|---|---|
| PD | Calibrated model output. `loan_paid_back` is a terminal outcome, not a rolling 12-month one, so it is treated as a lifetime PD, which is what CECL needs. |
| LGD | Not in the dataset, so it is an assumption proxied by `loan_purpose`: Home 25% (secured, real estate), Car 40% (secured, depreciating asset), all other purposes 65% (unsecured). These are illustrative benchmark severities, not derived from this data. |
| EAD / cost basis | `current_balance`, for both terms, since these are simple non-revolving installment loans. |
| Lifetime loss rate | PD x LGD |

Because EAD and cost basis are both `current_balance`, and PD is already lifetime, CECL reduces to the same calculation as EL. That is expected, not a bug.

Full portfolio, 20,000 loans, calibrated model:

| Measure | Value |
|---|---|
| Total outstanding balance | $486,667,892.63 |
| Expected loss (EL) | $57,557,149.10 |
| CECL allowance (ACL) | $57,557,149.10 |
| Reserve ratio (ACL / balance) | 11.83% |

The raw, uncalibrated model gave $96,257,066 (a 19.78% reserve ratio). Calibration cut the estimate by about 40%, because the raw model overestimated PD through the middle of the risk range. This is the clearest example of why calibration had to come before any dollar figure.

Out-of-sample check on the 4,000 test loans, which avoids the optimism of scoring training loans:

| Measure | Value |
|---|---|
| Test portfolio balance | $98,254,528.69 |
| Test EL / ACL | $11,507,727.64 |
| Test reserve ratio | 11.71% |

The full-portfolio ratio (11.83%) and the test-only ratio (11.71%) are close, so in-sample optimism looks small.

By `loan_purpose`, sorted by dollar EL:

| Purpose | Balance ($) | Avg PD | LGD | EL ($) | Reserve % |
|---|---|---|---|---|---|
| Debt consolidation | 193,868,700 | 0.201 | 0.65 | 25,240,150 | 13.02% |
| Other | 61,534,240 | 0.199 | 0.65 | 8,176,597 | 13.29% |
| Education | 40,313,050 | 0.215 | 0.65 | 5,877,585 | 14.58% |
| Business | 39,820,500 | 0.194 | 0.65 | 5,034,506 | 12.64% |
| Car | 58,492,900 | 0.201 | 0.40 | 4,881,721 | 8.35% |
| Medical | 28,230,240 | 0.215 | 0.65 | 4,015,735 | 14.22% |
| Home | 49,262,180 | 0.183 | 0.25 | 2,322,723 | 4.72% |
| Vacation | 15,146,120 | 0.217 | 0.65 | 2,008,129 | 13.26% |

The reserve ratio depends far more on the LGD assumption than on PD, which is fairly flat (18-22%) across purposes. Home has the lowest ratio despite an unremarkable average PD, purely because its assumed LGD is low.

Limitations:

- LGD is assumed, not fitted from recovery data, and it is the biggest remaining uncertainty in the $57.6M figure. If actual unsecured LGD were 50% instead of 65%, EL on the unsecured share would drop roughly in proportion.
- EAD is today's balance, not a projected balance at a future default date. For amortizing loans the true EAD at default is somewhat lower for loans further from origination. That is ignored to stay with the basic equation.
- PD is a static estimate fit on history. Full CECL practice adjusts for forward-looking macro scenarios, such as a recession overlay. This is a point-in-time model.
- About 21% of loans have PD of exactly 0.0 from isotonic calibration and contribute $0 to EL.

## Feature importance

Top features in the XGBoost model:

1. `employment_status` = Unemployed
2. `employment_status` = Student
3. `employment_status` = Retired
4. `debt_to_income_ratio`
5. `credit_score`
6. `interest_rate`
7. to 15. various `grade_subgrade` levels, `loan_purpose`, self-employed status

That matches the EDA: employment status, credit grade, DTI and credit score dominate, which is typical for credit risk.

## Pricing optimizer

Given an applicant's PD, the optimizer recommends the `interest_rate` that makes that loan hit a target risk-adjusted return on capital (RAROC), instead of pricing from `grade_subgrade` lookup tables.

```
RAROC = (Interest Income - Cost of Funds - Opex - Expected Loss) / Economic Capital
```

Per-loan terms, where `r` is the rate being solved for:

| Term | Formula |
|---|---|
| Interest income | `r * EAD` |
| Cost of funds | `cof_rate * EAD` (assumed 3.0%) |
| Opex | `opex_rate * EAD` (assumed 1.5%) |
| Expected loss | `PD * LGD * EAD` |
| Economic capital | `k * LGD * EAD * sqrt(PD * (1 - PD))` (default `k = 1`) |

Economic capital is the standard deviation of a single Bernoulli default event, scaled by a capital multiplier `k`. PD comes from the calibrated model and LGD uses the same `loan_purpose` assumption as the EL section. EAD is `loan_amount` here, because pricing happens at origination before any balance has amortized.

Interest income is the only term that depends on `r`, and every other term is linear in EAD, so setting RAROC equal to the target has a closed-form answer:

```
r* = cof_rate + opex_rate + PD*LGD + target_RAROC * k * LGD * sqrt(PD * (1 - PD))
```

That is cost of funds, operating cost, the expected-loss rate, and a capital charge that grows with the target return and the loan's own risk. Recommended rates are clipped at 60%, the `interest_rate` bound in `cleaning_system/config.yaml`.

Recomputing achieved RAROC at each unclipped, interior-PD recommended rate reproduces the target to 1e-9. Loans with PD of exactly 0 or 1 (about 4,146 of 20,000) are priced at the 4.5% cost-of-funds plus opex floor, since their economic capital is 0.

Results at a 15% target RAROC, full portfolio, calibrated PD:

| | Current | Recommended |
|---|---|---|
| Mean interest rate | 12.40% | 17.54% |
| Median interest rate | 12.40% | 13.25% |
| Min | 3.14% | 4.50% |
| Max | 22.51% | 60.00% (clipped) |

- Priced below the target-RAROC rate (underpriced): 10,698 loans (53.5%)
- Priced above it (overpriced): 9,302 loans (46.5%)

With the raw PD, the mean recommended rate was 26.1% and 71.1% of loans looked underpriced. Calibration brought the recommendations much closer to the current book. The average gap between recommended and current rate still rises with credit grade, from about -2.1 points for A1 to about +10.6 points for F1, because higher PD pushes both the expected-loss and capital-charge terms up.

Worked example (PD = 0.20%, LGD = 65%, loan amount $13,722):

| Target RAROC | Recommended rate |
|---|---|
| 10% | 4.91% |
| 15% | 5.06% |
| 20% | 5.20% |

Caveats:

- The rates are sensitive to four assumed inputs (`cof_rate` 3.0%, `opex_rate` 1.5%, `k` = 1.0, target RAROC 15%). None are fitted from data.
- About 21% of loans get PD = 0.0 from isotonic calibration and are priced at the 4.5% floor with no risk premium.
- Very high-PD applicants still hit the 60% cap. No rate can deliver the target RAROC for them under these assumptions, which in practice is a signal to decline, not reprice.
- PD, LGD and EAD are treated as fixed. Raising the rate on a marginal borrower can raise their true PD (adverse selection), and that feedback loop is not modeled.

## The grade_subgrade question

`grade_subgrade` is very likely a risk grade the lender assigns at underwriting, after already assessing the applicant. That makes it close to a proxy for the answer, not an independent input. For pre-underwriting screening, where no grade exists yet, it should be dropped.

The model was retrained without it: same recipe, same hyperparameters, same constraints, so only that one field differs.

| | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Main model (with `grade_subgrade`) | 0.8933 | n/a | 0.0786 |
| Pre-underwriting (without) | 0.8939 | 0.7890 | 0.0787 |

Dropping the grade barely changes ranking power. The two ROC-AUCs cannot be told apart given the +/-0.46-point CV spread, and the Brier scores match to four decimals. A lower but more honest number had been expected. The likely reason is that `grade_subgrade` is a near-deterministic function of the same raw fields (`debt_to_income_ratio`, `credit_score`, `employment_status` and so on), so once those are in the model the grade adds little.

What does change is the balance at the 0.5 threshold. Recall on defaults falls from about 0.70 to 0.54, and precision on defaults rises to 0.96. Without the grade to lean on, the model is more conservative about flagging applicants, even though it ranks them just as well. Importance shifts to `debt_to_income_ratio`, `credit_score`, `employment_status`, income and delinquency history.

Which model to use:

- Pre-approval and application-time scoring: the pre-underwriting model.
- Portfolio loss, pricing and the fair-lending audit: the main model, because those concern loans that were already graded and approved. It is not meaningfully more accurate.

## Fair lending audit

### Legal framework (U.S.)

Credit decisions fall mainly under the Equal Credit Opportunity Act (ECOA, 15 U.S.C. Section 1691 et seq.) and Regulation B (12 C.F.R. Part 1002). They prohibit credit discrimination based on race, color, religion, national origin, sex, marital status, age (if the applicant can enter a contract), receipt of public assistance income, or good-faith exercise of rights under the Consumer Credit Protection Act. When a loan is secured by a dwelling, the Fair Housing Act (FHA, 42 U.S.C. Section 3601 et seq.) adds similar protections.

Both recognize disparate-impact liability (neutral practices with discriminatory effects), not just disparate treatment. The Supreme Court affirmed disparate impact under the FHA in *Texas Dept. of Housing & Community Affairs v. Inclusive Communities Project*, 576 U.S. 519 (2015).

Regulation B Section 1002.6(b)(2)-(3) sets the "elderly rule" for age in empirically derived scoring systems. Age can be a predictive variable, but applicants 62 and over cannot be given a negative factor for being elderly compared with a similarly situated younger applicant.

### Data limitation

The dataset has no race, ethnicity, religion or national origin fields, which are the attributes most central to ECOA and FHA exams. The audit covers only `gender`, `age`, and `education_level` as an exploratory proxy check. It is not a complete ECOA/FHA review. A production review would get race and ethnicity directly or through Bayesian Improved Surname Geocoding (BISG) and involve fair-lending counsel.

- `gender` and `age` (62+ versus under 62, the Reg B threshold) are ECOA-protected classes.
- `education_level` is not a protected class. It is included as a proxy-discrimination screen, since a neutral-looking variable can still cause disparate impact if it correlates with a protected class.

### Metrics

Computed with [Fairlearn](https://fairlearn.org) on the held-out test set with the calibrated main model. Approve means predicted P(default) < 0.5.

- Adverse Impact Ratio (AIR) is the approval rate of a group divided by the approval rate of a reference group. It is the lending version of the EEOC four-fifths rule (29 C.F.R. Section 1607.4(D)). U.S. bank regulators use it as a screening threshold: an AIR below 0.80 flags a group for closer review. It is a screen, not a finding of illegal discrimination.
- Equal Opportunity Difference (Hardt, Price & Srebro, NeurIPS 2016) is the gap in true positive rate between groups, where true positive rate is P(model approves | applicant would have repaid). Values near 0 mean creditworthy applicants in each group are treated alike.

### Results

Test set, n = 4,000, calibrated PD.

| Attribute | Group (n) | AIR vs reference | EO difference |
|---|---|---|---|
| gender (ref: Male, n=1,893) | Female (2,022) | 0.992 | +0.001 |
| | Other (85) | 0.980 | -0.007 |
| age group (ref: under 62, n=2,978) | 62+ (1,022) | 1.019 | +0.007 |
| education level (ref: PhD, n=167) | Bachelor's (1,617) | 0.959 | -0.002 |
| | High School (1,157) | 0.974 | -0.000 |
| | Master's (768) | 0.991 | +0.002 |
| | Other (291) | 0.967 | +0.003 |

- Age (elderly rule, 62+): clean. AIR is 1.019 and the EO difference is +0.007. If anything, elderly applicants do slightly better. This is the one attribute with a specific statutory test, and it shows no sign of a violation.
- Gender (Male vs Female): clean. AIR 0.992, EO difference +0.001, well above the 0.80 screen.
- Gender ("Other", n = 85): clean. Before the calibration fix this group had a borderline AIR of 0.820 and an EO difference of -0.152. With the calibrated model it is 0.980 and -0.007. An apparent flag on a small group can come from a poorly calibrated model and not from a real disparity.
- Education level (not protected, exploratory): no flags. AIRs run 0.96 to 1.00 and EO differences are essentially zero.

Fairlearn's worst-case summary metrics agree: gender `demographic_parity_ratio` is 0.980 (it was 0.815 before calibration), age group is 0.981, and education level is 0.959.

### Next steps for production

1. Get race, ethnicity and national origin (directly or through BISG) to run the full ECOA/FHA comparison set.
2. Rerun the audit after any model change. If a flag ever appears, search for a less discriminatory alternative (for example Fairlearn's `ExponentiatedGradient` or `ThresholdOptimizer`), in line with CFPB guidance on model risk management for ML underwriting models.
3. Check whether `grade_subgrade` or `employment_status`, the dominant features, correlate with any flagged group. That would make them a proxy channel even though neither is protected.
4. Rerun the audit periodically, say quarterly. It is a snapshot, not a standing guarantee.

### References

- Equal Credit Opportunity Act, 15 U.S.C. Section 1691 et seq.; Regulation B, 12 C.F.R. Part 1002 (including Section 1002.6(b)(2)-(3))
- Fair Housing Act, 42 U.S.C. Section 3601 et seq.
- *Texas Dept. of Housing & Community Affairs v. Inclusive Communities Project*, 576 U.S. 519 (2015)
- Uniform Guidelines on Employee Selection Procedures, 29 C.F.R. Section 1607.4(D)
- Interagency Fair Lending Examination Procedures (CFPB/OCC/FDIC/Federal Reserve)
- Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." NeurIPS.
- [Fairlearn documentation](https://fairlearn.org)

## Model risk management

U.S. bank supervisors (Federal Reserve SR 11-7 and OCC Bulletin 2011-12) expect three things from any model used in a credit decision: sound development, independent validation, and ongoing monitoring.

Development: covered.

- Conceptual soundness: methodology, assumptions and math are documented above.
- Data quality: outlier, domain-rule and consistency checks ran before modeling (`cleaning_system/`).
- Performance testing: held-out test set, 5-fold cross-validation, hyperparameter search.
- Calibration testing: before-and-after calibration curves and Brier scores, not just ranking metrics.
- Sensitivity and robustness: monotonic-constraint verification, the alternate feature-set model, and sensitivity notes on each dollar-based assumption.
- Fair lending and bias testing: see the audit above.

Independent validation: not done, and it is a gap. SR 11-7 requires validation by staff independent of the model's developer, with authority to require changes. One person built and checked everything in this notebook, and no one has reviewed it independently. A real deployment needs a separate validation team to replicate the key results, challenge the LGD, cost-of-funds, opex and capital-multiplier assumptions, and rerun the fair-lending audit on production data, including race and ethnicity.

Ongoing monitoring: not set up, and it is a gap. A write-up is not a live monitoring process. A production deployment would need at least:

- Regular revalidation of calibration on fresh outcomes instead of assuming it holds.
- Population stability monitoring, to catch drift in the applicant pool away from the training data.
- A periodic rerun of the fair-lending audit, because both the applicant population and the model change over time.
- A defined retraining trigger, such as calibration drift past a threshold or a fixed review schedule.

Effective challenge: each fix in this project came from checking the model's own output against another test. That was useful self-checking, but it does not replace challenge from someone whose job is to find what this analysis missed.

Bottom line: development is solid and documented. The model is not ready for production under SR 11-7 / OCC 2011-12. It has no independent validation and no monitoring setup, and both are organizational requirements that more notebook work cannot supply.
