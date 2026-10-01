Loan default prediction model: write-up
========================================

1. Goal
-------
Predict whether a borrower will default, using the applicant and loan
fields in loan_dataset_20000.csv (20,000 rows).

The raw target is `loan_paid_back` (1 = paid back, 0 = defaulted). The
notebook flips it to `default = 1 - loan_paid_back`, so 1 always means
the bad outcome, and every model predicts that `default` label.

The class split is 80% paid back and 20% defaulted. That's real, not a
data error, so it's handled with class weighting (section 4).


2. Input features
-----------------
Numeric (15):
  age, annual_income, monthly_income, debt_to_income_ratio, credit_score,
  loan_amount, interest_rate, loan_term, installment, num_of_open_accounts,
  total_credit_limit, current_balance, delinquency_history, public_records,
  num_of_delinquencies

Categorical (6):
  gender, marital_status, education_level, employment_status, loan_purpose,
  grade_subgrade

Nothing is missing, so there was no imputation.


3. EDA walk-through
----------------------
- employment_status is the strongest single driver. Unemployed borrowers
  default 82% of the time, students 59%, self-employed and employed
  about 11%, retired about 0.5%.
- grade_subgrade is a lender-assigned credit grade (A1 best, F5 worst).
  It lines up almost perfectly with default rate: A grades default
  around 4-8%, F grades around 30-35%.
- Among the numeric fields, debt_to_income_ratio (correlation +0.22) and
  credit_score (-0.20) are the strongest on their own. Income, loan
  amount, credit limit and balance have almost no linear correlation 
  with default.
- A separate outlier pass (cleaning_system/) flagged 1,455 rows (7.3%)
  on statistical grounds, mostly from the skewed income and balance
  columns. It found no domain-rule violations and no cross-field
  inconsistencies. The flagged rows are unusual but not broken, so none
  were dropped before modeling.


4. Modeling approach
--------------------
The split is 80% train, 20% test, stratified on the target so both keep
the 80/20 class ratio.

Preprocessing is one scikit-learn ColumnTransformer, fit on the training
data only so nothing leaks from the test set:
  - Numeric columns go through StandardScaler.
  - Categorical columns go through OneHotEncoder. The first category is
    dropped to avoid redundant dummies, and unknown categories at test
    time are ignored.

There are three models that was trained, each as a Pipeline of preprocessing plus
classifier.

  a) Logistic regression
     A simple linear baseline. class_weight="balanced" reweights the
     loss so the default class isn't ignored. It's the easiest to read,
     since the coefficients show the direction and size of each effect.

  b) Random forest
     400 trees, max_depth=8, min_samples_leaf=20. The shallow trees and
     leaf floor keep it from overfitting on 16k training rows. It also
     uses class_weight="balanced", and it picks up non-linear effects
     and interactions without being told about them.

  c) XGBoost
     400 trees, max_depth=4, learning_rate=0.05, subsample and
     colsample at 0.8. scale_pos_weight is the majority/minority ratio
     in the training set, which is XGBoost's version of class weighting.
     Boosted trees usually do best on tabular data with mixed numeric
     and categorical signal, and that held here.


5. Results (held-out test set, 4,000 rows)
------------------------------------------
                        ROC-AUC   PR-AUC   Recall(default)
  Logistic Regression    0.886    0.772        0.73
  Random Forest          0.882    0.769        0.69
  XGBoost                0.892    0.787        0.70

ROC-AUC measures how well the model ranks defaulters above
non-defaulters (0.5 is random, 1.0 is perfect). About 0.89 is strong for
credit-risk data like this.

PR-AUC says more than ROC-AUC when classes are imbalanced. It's also
strong here at 0.77-0.79.

Recall is the share of actual defaulters the model catches at a 0.5
threshold. All three catch roughly 70-73%, and the price is some false
positives: good borrowers who get flagged. Moving the threshold trades
one against the other. A bank that cares most about catching defaulters
would push recall up, and one that hates turning away good borrowers
would push precision up.

XGBoost won on both ROC-AUC and PR-AUC, so it's the model used from here
on.


6. Monotonic constraints (financial domain rules)
-------------------------------------------------
None of the three models above has a monotonic constraint. A tree model
can learn that risk dips at some high debt-to-income value purely from
noise, and nothing stops it, even though finance says the relationship
should run one way.

To fix that, a fourth model (xgboost_monotonic) was trained with
XGBoost's `monotone_constraints`. Each constrained numeric feature can
now move predicted default probability in only one direction. The
one-hot categorical columns (gender, marital_status, education_level,
employment_status, loan_purpose, grade_subgrade) are unordered, so they
get constraint 0.

The constraints follow standard credit-risk logic.

  Raising the feature can never lower predicted risk (+1):
    - debt_to_income_ratio    less room to repay
    - loan_amount             bigger exposure
    - interest_rate           a higher rate reflects and adds to risk
    - installment             heavier payment burden
    - delinquency_history     more past delinquencies
    - num_of_delinquencies    same idea
    - public_records          more adverse records

  Raising the feature can never raise predicted risk (-1):
    - credit_score            a higher score means lower risk
    - annual_income           more ability to repay
    - monthly_income          same as annual_income

  No constraint (0), because the direction is genuinely unclear:
    - age                     risk isn't monotonic across life stages
    - loan_term               tangled up with pricing
    - num_of_open_accounts    access on one side, exposure on the other
    - total_credit_limit      cushion on one side, exposure on the other
    - current_balance         ambiguous without scaling by limit or income
    - all one-hot categorical columns

To check, each constrained feature was swept across its observed range,
with everything else held at the median or mode. Predicted default
probability moved only in the allowed direction for all 10 constrained
features. Accuracy didn't suffer. It improved slightly:

                           ROC-AUC   PR-AUC
  XGBoost (unconstrained)   0.8918    0.7865
  XGBoost (monotonic)       0.8932    0.7891

So the model now follows the financial rules by construction, not just
as a tendency in the data, and it costs nothing in accuracy. That
matters for a model that would drive or explain real lending decisions:
an auditor can point to a guaranteed direction instead of "the data
happened to show this."


7. Cross-validation
-------------------
Every number so far comes from one 80/20 split. That's one sample of
how the model does on unseen data, with no sense of how much it would
move on a different split. So 5-fold stratified cross-validation was run
on the training set, using the same monotonic XGBoost pipeline. It was
unfitted and refit fold by fold, and the test set stayed untouched.

  ROC-AUC: 0.8890 +/- 0.0046  (folds: 0.8882, 0.8909, 0.8960, 0.8819, 0.8879)
  PR-AUC:  0.7818 +/- 0.0091  (folds: 0.7865, 0.7799, 0.7935, 0.7660, 0.7831)

The held-out test set gave ROC-AUC 0.8932 and PR-AUC 0.7891. Both are
within about one fold-to-fold standard deviation of the CV mean, so the
test result isn't a lucky split.


8. Hyperparameter tuning
------------------------
The XGBoost settings used so far (n_estimators=400, max_depth=4,
learning_rate=0.05, subsample=0.8, colsample_bytree=0.8) were picked by
hand. To see whether they're near the best, a RandomizedSearchCV was run
with 25 candidates and 3-fold CV, scored on ROC-AUC. The monotone
constraints stayed fixed, since they're a domain requirement and not
something to tune.

  Best CV ROC-AUC in the search: 0.8926
  Best params: n_estimators=442, max_depth=5, learning_rate=0.019,
    subsample=0.855, colsample_bytree=0.732, min_child_weight=8,
    reg_lambda=3.78

  Tuned model, held-out test set:  ROC-AUC 0.8963, PR-AUC 0.7932
  Hand-picked model:               ROC-AUC 0.8932, PR-AUC 0.7891

Decision: keep the hand-picked hyperparameters. The tuned model is about
0.3 points better, which is less than the 0.46-point fold-to-fold
spread from section 7, so it's within noise. Switching would mean
redoing calibration, EL/CECL, pricing and the fair-lending audit for a
gain that can't be told apart from sampling variation. The tuned values are
recorded here in case someone wants to revisit this with a bigger
search or more data.


9. Probability calibration
--------------------------
Everything downstream (portfolio EL/CECL, the RAROC pricing optimizer,
the fair-lending approval decisions) uses the model's predicted
probability directly as PD. That only works if predict_proba is
calibrated: of all loans predicted at PD=30%, about 30% should actually
default. XGBoost trained with scale_pos_weight to handle the 80/20
imbalance gives up exactly this. Reweighting the loss pushes predicted
probabilities away from true frequencies.

The check was a 10-bin calibration curve on the test set. The raw
xgboost_monotonic model was fine at the extremes and badly overconfident
in the middle:

  mean predicted PD   observed default rate   gap
        0.507               0.190             -0.317
        0.683               0.347             -0.335

Raw Brier score: 0.1131.

The fix was sklearn's CalibratedClassifierCV (isotonic, 5-fold) wrapped
around the same monotonic XGBoost pipeline. Isotonic regression is a
non-decreasing map, so it can't change how applicants are ranked. ROC-AUC
stays put, and only the translation from raw score to probability
changes.

After calibration:

  Brier score: 0.0786 (was 0.1131)
  ROC-AUC:     0.8933 (was 0.8932, so ranking is preserved as expected)
  Calibration gaps are within about 1.5 percentage points in every bin
  (they were up to 33.5 points off).

Sections 10 through 15 use this calibrated PD, not the raw output. It's
not a cosmetic change. Sections 10, 12 and 14 show how far the dollar
and fairness figures moved.

One caveat. Isotonic regression fits a step function, so it can flatten
a low-risk region to exactly PD=0.0 instead of a small positive number.
About 4,146 of 20,000 loans (21%) land at exactly 0.0. If smoother
low-end probabilities matter more than the best possible calibration
fit, Platt scaling (method="sigmoid") is a less flexible alternative
worth trying.


10. Portfolio loss: expected loss (EL) and CECL
-----------------------------------------------
Standard credit-risk framework:
    Expected Loss (EL) = PD x LGD x EAD

CECL, basic form:
    Allowance for Credit Losses (ACL) = Amortized Cost Basis x
                                        Estimated Lifetime Loss Rate

Here is how each term gets filled in from this data and model.

  PD: the predicted default probability from the calibrated monotonic
      XGBoost model. `loan_paid_back` is a terminal outcome (did the
      loan ultimately default), not a rolling 12-month one, so this PD
      is treated as a lifetime PD. That's what CECL needs, and it works
      directly in the EL formula too.

  LGD: not in the dataset (no collateral or recovery field), so it's an
       assumption, proxied by loan_purpose as a stand-in for secured
       versus unsecured:
         Home                -> 25%  (secured, real estate)
         Car                 -> 40%  (secured, depreciating asset)
         all other purposes  -> 65%  (unsecured: debt consolidation,
                                      business, medical, education,
                                      vacation, other)
       These are illustrative benchmark severities, not derived from
       this data. A real deployment would use actual collateral and
       recovery history.

  EAD / Amortized Cost Basis: current_balance, the outstanding balance.
       The same figure serves both terms because these are simple,
       non-revolving installment loans.

  Estimated lifetime loss rate (CECL): PD x LGD.

Since EAD and amortized cost basis are both current_balance, and PD is
already lifetime, the CECL formula reduces to the same calculation as
EL. That's expected, not a bug. CECL replaced the old "incurred loss"
approach with a lifetime expected-loss reserve, so it lands on the same
number as lifetime EL when both use the same PD, LGD and exposure.

Results (full portfolio, 20,000 loans, calibrated model from section 9):

  Total outstanding balance         $486,667,892.63
  Portfolio expected loss (EL)       $57,557,149.10
  Portfolio CECL allowance (ACL)     $57,557,149.10  (same as EL)
  Implied reserve ratio (ACL/bal.)   11.83%

The raw, uncalibrated model gave EL/ACL of $96,257,066 (a 19.78%
reserve ratio). Calibration cut the portfolio loss estimate by about
40%, because the raw model overestimated PD through the middle of the
risk range. This is the clearest example of why calibration had to come
before any dollar figure.

Out-of-sample check on the test set only (4,000 loans the model never
trained on), which is more trustworthy because it avoids the optimism of
scoring training loans:

  Test portfolio balance            $98,254,528.69
  Test portfolio EL / ACL           $11,507,727.64
  Test reserve ratio                11.71%

The full-portfolio ratio (11.83%) and the test-only ratio (11.71%) are
close, so in-sample optimism looks small here.

By loan_purpose (full portfolio), sorted by dollar EL:

  loan_purpose         balance($)     avg PD   LGD    EL($)        reserve %
  Debt consolidation   193,868,700     0.201   0.65   25,240,150    13.02%
  Other                 61,534,240     0.199   0.65    8,176,597    13.29%
  Education             40,313,050     0.215   0.65    5,877,585    14.58%
  Business              39,820,500     0.194   0.65    5,034,506    12.64%
  Car                   58,492,900     0.201   0.40    4,881,721     8.35%
  Medical               28,230,240     0.215   0.65    4,015,735    14.22%
  Home                  49,262,180     0.183   0.25    2,322,723     4.72%
  Vacation              15,146,120     0.217   0.65    2,008,129    13.26%

The reserve ratio depends far more on the LGD assumption than on PD,
which is fairly flat (18-22%) across purposes. Home has the lowest
reserve ratio (4.72%) even though its average PD is unremarkable, purely
because its assumed LGD is low. Car sits in between for the same reason.

Limitations:
  - LGD is assumed, not fitted from recovery data, and it's the biggest
    remaining uncertainty in the $57.6M figure. If actual unsecured LGD
    in this book were 50% instead of 65%, EL on the unsecured share would
    drop roughly in proportion.
  - EAD is today's balance, not a projected balance at some future
    default date. For amortizing loans the true EAD at default is
    somewhat lower for loans further from origination. That is ignored
    here to stay with the basic equation.
  - PD is a static estimate fit on history. Full CECL practice adjusts
    for forward-looking macro scenarios, such as a recession overlay.
    This is a point-in-time model, not a scenario-conditioned one.
  - About 21% of loans have PD of exactly 0.0 from isotonic calibration
    (section 9) and contribute $0 to EL. A smoother calibration method
    would give them a small positive PD.


11. What drives the predictions
-------------------------------
Top features by importance in the XGBoost model:
  1. employment_status = Unemployed
  2. employment_status = Student
  3. employment_status = Retired
  4. debt_to_income_ratio
  5. credit_score
  6. interest_rate
  7-15. various grade_subgrade levels, loan_purpose, self-employed status

That matches the EDA. Employment status, credit grade, DTI and credit
score dominate, which is typical for credit risk.


12. Pricing optimizer: interest rate for a target RAROC
-------------------------------------------------------
Given an applicant's PD, recommend the interest_rate that makes that
loan hit a target risk-adjusted return on capital (RAROC), instead of
pricing from grade_subgrade lookup tables.

RAROC as defined here:

    RAROC = (Interest Income - Cost of Funds - Opex - Expected Loss)
            / Economic Capital

Per-loan terms, where r is the interest rate being solved for:

  Interest income     = r * EAD
  Cost of funds       = cof_rate * EAD        (assumed 3.0%)
  Opex                = opex_rate * EAD       (assumed 1.5%)
  Expected loss       = PD * LGD * EAD
  Economic capital    = k * LGD * EAD * sqrt(PD * (1 - PD))
                        (unexpected loss, the standard deviation of a
                        single Bernoulli default event, scaled by a
                        capital multiplier k; default k=1)

PD comes from the calibrated xgboost_monotonic model (section 9). LGD
uses the same loan_purpose assumption as section 10 (Home 25%, Car 40%,
else 65%). EAD is loan_amount here, because pricing happens at
origination before any balance has amortized. That differs from section
10, which used current_balance for the existing book.

Closed-form solution: interest income is the only RAROC term that
depends on r, and every other term is linear in EAD. Setting RAROC(r)
equal to the target therefore has a closed-form answer, and no numerical
solver is needed:

    r* = cof_rate + opex_rate + PD*LGD
         + target_RAROC * k * LGD * sqrt(PD * (1 - PD))

Read left to right, that's cost of funds, operating cost, the
expected-loss rate, and a capital charge that grows with the target
return and the loan's own risk. Higher PD, higher LGD or a higher target
RAROC all push the rate up. Recommended rates are clipped at 60%, the
interest_rate bound from cleaning_system/config.yaml.

Verification: recomputing achieved RAROC at each unclipped, interior-PD
(0 < PD < 1) recommended rate reproduces the target, matching to 1e-9.
Loans with PD of exactly 0 or 1 (about 4,146 of 20,000, from isotonic
calibration, see section 9) are priced at the 4.5% cost-of-funds plus
opex floor. Their economic capital is exactly 0, so RAROC is undefined
there, or trivially met at any rate above the floor.

Results at a 15% target RAROC, full portfolio, 20,000 loans, calibrated
PD:

                              current    recommended
  mean interest rate           12.40%        17.54%
  median interest rate         12.40%        13.25%
  min                           3.14%         4.50%
  max                          22.51%        60.00% (clipped)

  Loans priced below the target-RAROC rate (underpriced):  10,698 (53.5%)
  Loans priced above the target-RAROC rate (overpriced):    9,302 (46.5%)

With the raw, uncalibrated PD, the mean recommended rate was 26.1% and
71.1% of loans looked underpriced. After calibration the recommendations
sit much closer to the current book: a 17.5% mean, a median of 13.3%
against 12.4%, and an almost even underpriced/overpriced split. Along
with the EL/CECL figure in section 10, that's a second case of the
calibration fix changing a business-facing number and not just an
internal metric.

The average gap between recommended and current rate still rises with
credit grade, from about -2.1 points for A1 (those loans now look
slightly overpriced) to about +10.6 points for F1. The biggest repricing
gap is concentrated in the worst-graded loans, since their higher PD
pushes both the expected-loss and capital-charge terms up. The gap is
much smaller than it was with the uncalibrated PD.

Worked example (one applicant, PD=0.20%, LGD=65%, loan_amount=$13,722):
  target RAROC 10% -> recommended rate 4.91%
  target RAROC 15% -> recommended rate 5.06%
  target RAROC 20% -> recommended rate 5.20%

Caveats:
  - The recommended rates are still sensitive to four assumed inputs
    (cof_rate 3.0%, opex_rate 1.5%, capital multiplier k=1.0, target
    RAROC 15%). None are fitted from data.
  - About 21% of loans get PD=0.0 from isotonic calibration and are
    priced at the 4.5% floor with no risk premium at all. See the
    isotonic caveat in section 9.
  - Very high-PD applicants still hit the 60% cap, which means no rate
    can deliver the target RAROC for them under these assumptions. In
    practice that's a signal to decline, not reprice.
  - PD, LGD and EAD are treated as fixed. In reality, raising the rate
    on a marginal borrower can raise their true PD (adverse selection).
    That feedback loop isn't modeled.


13. A caveat about grade_subgrade (resolved in section 15)
----------------------------------------------------------
grade_subgrade and employment_status dominate the model's predictions.
grade_subgrade is very likely a risk grade the lender assigns at
underwriting, after already assessing the applicant. That makes it close
to a proxy for the answer, not an independent input.

It matters like this:
  - To explain or audit why defaults happen, or to score loans that were
    already approved, the current model (grade_subgrade included) is
    fine and performs best.
  - For pre-underwriting screening, meaning scoring an applicant before
    any grade exists, grade_subgrade should be dropped and the model
    retrained on the raw applicant fields.

Section 15 builds and tests that. The result was a surprise: dropping
grade_subgrade barely moved ROC-AUC (0.8939 against 0.8933). A lower but
more honest number had been expected.


14. Fair lending: bias and disparate-impact audit
-------------------------------------------------
Legal framework (U.S.):
Credit decisions fall mainly under the Equal Credit Opportunity Act
(ECOA, 15 U.S.C. Section 1691 et seq.) and its Regulation B (12 C.F.R.
Part 1002). They prohibit credit discrimination based on race, color,
religion, national origin, sex, marital status, age (if the applicant
can enter a contract), receipt of public assistance income, or
good-faith exercise of rights under the Consumer Credit Protection Act.
When a loan is secured by a dwelling, the Fair Housing Act (FHA, 42
U.S.C. Section 3601 et seq.) adds similar protections. Both recognize
disparate-impact liability (neutral practices with discriminatory
effects), not just disparate treatment. The Supreme Court affirmed
disparate impact under the FHA in Texas Dept. of Housing & Community
Affairs v. Inclusive Communities Project, 576 U.S. 519 (2015).
Regulation B Section 1002.6(b)(2)-(3) sets a specific rule for age in
empirically derived scoring systems, the "elderly rule." Age can be a
predictive variable, but applicants 62 and over can't be given a
negative factor for being elderly compared with a similarly situated
younger applicant.

Data limitation: the dataset has no race, ethnicity, religion or
national origin fields, which are the attributes most central to ECOA
and FHA exams. The audit therefore covers only what's available
(gender, age, and education_level as an exploratory proxy check). It is
not a complete ECOA/FHA review. A production review would get
race/ethnicity directly or through Bayesian Improved Surname Geocoding
(BISG) and involve fair-lending counsel.

  - gender and age (62+ versus under 62, the Reg B threshold) are
    ECOA-protected classes.
  - education_level is not an ECOA/FHA-protected class. It's included as
    a proxy-discrimination screen, since a neutral-looking variable can
    still cause disparate impact if it correlates with a protected
    class.

Metrics, computed with the Fairlearn library (fairlearn.org) on the
held-out test set with the calibrated xgboost_monotonic model from
section 9. Approve means predicted P(default) < 0.5.

  Adverse Impact Ratio (AIR) = approval rate of a group / approval rate
  of a reference group. This is the lending version of the EEOC
  "four-fifths rule" (29 C.F.R. Section 1607.4(D), originally an
  employment-selection standard). U.S. bank regulators (CFPB, OCC, FDIC,
  Federal Reserve, per the Interagency Fair Lending Examination
  Procedures) use it as a screening threshold. An AIR below 0.80 flags a
  group for closer review. It's a screen, not a finding of illegal
  discrimination.

  Equal Opportunity Difference (Hardt, Price & Srebro, "Equality of
  Opportunity in Supervised Learning," NeurIPS 2016) = the gap in true
  positive rate between groups, where true positive rate is P(model
  approves | applicant would actually have repaid). Values near 0 mean
  creditworthy applicants in each group are treated alike.

Results (test set, n=4,000, calibrated PD):

  Attribute             Group (n)                AIR vs ref   EO diff vs ref
  gender (ref: Male,    Female (2,022)              0.992         +0.001
   n=1,893)             Other (85)                  0.980         -0.007
  age_group (ref:       62+ / elderly (1,022)       1.019         +0.007
   under 62, n=2,978)
  education_level       Bachelor's (1,617)          0.959         -0.002
   (ref: PhD,           High School (1,157)         0.974         -0.000
   n=167)               Master's (768)              0.991         +0.002
                        Other (291)                 0.967         +0.003

Findings:
  - Age (ECOA elderly rule, 62+): clean. AIR is 1.019 and the EO
    difference is +0.007, so no meaningful disparity. If anything,
    elderly applicants do slightly better. This is the one attribute
    with a specific statutory test (Reg B Section 1002.6(b)(2)-(3)), and
    it shows no sign of a violation.
  - Gender (Male vs Female): clean. AIR 0.992, EO difference +0.001,
    well above the 0.80 screen.
  - Gender ("Other", n=85): clean now too. Before the calibration fix
    (section 9) this group had a borderline AIR of 0.820 and an EO
    difference of -0.152, both worrying on their face. With the
    calibrated model it's AIR 0.980 and EO difference -0.007. So an
    apparent fair-lending flag on a small group can come from a
    poorly calibrated model and not from a real disparity. Chasing it
    would have meant investigating the wrong cause.
  - Education level (not protected, exploratory): no flags, same as
    before calibration. AIRs run 0.96 to 1.00, and EO differences are
    essentially zero (-0.002 to +0.003).

Fairlearn's own worst-case summary metrics agree: gender
demographic_parity_ratio is 0.980 (it was 0.815 before calibration),
age_group is 0.981, and education_level is 0.959. All clean.

Next steps for production:
  1. Get race, ethnicity and national origin (directly or through BISG)
     to run the full ECOA/FHA comparison set.
  2. No flags stand today, but rerun the audit after any model change
     (retraining, recalibration, new features). If a flag ever appears,
     search for a less discriminatory alternative (for example
     Fairlearn's ExponentiatedGradient or ThresholdOptimizer) to see
     whether a comparably accurate model has a smaller gap, in line
     with CFPB guidance on model risk management for ML underwriting
     models.
  3. Check whether grade_subgrade or employment_status, the dominant
     features, correlate with any flagged demographic group. That would
     make them a proxy channel even though neither is itself protected.
  4. Rerun the audit periodically, say quarterly. It's a snapshot, not a
     standing guarantee.

Citations:
  - Equal Credit Opportunity Act, 15 U.S.C. Section 1691 et seq.;
    Regulation B, 12 C.F.R. Part 1002 (including Section 1002.6(b)(2)-(3),
    the "elderly rule").
  - Fair Housing Act, 42 U.S.C. Section 3601 et seq.
  - Texas Dept. of Housing & Community Affairs v. Inclusive Communities
    Project, 576 U.S. 519 (2015) (disparate impact under the FHA).
  - Uniform Guidelines on Employee Selection Procedures, 29 C.F.R.
    Section 1607.4(D) (source of the four-fifths rule used as the AIR
    screening convention).
  - Interagency Fair Lending Examination Procedures (CFPB/OCC/FDIC/
    Federal Reserve), which describe AIR-style disparate-impact
    screening in bank exams.
  - Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity
    in Supervised Learning." NeurIPS. (Source of the Equal Opportunity
    Difference metric.)
  - Fairlearn documentation, https://fairlearn.org (metric
    implementations used in this audit).


15. Pre-underwriting model (dropping grade_subgrade)
----------------------------------------------------
This resolves the caveat in section 13. The same recipe was retrained
(monotonic-constrained XGBoost, isotonic calibration, same
hyperparameters and constraint logic) on the raw applicant and loan
fields only, with grade_subgrade removed. That isolates the effect of
dropping that one field.

Results (held-out test set):
                                        ROC-AUC   PR-AUC   Brier
  Main model (with grade_subgrade)       0.8933    n/a     0.0786
  Pre-underwriting (no grade_subgrade)   0.8939   0.7890   0.0787

Dropping grade_subgrade barely changes ranking power. The two ROC-AUCs
can't be told apart given the +/-0.46-point cross-validation spread
from section 7, and the Brier scores match to four decimals. That
contradicts the assumption in section 13 that removing the lender's own
grade would clearly cost accuracy. The likely reason is that
grade_subgrade is a near-deterministic function of the same raw fields
(debt_to_income_ratio, credit_score, employment_status and so on). Once
those are in the model, the grade adds little.

What does change is the precision/recall balance at the 0.5 threshold.
Recall on defaults falls from about 0.70 to 0.54, and precision on
defaults rises to 0.96. Without the grade to lean on, the model is more
conservative about flagging high-risk applicants, even though it ranks
them just as well. Feature importance shifts to debt_to_income_ratio,
credit_score, employment_status, income and delinquency history, the
raw applicant signals expected when the underwriter's own grade
isn't available.

Use: this model (saved as
xgboost_monotonic_calibrated_preunderwriting.joblib, section 16) is the
one for pre-approval and application-time scoring. The main calibrated
model stays in use for portfolio loss, pricing and the fair-lending
audit, because those concern loans that were already graded and
approved. It is not meaningfully more accurate.


16. Saved model files
---------------------
Both production models are saved with joblib, so new applicants can be
scored without rerunning the notebook.

  models/xgboost_monotonic_calibrated.joblib
      Full pipeline (preprocessing, monotonic XGBoost, isotonic
      calibration), with grade_subgrade. For scoring or explaining
      already-graded or approved loans.
  models/xgboost_monotonic_calibrated_preunderwriting.joblib
      Same recipe without grade_subgrade. For pre-approval and
      application-time scoring.
  models/metadata.json
      Feature lists, use case, and test-set ROC-AUC and Brier for each
      model, in machine-readable form.
  requirements.txt
      The package versions the models were fit under (pandas 2.3.2,
      numpy 2.3.2, scikit-learn 1.7.1, xgboost 3.0.5, fairlearn 0.14.0,
      matplotlib 3.10.6, seaborn 0.13.2, scipy 1.16.1, joblib 1.5.2).
      These are needed to reload the .joblib files without version
      mismatch errors.

xgboost_monotonic_calibrated.joblib was reloaded in a fresh Python
process to score a sample applicant. The PD matched the in-memory model from
the notebook run exactly.


17. Model risk management (SR 11-7)
-----------------------------------
U.S. bank supervisors (Federal Reserve SR 11-7 and OCC Bulletin 2011-12,
"Supervisory Guidance on Model Risk Management") expect three things
from any model used in a credit decision: sound development,
independent validation, and ongoing monitoring. Here is what this
analysis covers for each, and what a real deployment would still need.

Development (covered):
  - Conceptual soundness: the methodology, assumptions and math are
    written up in sections 4, 6, 9, 10 and 12.
  - Data quality: outlier, domain-rule and consistency checks ran before
    modeling (cleaning_system/, section 3).
  - Performance testing: held-out test set (section 5), 5-fold
    cross-validation (section 7), hyperparameter search (section 8).
  - Calibration testing: before-and-after calibration curves and Brier
    scores (section 9), not just ranking metrics.
  - Sensitivity and robustness: monotonic-constraint verification
    (section 6), the alternate feature-set model (section 15), and
    sensitivity notes on each dollar-based assumption (LGD, cof_rate,
    opex_rate, capital multiplier; sections 10 and 12).
  - Fair lending and bias testing: section 14.

Independent validation (not done, a gap):
  SR 11-7 requires validation by staff independent of the model's
  developer, with authority to require changes. One person built and
  checked everything in this notebook, and no one has reviewed it
  independently. That's the largest governance gap before any
  production use. A real deployment needs a separate validation team to
  replicate the key results, challenge the LGD, cof_rate, opex_rate and
  capital-multiplier assumptions, and rerun the fair-lending audit on
  production data, including race and ethnicity (see the data
  limitation in section 14).

Ongoing monitoring (not set up, a gap):
  A model write-up isn't a live monitoring process. A production
  deployment would need at least:
  - Regular revalidation of calibration (recompute the section 9 curve
    on fresh outcomes instead of assuming it holds).
  - Population stability monitoring, to catch drift in the applicant
    pool away from the training data.
  - A periodic rerun of the fair-lending audit (section 14), because
    both the applicant population and the model change over time.
  - A defined retraining trigger, such as calibration drift past a
    threshold or a fixed review schedule, instead of ad hoc retraining.

Effective challenge:
  Each fix in this project came from checking the model's own output
  against another test (EDA, model, calibration, financial
  application, fairness audit, robustness checks, alternate model). That
  was useful self-checking, but it doesn't replace challenge from
  someone whose job is to find what this analysis missed.

Bottom line: development is solid and documented. The model is not
ready for production under SR 11-7 / OCC 2011-12. It has no independent
validation and no monitoring setup, and both are organizational
requirements that more notebook work can't supply.


18. Files
---------
  loan_default_eda_model.ipynb   EDA and training notebook (executed,
                                 with plots and tables embedded)
  model_explanation.txt          This document
  loan_dataset_20000.csv         Source data
  cleaning_system/               Outlier and anomaly detection used to
                                 vet the data before modeling
  models/                        Saved models (section 16):
                                 xgboost_monotonic_calibrated.joblib,
                                 xgboost_monotonic_calibrated_preunderwriting.joblib,
                                 metadata.json
  requirements.txt               Package versions the models were fit
                                 under (section 16)
