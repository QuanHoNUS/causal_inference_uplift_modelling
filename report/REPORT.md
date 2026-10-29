# Marketing Campaign Optimization: Technical Report

## Technical Summary

The Criteo campaign increases conversion. Across four primary estimators, the estimated average treatment effect
ranges from 0.000966 to 0.001152. The AIPW estimate is 0.000966 with a 95% confidence interval of
[0.000898, 0.001033]. This corresponds to an increase of about 0.10 percentage points over a control
conversion rate of 0.194%.

Treatment effects are heterogeneous enough to support targeted delivery. S-Learner and DR-Learner produce
the highest Qini point estimates, but a paired bootstrap does not distinguish between them. The policy
workflow selects S-Learner and 2% coverage on a validation set. On an independent policy-test set, this
fixed policy has an estimated profit of $8,441 with a 95% interval of [$1,504, $15,377]. Treating everyone
has an estimated profit of -$203,569 on the same cohort.

These profit estimates use an assumed value of $50 per conversion and a cost of $0.50 per contact. They are
useful for comparing policies, but they are not estimates of Criteo's actual campaign economics.

## 1. Business Question and Data

The analysis addresses three questions:

1. Does the campaign cause additional conversions?
2. Which customers have the largest estimated treatment effects?
3. Which targeting rule gives the best value after contact costs?

The Criteo Uplift Prediction Dataset v2.1 contains 13,979,592 observations from a randomized
incrementality experiment. Each observation has 12 anonymized pre-treatment features, treatment assignment,
realized exposure, visit, and conversion. Conversion is the primary outcome.

| Data characteristic | Value |
|---|---:|
| Rows | 13,979,592 |
| Treatment share | 85.0% |
| Control share | 15.0% |
| Overall conversion rate | 0.292% |
| Treated conversion rate | 0.309% |
| Control conversion rate | 0.194% |
| Missing values | 0 |

The raw data contains about 1.66 million exact duplicate rows. They are retained because several anonymized
features have low cardinality. Removing duplicates would treat shared feature vectors as duplicate
experimental units without evidence that they represent the same customer.

## 2. Treatment Assignment and Identification

The assignment diagnostics are consistent with randomization.

| Diagnostic | Result |
|---|---:|
| Features with absolute SMD above 0.1 | 0 of 12 |
| Maximum absolute SMD | 0.049 |
| Cross-validated treatment-prediction AUC | 0.509 |

The propensity distributions for treated and control observations overlap and remain close to the overall
treatment rate of 0.85. These checks support an intent-to-treat analysis based on assigned treatment.

The causal graph distinguishes pre-treatment features from post-treatment variables. Exposure and visit
are mediators of the campaign effect, so they are not included as adjustment variables. Visit may also act
as a collider if unobserved intent affects both visiting and conversion. Conditioning on visitors would
therefore change the estimand and could introduce bias.

Under randomized assignment, Difference in Means identifies the ATE. In an observational campaign,
identification would require treatment assignment to be conditionally independent of potential outcomes
after adjustment for the measured features. That assumption cannot be tested directly when relevant
confounders are unobserved.

## 3. Average Treatment Effect

All estimators find a positive effect on conversion.

| Estimator | ATE | SE | 95% CI |
|---|---:|---:|---:|
| Difference in Means | 0.001152 | 0.0000344 | [0.001085, 0.001219] |
| Regression Adjustment | 0.000983 | 0.0000016 | [0.000979, 0.000986] |
| IPW (Horvitz-Thompson) | 0.000991 | 0.0000362 | [0.000920, 0.001062] |
| Doubly Robust (AIPW) | 0.000966 | 0.0000344 | [0.000898, 0.001033] |
| IPW (Hajek, stabilized) | 0.000991 | 0.0000360 | [0.000920, 0.001062] |

The point estimates imply a relative lift of roughly 50% to 59% over the control conversion rate. AIPW is
used for the main confidence interval because its influence function includes residual outcome variation.
The Regression Adjustment interval is too narrow for inference: its plug-in variance treats fitted outcome
models as fixed and omits their estimation error.

The visit outcome gives the same qualitative result, with an absolute effect of about 0.7 to 1.0 percentage
points depending on the estimator.
Detailed estimates are in `report/tables/ate_estimators_visit.csv`.

## 4. Confounding Stress Test

A semi-synthetic experiment tests how the estimators behave after measured confounding is introduced. The
observed `(X, T, Y)` values are unchanged. Retention depends on treatment and standardized `f4`, the feature
most correlated with conversion. The standardized value is clipped to [-2, 2] and the retention coefficient
is set to 1.0. In the retained sample of two million rows, the SMD for `f4` increases from 0.004 to 0.272.

Retention changes both treatment assignment and the distribution of customer features. The appropriate
reference is therefore the ATE in the retained target population, not the full-population ATE. The reference
weights the original randomized data by the marginal retention probability
`q(X) = e*s1(X) + (1-e)*s0(X)`. Its estimate is 0.001586 with a 95% interval of [0.001488, 0.001685].

| Estimator | ATE in retained sample | 95% CI | Bias relative to target reference |
|---|---:|---:|---:|
| Difference in Means | 0.004116 | [0.003970, 0.004262] | +0.002530 |
| Regression Adjustment | 0.001531 | [0.001520, 0.001543] | -0.000055 |
| IPW (Horvitz-Thompson) | 0.000360 | [-0.000371, 0.001091] | -0.001226 |
| Doubly Robust (AIPW) | 0.001619 | [0.000931, 0.002308] | +0.000033 |
| IPW (Hajek, stabilized) | 0.000378 | [-0.000348, 0.001103] | -0.001208 |

Difference in Means overstates the target effect because treatment is confounded after retention.
Regression Adjustment and AIPW recover point estimates close to the weighted RCT reference. Standalone IPW
is unstable in this sample: control weights reach the clipping limit of 100 and the effective control sample
size falls to about 212,000. Hajek normalization changes the IPW estimate only slightly, so normalization
does not resolve the overlap and propensity-model problem in this stress test.

## 5. Uplift Model Evaluation

Five models are trained on a stratified sample of three million observations. The training split contains
2,099,999 rows and the model holdout contains 900,001 rows.

Raw Qini and AUUC values scale with the number of evaluated customers. The normalized columns below report
incremental conversions per 100,000 customers. Confidence intervals come from 200 paired,
treatment-stratified bootstrap samples of the full holdout. The same bootstrap rows are used for every model.

| Model | Raw Qini | Qini / 100k (95% CI) | AUUC / 100k (95% CI) |
|---|---:|---:|---:|
| DR-Learner | 327.21 | 36.36 [23.39, 47.17] | 36.16 [21.06, 49.44] |
| S-Learner | 326.57 | 36.29 [23.79, 47.32] | 36.82 [20.92, 49.31] |
| Causal Forest | 275.26 | 30.58 [16.83, 41.23] | 31.66 [15.45, 44.21] |
| X-Learner | 238.05 | 26.45 [15.22, 36.63] | 26.17 [12.05, 39.08] |
| T-Learner | 158.46 | 17.61 [3.98, 28.11] | 16.21 [-0.19, 28.34] |

The paired Qini difference between S-Learner and DR-Learner is -0.07 per 100,000 customers, with a 95%
interval of [-5.55, 4.48]. Their ranking is not statistically resolved. Both perform better than T-Learner,
and both have higher Qini than X-Learner in the paired bootstrap. Comparisons involving Causal Forest are
less decisive: it sits between the leading models and the weaker point estimates, but several pairwise
intervals include zero.

This pattern is consistent with the data constraints. The treatment split is 85/15 and conversion is rare,
so methods that fit separate response surfaces have fewer control conversions available. This is a possible
explanation for T-Learner's weaker result, not a causal test of model architecture.

### Calibration and Segment Effects

The highest-ranked decile has the largest observed uplift for each model. On the independent policy-test
set, the top S-Learner decile has observed uplift of 0.00667 with a Newcombe 95% interval of
[0.00261, 0.01023]. The bottom decile estimate is 0.00042 with an interval of [-0.00053, 0.00093].

Several other deciles have negative point estimates at different positions in the rankings. There are two
or three such deciles per model, but
every corresponding 95% interval contains zero. The data therefore do not identify a harmed customer
segment. They also do not establish that harm is absent.

## 6. Policy Evaluation

The 900,001-row model holdout is divided into 450,000 policy-validation observations and 450,001 policy-test
observations, stratified by treatment and conversion. Validation outcomes select both the model and the
coverage threshold. Test outcomes are used once to evaluate the resulting fixed policy.

S-Learner has the highest validation Qini. At a value of $50 per conversion and a cost of $0.50 per contact,
the validation set selects 2% coverage. Policy values on the test set are estimated with inverse propensity
scoring using the treatment probability of 0.85.

| Policy | Coverage | Customers targeted | Test profit | 95% CI |
|---|---:|---:|---:|---:|
| Top-K selected on validation | 2.0% | 9,001 | $8,441 | [$1,504, $15,377] |
| Treat None | 0% | 0 | $0 | [$0, $0] |
| Top 10% Uplift | 10% | 45,001 | -$3,226 | [-$11,326, $4,874] |
| Top 20% Uplift | 20% | 90,001 | -$25,804 | [-$34,238, -$17,371] |
| Random 50% | 50.0% | 225,219 | -$99,080 | [-$104,970, -$93,190] |
| Positive Uplift Only | 96.1% | 432,514 | -$195,669 | [-$204,268, -$187,070] |
| Treat All | 100% | 450,001 | -$203,569 | [-$212,295, -$194,843] |

The selected top-2% policy remains profitable on the independent test cohort. The top-10% interval includes
zero, while policies with coverage of 20% or more have intervals below zero. The average incremental revenue
per contact is too small to cover the assumed cost when targeting becomes broad.

The cost-sensitivity analysis repeats this selection/evaluation sequence for each value-cost pair. At
$200 per conversion and $0.10 per contact, validation selects 20% coverage. At the headline $50/$0.50 pair,
it selects 2%. The accompanying CSV reports held-out profit intervals for every scenario.

## 7. Model Comparison and Practical Interpretation

The estimator and model comparisons support four conclusions:

1. Adjustment has little effect on the sign of the ATE in the original data, but it matters after
   confounding is introduced.
2. AIPW is more stable than standalone IPW in the retained sample because weak overlap produces large IPW
   weights.
3. S-Learner and DR-Learner form the leading group on Qini point estimates, but the bootstrap cannot order
   them reliably.
4. Model ranking alone is not the business objective. The validation-selected S-Learner policy produces a
   positive held-out profit under the stated economics.

The recommended operating policy is therefore conditional: score customers with the validation-selected
model, target the top 2% only if the assumed value and cost are appropriate, and update the threshold when
actual campaign economics are available.

## 8. Limitations and Next Steps

- The campaign economics are illustrative. Production decisions require actual conversion value and contact
  cost.
- The Qini/AUUC bootstrap conditions on fitted scores. It captures holdout-sample uncertainty but not
  variation from refitting models on a new training sample.
- The policy interval is conditional on the validation-selected model, threshold, and economic assumptions.
  Transport to a future campaign may add uncertainty.
- The semi-synthetic confounding result depends on the selected retention function. It demonstrates estimator
  behavior under one controlled scenario rather than measuring real observational bias.
- The uplift analysis uses three million rows rather than the full dataset. This retains enough conversions
  for evaluation while keeping Causal Forest training tractable.
- Some decile effects are imprecise. More control observations or repeated experiments would improve power
  for detecting small negative effects.
- Propensity Score Matching is omitted because propensity is nearly constant and matching would discard a
  large amount of information. IPW and AIPW provide more useful adjustment comparisons in this setting.
- R-Learner and DragonNet are not included. The current model set focuses on four meta-learners and Causal
  Forest.

For deployment, the next step is a prospective campaign test using the selected policy and real cost/value
inputs. The main monitoring quantities should be realized treatment coverage, incremental conversion,
profit per contacted customer, and calibration in the highest-ranked groups.

## References

- Diemert, E., Betlei, A., Renaudin, C., and Amini, M. R. (2018). *A Large Scale Benchmark for Uplift Modeling*.
- Robins, J. M., Rotnitzky, A., and Zhao, L. P. (1994). *Estimation of Regression Coefficients When Some Regressors Are Not Always Observed*.
- Radcliffe, N. J. (2007). *Using Control Groups to Target on Predicted Lift*.
- Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). *Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning*.
