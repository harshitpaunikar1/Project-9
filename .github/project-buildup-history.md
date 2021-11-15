# Project Buildup History: Linear Regression Cross Validation

- Repository: `linear-regression-cross-validation`
- Category: `data_science`
- Subtype: `generic`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2021-10-04 - Day 2: Dataset preparation

- Task summary: Started the Linear Regression Cross Validation project properly today after a rough start earlier in the year. The dataset needed more care than expected — the raw file had inconsistent decimal formatting in a couple columns depending on which rows you looked at, some using period and some using comma. Fixed the parsing step to handle both. Also decided to use a public housing dataset for this one since it has a good mix of continuous and categorical predictors and a well-understood target.
- Deliverable: Dataset loaded cleanly. Decimal format issue resolved. Housing data confirmed as good fit for this study.
## 2021-10-11 - Day 3: Baseline regression

- Task summary: Implemented the OLS baseline for the cross-validation study today. Set up a clean train/test split first and trained a plain linear regression. The RMSE was acceptable but the residual plot showed clear heteroscedasticity — the errors grew with the predicted value. That is worth noting as a limitation but not blocking the cross-validation comparison work. Started setting up the k-fold structure and got through 5-fold working with the baseline coefficients logged per fold.
- Deliverable: Baseline OLS done. Heteroscedasticity noted. 5-fold scaffold up.
## 2021-10-11 - Day 3: Baseline regression

- Task summary: Spent an extra hour adding a Ridge and Lasso baseline alongside the OLS so the cross-validation comparison would be more meaningful across regularization strength.
- Deliverable: Ridge and Lasso baselines added to comparison set.
## 2021-10-11 - Day 3: Baseline regression

- Task summary: Late-night fix: caught that the categorical features were being passed as-is to the linear model without encoding. Added one-hot encoding into the fold loop properly.
- Deliverable: One-hot encoding now inside the fold loop. No more dtype warnings.
## 2021-11-15 - Day 5: Cross-validation result analysis

- Task summary: Spent today analyzing the cross-validation results for the linear regression study. The mean and standard deviation of RMSE across folds were reasonable but there was one fold that consistently performed worse than the others. Investigated and found that fold happened to include a cluster of high-value outlier properties that the model underpredicts systematically. Added a note about this in the analysis and ran the same study with stratified splits to see if it helped.
- Deliverable: Fold instability traced to outlier cluster. Stratified splits reduce variance across folds.
