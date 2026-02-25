# Comparative Statistical and ML Analysis of Red Wine Quality
Exploratory analysis and predictive modeling of wine quality using linear, ordinal, tree-based, and XGBoost models with cross-validation and SHAP interpretation.

## Introduction

This project investigates the relationship between the physicochemical properties of red wine and quality ratings using a combination of parametric statistical models and non-parametric machine learning methods. The primary goal of this analysis is to determine if measurable chemical attributes can reliably predict how a wine is rated by human testers.

Predicting wine quality is complex because of the nature of the target variable. Quality scores are discrete, ordered integers that are heavily concentrated around scores 5 and 6. This distribution indicates that the data doesn't fit into a standard regression or classification framework. This analysis evaluates four modeling approaches suited to this structure, compares their performance using appropriate metrics, and stress-tests the two most strongest models with a controlled covariate shift to evaluate their predictive stability.

---

## Dataset

**Source:** UCI Machine Learning Repository — [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality), Cortez et al. (2009).

The dataset contains 1,599 red wine samples. Each sample is described by 11 chemical features including alcohol content, volatile acidity, and sulphates. A key challenge in this project was addressing the significant class imbalance. Most wines are rated as 5 or 6, while scores at the extremes (3, 4, 7, and 8) are sparsely represented. I accounted for this distribution explicitly in the preprocessing and validation section.

<img width="699" height="396" alt="image" src="https://github.com/user-attachments/assets/ea67bfcb-42b1-41ab-b4aa-da4dbdce8cf3" />

---

## Exploratory Data Analysis

During initial data exploration, I observed that several features exhibit a meaningful right skew and contain notable outliers. This is particularly evident in [------]. I retained these outliers as they represent real variations in wine chemistry. Additionally, the tree-based models used are naturally robust to these distributions.

<img width="1229" height="917" alt="image" src="https://github.com/user-attachments/assets/7ae419be-7169-440c-b727-acd50466ef2b" />

The correlation analysis shows that alcohol has the strongest postive relationship with quality. Conversely, volatile acidity has the strongest negative correlation. Also noted are moderate positive associations with citric acid and sulphates. 

<img width="1021" height="792" alt="image" src="https://github.com/user-attachments/assets/74381670-2184-407f-95f3-b2d84ca7ecb1" />

---

## Statistical Diagnostics and Methodology

### Multicollinearity (VIF)

Before splitting the data, Variance Inflation Factors (VIF) were calculated across all features to identify potential multicollinearity. To ensure accuracy of the diagnostics, I added a constant intercept to the design. The results showed elevated VIF scores for features like fixed acidity and density, which indicates that these variables are highly correlated. These dependencies suggest that linear regression coefficents should be interpreted cautiously, as the estimates will be unstable and standard errors will be inflated. For the ensemble models used later, multicollinearity is not a concern due to tree-based models splitting on individual features resulting in them being unaffected by linear dependencies among predictors. 

<img width="364" height="348" alt="image" src="https://github.com/user-attachments/assets/df9e0b4f-135d-4db5-abca-f67322d1f1f8" />

### Modeling Ordinality

Wine quality is measured on an ordered integer scale (3–8). Treating it as a purely continuous ignores its discrete structure, while treating it as unordered categories loses the established ranking. Ordinal Regression is included as a methodologically sound middle ground. By treating quality as a series of thresholds, this model aligns more closely with the ordered structure without forcing a continuous prediction.

---

## Modeling Methodology

### Preprocessing 

StandardScaler was utilized within a Scikit-Learn Pipeline for the parametric models. This ensured that the scaling was fit exclusively on the training data during cross-validation, which prevented the risk of data leakage. Tree-based models (Random Forest, XGBoost) do not require scaling and were trained directly on raw features. 

To handle the class imbalance, a Stratified K-fold cross-validation was employed to ensure that every fold contained a representation of the less common quality scores. The same stratification logic was applied to the train/test split via `stratify=y`.

### Validation 

To handle the class imbalance, a Stratified K-fold cross-validation was employed to ensure that every fold contained a representation of the less common quality scores. The same stratification logic was applied to the train/test split via `stratify=y`.

### Models

| Model | Type | Rationale |
|---|---|---|
| Linear Regression | Parametric, continuous | Interpretable baseline and establishes the linear ceiling |
| Ordinal Regression (`mord` LogisticAT) | Parametric, ordinal | Methodologically suited to ordered integer targets |
| Random Forest | Non-parametric ensemble | Handles non-linearity and multicollinearity |
| XGBoost | Gradient boosted ensemble | Regularized boosting and strong generalization performance |

---

## Model Evaluation

### Performance Summary

| Model | Test RMSE | Test MAE | Test R² | CV R² Mean | CV MAE Mean | R² Drop (Shift) |
|---|---|---|---|---|---|---|
| Linear Regression | 0.671 | 0.517 | 0.303 | 0.348 | — | — |
| Ordinal Regression | — | 0.459 | — | — | 0.437 | — |
| Random Forest | 0.589 | 0.412 | 0.463 | 0.5 | — | 0.461 |
| XGBoost | 0.601 | 0.390 | 0.441 | 0.478 | — | 0.507 |

**Key findings:**

- **Linear Regression** achieves an R² of approximately 0.35 in cross-validation. This confirms that while wine quality contains meaningful signal, it also contains a non-linear structure that a linear model cannot fully capture. The residuals versus fitted plot reveals clear bias and heteroscedasticity, which suggests systematic model misspecification.

<img width="691" height="473" alt="image" src="https://github.com/user-attachments/assets/4df38aad-a012-414e-9a58-e2642127a66e" />

- **Ordinal Regression** achieves a lower Mean Absolute Error (MAE) than the standard linear model, suggesting it better respects the ordered distance between quality categories. Because this is a classification-based approach, R² isn't calculated. Instead, the analysis focuses on MAE as a better match for the target structure. 

<img width="691" height="473" alt="image" src="https://github.com/user-attachments/assets/9583f961-c5db-4174-a84b-84d35ed75e70" />

- **Random Forest** outperforms linear models on all regression metrics, with an R² of 0.50 and the lowest RMSE (0.589). Cross-validation scores track closely with test scores, indicating no significant overfitting in the model.

- **XGBoost** achieves a slightly lower R² than Random Forest but produced a lower Test MAE of 0.390. This result implies that XGBoost predictions tend to land closer to true scores on average. The inclusion of L1 and L2 regularization maintains stable generalization, and a `max_depth` of 5 was chosen to balance model complexity against the risk of overfitting.

Both tree-based models were further evaluated using `RandomizedSearchCV` to determine whether default hyperparameters were already near-optimal. The Random Forest model saw a marginal improvement in cross-validation R², moving from 0.496 to 0.500 (+0.004). Similarly, the XGBoost model saw an improvement from a baseline R² of 0.459 to 0.478 (+0.019). 



---

## Robustness & Stress Testing

### Experimental Design

To test model robustness under a realistic covariate shift, models were retrained on wines with **alcohol ≤ median** and evaluated on wines with **alcohol > median**. Alcohol was selected as the split variable because it is the single most predictive feature across all models and importance methods — making it the most meaningful axis on which to stress-test generalization.

> 📊 *Figure: Alcohol Distribution — Train vs. Test Under Shift*

### Target Drift Diagnostic

Before interpreting shift performance, it is essential to check whether the quality distribution itself differs between groups — if it does, R² collapse may reflect **target drift** (a mismatch in the label distribution) rather than pure model fragility.

The mean quality in the low-alcohol training group is **5.324**, compared to **5.983** in the high-alcohol test group. This ~0.66 point difference is meaningful: the models were asked to predict a quality range they had limited exposure to during training. The R² drop observed under the shift is therefore attributable to a combination of covariate shift *and* target drift — and should not be interpreted as model fragility alone.

> 📊 *Figure: Quality Distribution — Train vs. Test Under Shift (KDE)*

### Results

| Model | Original R² | Shifted R² | R² Drop |
|---|---|---|---|
| Random Forest | — | — | — |
| XGBoost | — | — | — |

Both models show substantial R² degradation under the shift. This is expected given alcohol's dominant predictive weight — when the alcohol distribution changes, the learned associations shift accordingly. In a production context, this would motivate periodic model retraining or recalibration as the underlying wine population evolves.

---

## Feature Importance & Interpretability

### Built-in Feature Importance

Both Random Forest (impurity-based) and XGBoost (information gain) provide native feature importance scores. **Alcohol and volatile acidity are the top-ranked features in both models**, with sulphates and sulphur dioxide as secondary contributors.

> 📊 *Figure: Random Forest Feature Importance (bar chart)*
> 📊 *Figure: XGBoost Feature Importance — Gain*

### SHAP Analysis

Built-in importance scores indicate *which* features matter but not *how* they affect predictions. SHAP (SHapley Additive exPlanations) values address this by decomposing each prediction into signed feature contributions.

> 📊 *Figure: SHAP Summary Plot — Random Forest*
> 📊 *Figure: SHAP Summary Plot — XGBoost*

Key interpretive findings consistent across both models:
- **Alcohol:** Strong positive effect — higher alcohol content pushes predicted quality up
- **Volatile acidity:** Strong negative effect — higher volatile acidity (associated with vinegar notes) pulls quality down
- **Sulphates:** Moderate positive effect

The consistency of these findings across two different model families and two different importance methods strengthens confidence that these are genuine data patterns rather than modeling artifacts.

---

## Limitations & Future Work

- **Moderate predictive ceiling:** The best R² achieved is approximately 0.50. Quality ratings are assigned by human tasters and carry inherent subjectivity, creating irreducible noise that no model can overcome.
- **Distribution shift conflates two effects:** The alcohol-split design introduces both covariate shift and target drift simultaneously. A cleaner robustness test would hold the quality distribution constant while shifting only the feature distribution.
- **Single wine type:** The dataset contains only red wines. Whether these models generalize to white wine is untested and unlikely without retraining.
- **Feature engineering:** No interaction terms, polynomial features, or domain-derived features were explored. Ratios like free/total sulfur dioxide or acid balance may carry additional signal.
- **Classification framing:** Grouping quality scores into low (3–4), medium (5–6), and high (7–8) categories and applying classification models may yield better-calibrated predictions for rare scores.

---

## Reproducibility

**Key libraries:** pandas, numpy, scikit-learn, xgboost, mord, shap, matplotlib, seaborn, statsmodels

All random operations use `random_state=123`. Dataset available from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

```bash
pip install pandas numpy scikit-learn xgboost mord shap matplotlib seaborn statsmodels
```

---

## Reference

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4):547–553, 2009.
