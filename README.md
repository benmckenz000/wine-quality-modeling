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


