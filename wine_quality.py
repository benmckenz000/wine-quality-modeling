import pandas as pd
import numpy as np
import xgboost as xgb
import mord as m
import matplotlib.pyplot as plt
import seaborn as sns
import shap
sns.set_theme(style="whitegrid")
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Data Exploration and Visual Analysis

df = pd.read_csv("data/winequality-red.csv", sep=";")
print("Shape:", df.shape)
print(df.isnull().sum())

plt.figure(figsize=(8, 4))
sns.countplot(x="quality", data=df)
plt.title("Quality Scores Distribution")
plt.show()

df.hist(bins=15, figsize=(15, 10), layout=(4, 3))
plt.suptitle("Feature Distribution")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 2. Statistical Diagnostics: Multicollinearity and Preprocessing

X = df.drop("quality", axis=1)
y = df["quality"]

# Adding a constant (intercept) for statsmodels VIF calc
X_vif = X.copy()
X_vif["intercept"] = 1

vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
vif_results = vif_data[vif_data["Feature"] != "intercept"].sort_values(by="VIF", ascending=False)
print(vif_results)

# Stratified split preserves class proportions for less common quality scores (3 and 8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# 3. Parametric Modeling: Linear and Ordinal Regression

# Linear Regression 
# StandardScaler applied inside Pipeline to prevent data leakage during CV
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
predict = lr_pipeline.predict(X_test)

lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=skf, scoring="r2")
print(f"Linear Regression | Test R²: {r2_score(y_test, predict):.3f} | CV R²: {np.mean(lr_cv_scores):.3f}")

residuals = y_test - predict
plt.figure(figsize=(8, 5))
plt.scatter(predict, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Quality")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values - Linear Regression")
plt.show()

# Ordinal Regression
# LogisticAT treats quality as ordered category. CV scored on MAE rather than Rsquared
ord_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", m.LogisticAT(alpha=1.0))
])

ord_pipeline.fit(X_train, y_train)
ord_preds = ord_pipeline.predict(X_test)

ord_cv_scores = cross_val_score(ord_pipeline, X, y, cv=skf, scoring="neg_mean_absolute_error")
print(f"Ordinal Regression | Accuracy: {accuracy_score(y_test, ord_preds):.3f} | Test MAE: {mean_absolute_error(y_test, ord_preds):.3f} | CV MAE: {-np.mean(ord_cv_scores):.3f}")

ord_residual = y_test - ord_preds
plt.figure(figsize=(8, 5))
plt.scatter(ord_preds, ord_residual)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Quality (Ordinal)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values - Ordinal Regression")
plt.show()

# 4. Non-Parametric Ensemble Models: Random Forest and XGBoost

# Random Forest 

# Baseline
rf = RandomForestRegressor(random_state=123)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="r2")
print(f"RF Baseline | Test R²: {r2_score(y_test, rf_preds):.3f} | CV R²: {np.mean(rf_cv_scores):.3f}")

# Tuned
rf_base = RandomForestRegressor(random_state=123)
param_dist_rf = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4, 6, 8]
}
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist_rf,
    n_iter=50,
    cv=skf,
    scoring="r2",
    random_state=123,
    n_jobs=-1,
    verbose=1
)
rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

rf_tuned_cv_scores = cross_val_score(best_rf, X, y, cv=skf, scoring="r2", n_jobs=-1)
rf_tuned_preds = best_rf.predict(X_test)

print(f"RF Tuned    | Test R²: {r2_score(y_test, rf_tuned_preds):.3f} | CV R²: {np.mean(rf_tuned_cv_scores):.3f} (±{np.std(rf_tuned_cv_scores):.3f}) | Improvement: {np.mean(rf_tuned_cv_scores) - np.mean(rf_cv_scores):.3f}")
print(f"Best RF Params: {rf_random.best_params_}")

# XGBoost 

# Baseline
xgb_baseline = xgb.XGBRegressor(objective="reg:squarederror", random_state=123)
xgb_baseline.fit(X_train, y_train)
xgb_baseline_preds = xgb_baseline.predict(X_test)
xgb_baseline_scores = cross_val_score(xgb_baseline, X, y, cv=skf, scoring="r2")
print(f"XGB Baseline | Test R²: {r2_score(y_test, xgb_baseline_preds):.3f} | CV R²: {np.mean(xgb_baseline_scores):.3f}")

# Tuned
xgb_base = xgb.XGBRegressor(objective="reg:squarederror", random_state=123)
param_dist_xgb = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
}
xgb_random = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist_xgb,
    n_iter=50,
    cv=skf,
    scoring="r2",
    random_state=123,
    n_jobs=-1,
    verbose=1
)
xgb_random.fit(X_train, y_train)
best_xgb = xgb_random.best_estimator_

xgb_tuned_cv_scores = cross_val_score(best_xgb, X, y, cv=skf, scoring="r2", n_jobs=-1)
xgb_tuned_preds = best_xgb.predict(X_test)
xgb_tuned_cv_mean = np.mean(xgb_tuned_cv_scores)
xgb_tuned_cv_std = np.std(xgb_tuned_cv_scores)

print(f"XGB Tuned    | Test R²: {r2_score(y_test, xgb_tuned_preds):.3f} | CV R²: {xgb_tuned_cv_mean:.3f} (±{xgb_tuned_cv_std:.3f}) | Improvement: {xgb_tuned_cv_mean - np.mean(xgb_baseline_scores):.3f}")
print(f"Best XGB Params: {xgb_random.best_params_}")

# 5. Feature Importance and Interpretability

# Feature Importance plots
xgb.plot_importance(best_xgb, height=0.5, importance_type="gain", show_values=False)
plt.title("XGBoost Feature Importance (Gain)")
plt.show()

rf_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
rf_importances.sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# SHAP Summary Plots for feature influence direction
shap_models = {
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

for name, model in shap_models.items():
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"{name} SHAP Summary")
    plt.tight_layout()
    plt.show()

# 6. Robustness Testing: Controlled Distribution Shift

# Split on alcohol median (top feature for both models)
median_alcohol = df["alcohol"].median()
train_shift = df[df["alcohol"] <= median_alcohol]
test_shift = df[df["alcohol"] > median_alcohol]

X_train_shift = train_shift.drop("quality", axis=1)
y_train_shift = train_shift["quality"]
X_test_shift = test_shift.drop("quality", axis=1)
y_test_shift = test_shift["quality"]

# Diagnostic check for target drift
print(f"Mean Quality (Low Alcohol Train): {y_train_shift.mean():.3f}")
print(f"Mean Quality (High Alcohol Test): {y_test_shift.mean():.3f}")

plt.figure(figsize=(8, 4))
sns.kdeplot(y_train_shift, label="Train Quality (Low Alcohol)")
sns.kdeplot(y_test_shift, label="Test Quality (High Alcohol)")
plt.title("Target Distribution Shift Check")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
sns.kdeplot(train_shift["alcohol"], label="Train (Low Alcohol)")
sns.kdeplot(test_shift["alcohol"], label="Test (High Alcohol)")
plt.title("Alcohol Distribution Under Controlled Shift")
plt.xlabel("Alcohol")
plt.ylabel("Density")
plt.legend()
plt.show()

# Evaluate both tuned models under shift
shift_models = {
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

shift_results = {}
for name, model in shift_models.items():
    model.fit(X_train_shift, y_train_shift)
    preds = model.predict(X_test_shift)
    shift_results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test_shift, preds)),
        "MAE": mean_absolute_error(y_test_shift, preds),
        "R2": r2_score(y_test_shift, preds)
    }
    print(f"{name} under Distribution Shift | RMSE: {shift_results[name]['RMSE']:.3f} | MAE: {shift_results[name]['MAE']:.3f} | R²: {shift_results[name]['R2']:.3f}")

rf_shift_r2 = shift_results["Random Forest"]["R2"]
xgb_shift_r2 = shift_results["XGBoost"]["R2"]

rf_original_r2 = r2_score(y_test, rf_tuned_preds)
xgb_original_r2 = r2_score(y_test, xgb_tuned_preds)

comparison_df = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Original R²": [rf_original_r2, xgb_original_r2],
    "Shifted R²": [rf_shift_r2, xgb_shift_r2],
    "R² Drop": [rf_original_r2 - rf_shift_r2, xgb_original_r2 - xgb_shift_r2],
})
print(comparison_df.to_string(index=False))

# 7. Final Model Comparison and Summary

summary_df = pd.DataFrame({
    "Model": ["Linear Regression", "Ordinal Regression", "Random Forest (Tuned)", "XGBoost (Tuned)"],

    "Test RMSE": [
        np.round(np.sqrt(mean_squared_error(y_test, predict)), 3),
        "-",
        np.round(np.sqrt(mean_squared_error(y_test, rf_tuned_preds)), 3),
        np.round(np.sqrt(mean_squared_error(y_test, xgb_tuned_preds)), 3)
    ],
    "Test MAE": [
        np.round(mean_absolute_error(y_test, predict), 3),
        np.round(mean_absolute_error(y_test, ord_preds), 3),
        np.round(mean_absolute_error(y_test, rf_tuned_preds), 3),
        np.round(mean_absolute_error(y_test, xgb_tuned_preds), 3)
    ],
    "Test R²": [
        np.round(r2_score(y_test, predict), 3),
        "-",
        np.round(r2_score(y_test, rf_tuned_preds), 3),
        np.round(r2_score(y_test, xgb_tuned_preds), 3)
    ],
    "CV R² Mean": [
         f"{np.mean(lr_cv_scores):.3f} (±{np.std(lr_cv_scores):.4f})",
        "-",
        f"{np.mean(rf_tuned_cv_scores):.3f} (±{np.std(rf_tuned_cv_scores):.4f})",
        f"{xgb_tuned_cv_mean:.3f} (±{xgb_tuned_cv_std:.4f})"
    ],
    "CV MAE Mean": [
        "-",
        np.round(-np.mean(ord_cv_scores), 3),
        "-",
        "-"
    ],
    "R² Drop (Shift)": [
        "-",
        "-",
        np.round(r2_score(y_test, rf_tuned_preds) - rf_shift_r2, 3),
        np.round(r2_score(y_test, xgb_tuned_preds) - xgb_shift_r2, 3)
    ]
})
print(summary_df.to_string(index=False))
