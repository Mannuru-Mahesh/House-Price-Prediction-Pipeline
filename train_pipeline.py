"""
train_pipeline.py
-------------------
End-to-end ML pipeline for House Price Prediction.

Steps:
  1. Load data
  2. Exploratory Data Analysis (save plots to outputs/)
  3. Train/test split
  4. Build preprocessing pipeline (impute, scale, one-hot encode)
  5. Train multiple models (Linear Regression, Random Forest, Gradient Boosting)
  6. Hyperparameter tuning with GridSearchCV on the best candidate
  7. Evaluate with RMSE, MAE, R2
  8. Explain predictions with SHAP
  9. Save the final trained pipeline with joblib

Run with:  python src/train_pipeline.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import shap

sns.set_style('whitegrid')
pd.set_option('display.width', 120)

# =================================================================
# 1. LOAD DATA
# =================================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv('data/housing.csv')
print(f"Shape: {df.shape}")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nData types:\n{df.dtypes}")


# =================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =================================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# --- 2a. Target variable distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df['sale_price'], kde=True, ax=axes[0], color='#f5a623')
axes[0].set_title('Sale Price Distribution')
axes[0].set_xlabel('Sale Price ($)')

sns.boxplot(x=df['sale_price'], ax=axes[1], color='#6c9ef8')
axes[1].set_title('Sale Price — Outlier Check')
axes[1].set_xlabel('Sale Price ($)')
plt.tight_layout()
plt.savefig('outputs/01_price_distribution.png', dpi=120)
plt.close()
print("Saved: outputs/01_price_distribution.png")

# --- 2b. Correlation heatmap (numeric features only) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap (Numeric Features)')
plt.tight_layout()
plt.savefig('outputs/02_correlation_heatmap.png', dpi=120)
plt.close()
print("Saved: outputs/02_correlation_heatmap.png")
print(f"\nTop correlations with sale_price:")
print(corr['sale_price'].sort_values(ascending=False).head(8))

# --- 2c. Price by neighborhood ---
plt.figure(figsize=(10, 5))
order = df.groupby('neighborhood')['sale_price'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='neighborhood', y='sale_price', order=order, palette='Set2')
plt.title('Sale Price by Neighborhood')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/03_price_by_neighborhood.png', dpi=120)
plt.close()
print("Saved: outputs/03_price_by_neighborhood.png")

# --- 2d. Area vs Price scatter, colored by quality ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['area_sqft'], df['sale_price'],
                       c=df['overall_qual'], cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, label='Overall Quality (1-10)')
plt.xlabel('Area (sqft)')
plt.ylabel('Sale Price ($)')
plt.title('Area vs Price (colored by Overall Quality)')
plt.tight_layout()
plt.savefig('outputs/04_area_vs_price.png', dpi=120)
plt.close()
print("Saved: outputs/04_area_vs_price.png")


# =================================================================
# 3. TRAIN / TEST SPLIT
# =================================================================
print("\n" + "=" * 60)
print("STEP 3: Train/Test Split")
print("=" * 60)

X = df.drop(columns=['sale_price'])
y = df['sale_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {X_train.shape[0]} rows")
print(f"Test set:  {X_test.shape[0]} rows")


# =================================================================
# 4. PREPROCESSING PIPELINE
#
#    Numeric columns -> impute missing values with median -> scale
#    Categorical cols -> impute missing with most frequent -> one-hot encode
#
#    ColumnTransformer applies different preprocessing to different
#    column types, all wrapped in ONE pipeline object that can be
#    reused identically at training and prediction time.
# =================================================================
print("\n" + "=" * 60)
print("STEP 4: Building Preprocessing Pipeline")
print("=" * 60)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# =================================================================
# 5. TRAIN MULTIPLE MODELS & COMPARE
# =================================================================
print("\n" + "=" * 60)
print("STEP 5: Training & Comparing Models")
print("=" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Random Forest':     RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    print(f"{name:20s} | RMSE: ${rmse:>10,.0f} | MAE: ${mae:>10,.0f} | R²: {r2:.4f}")

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
results_df.to_csv('outputs/model_comparison.csv', index=False)

# Plot model comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Model', y='R2', palette='viridis')
plt.title('Model Comparison — R² Score (higher = better)')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('outputs/05_model_comparison.png', dpi=120)
plt.close()
print("\nSaved: outputs/05_model_comparison.png")
print("Saved: outputs/model_comparison.csv")

best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 Best model: {best_model_name}")


# =================================================================
# 6. HYPERPARAMETER TUNING (on Random Forest or Gradient Boosting)
#    GridSearchCV tries every combination of parameters and picks
#    the one with the best cross-validated score.
# =================================================================
print("\n" + "=" * 60)
print("STEP 6: Hyperparameter Tuning (GridSearchCV)")
print("=" * 60)

# We tune Gradient Boosting — typically the strongest model on
# structured/tabular data like this.
tuning_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators':    [100, 200],
    'regressor__learning_rate':   [0.05, 0.1],
    'regressor__max_depth':       [2, 3, 4],
}

grid_search = GridSearchCV(
    tuning_pipe,
    param_grid=param_grid,
    cv=5,                       # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_


# =================================================================
# 7. FINAL EVALUATION ON TEST SET
# =================================================================
print("\n" + "=" * 60)
print("STEP 7: Final Evaluation on Test Set")
print("=" * 60)

final_preds = best_pipeline.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_mae  = mean_absolute_error(y_test, final_preds)
final_r2   = r2_score(y_test, final_preds)

print(f"Final RMSE: ${final_rmse:,.0f}")
print(f"Final MAE:  ${final_mae:,.0f}")
print(f"Final R²:   {final_r2:.4f}")
print(f"\nInterpretation: The model explains {final_r2*100:.1f}% of the")
print(f"variance in house prices, with average error of ${final_mae:,.0f}.")

# Actual vs Predicted scatter plot
plt.figure(figsize=(7, 7))
plt.scatter(y_test, final_preds, alpha=0.4, color='#f5a623', edgecolor='none')
lims = [min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())]
plt.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction')
plt.xlabel('Actual Sale Price ($)')
plt.ylabel('Predicted Sale Price ($)')
plt.title(f'Actual vs Predicted Prices (R² = {final_r2:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/06_actual_vs_predicted.png', dpi=120)
plt.close()
print("\nSaved: outputs/06_actual_vs_predicted.png")

# Residuals plot
residuals = y_test - final_preds
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='#e05c5c')
plt.axvline(0, color='black', linestyle='--')
plt.title('Residuals Distribution (Prediction Errors)')
plt.xlabel('Actual - Predicted ($)')
plt.tight_layout()
plt.savefig('outputs/07_residuals.png', dpi=120)
plt.close()
print("Saved: outputs/07_residuals.png")


# =================================================================
# 8. MODEL EXPLAINABILITY WITH SHAP
#    SHAP values show how much each feature pushed an individual
#    prediction up or down from the average — crucial for
#    explaining "why" the model made a decision (important in
#    interviews and real-world stakeholder communication).
# =================================================================
print("\n" + "=" * 60)
print("STEP 8: Model Explainability (SHAP)")
print("=" * 60)

# Transform the test data through the preprocessor to get the
# actual numeric matrix the model sees
X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)

# Get feature names after one-hot encoding
feature_names = (
    numeric_features +
    list(best_pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)

# SHAP TreeExplainer works efficiently with tree-based models
explainer = shap.TreeExplainer(best_pipeline.named_steps['regressor'])

# Use a sample for speed (SHAP can be slow on large datasets)
sample_idx = np.random.choice(X_test_transformed.shape[0],
                               size=min(200, X_test_transformed.shape[0]),
                               replace=False)
X_sample = X_test_transformed[sample_idx]
if hasattr(X_sample, 'toarray'):  # handle sparse matrix
    X_sample = X_sample.toarray()

shap_values = explainer.shap_values(X_sample)

# Summary plot — global feature importance
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                   show=False, max_display=10)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('outputs/08_shap_summary.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: outputs/08_shap_summary.png")

# Print top features by mean absolute SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 10 most important features (by SHAP value):")
print(shap_importance.head(10).to_string(index=False))
shap_importance.to_csv('outputs/shap_feature_importance.csv', index=False)


# =================================================================
# 9. SAVE THE FINAL PIPELINE
#    We save the ENTIRE pipeline (preprocessing + model) as one
#    object so predictions on new raw data "just work" with
#    .predict() — no manual preprocessing needed at inference time.
# =================================================================
print("\n" + "=" * 60)
print("STEP 9: Saving Final Model")
print("=" * 60)

joblib.dump(best_pipeline, 'models/house_price_pipeline.pkl')
print("Saved: models/house_price_pipeline.pkl")

# Also save feature names + sample input for reference
sample_input = X_test.iloc[[0]]
sample_input.to_csv('models/sample_input.csv', index=False)
print("Saved: models/sample_input.csv (example input format)")

print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETE")
print("=" * 60)
print(f"""
Summary:
  - Dataset:        {len(df)} houses, {len(X.columns)} features
  - Best model:     Gradient Boosting (tuned)
  - Test R²:        {final_r2:.4f}
  - Test RMSE:      ${final_rmse:,.0f}
  - Test MAE:       ${final_mae:,.0f}
  - Saved model:    models/house_price_pipeline.pkl
  - All plots in:   outputs/
""")
