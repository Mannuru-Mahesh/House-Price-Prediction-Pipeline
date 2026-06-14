House Price Prediction — End-to-End ML Pipeline

An intermediate level data science project that predicts house sale prices using a full machine learning pipeline: EDA → preprocessing → model comparison → hyperparameter tuning → evaluation → explainability (SHAP).

Project Description

This project demonstrates a complete, production-style ML workflow on a realistic housing dataset (1,500 properties, 14 features). Unlike a single notebook with ad-hoc code, everything is wrapped in a reusable scikit-learn `Pipeline`, so the same object handles preprocessing, model inference, and can be saved/loaded with `joblib` for deployment.

The dataset is synthetically generated (see `src/generate_data.py`) using realistic real-estate relationships — square footage, quality scores, neighborhood premiums, age depreciation — plus injected missing values and outliers, so you get hands-on practice with real data-cleaning challenges.



Features

| Stage | What happens |
|---|---|
| EDA | Distribution plots, correlation heatmap, neighborhood comparisons, scatter plots |
| Cleaning | Missing value imputation (median / most-frequent), outlier-aware modeling |
| Preprocessing | `ColumnTransformer` — numeric scaling + categorical one-hot encoding |
| Modeling | Compares Linear Regression, Ridge, Random Forest, Gradient Boosting |
| Tuning | `GridSearchCV` with 5-fold cross-validation on Gradient Boosting |
| Evaluation | RMSE, MAE, R², actual-vs-predicted plot, residuals analysis |
| Explainability | SHAP summary plots — which features drive predictions and why |
| Deployment-ready | Full pipeline saved as a single `.pkl` file via `joblib` |

---

Tech Stack

- Python 3.10
- pandas / numpy — data manipulation
- scikit-learn — pipelines, models, GridSearchCV, metrics
- matplotlib / seaborn — visualization
- SHAP — model explainability
- joblib — model persistence
- Jupyter Notebook — narrative analysis

Folder Structure

house-price-prediction/
├── data/
│   └── housing.csv                  ← Generated dataset (1,500 rows)
├── notebooks/
│   └── House_Price_Prediction.ipynb ← Full narrative walkthrough (pre-run with outputs)
├── src/
│   ├── generate_data.py             ← Creates the synthetic dataset
│   ├── train_pipeline.py            ← Full pipeline: EDA → train → tune → evaluate → SHAP
│   ├── predict.py                   ← Load saved model, predict on new houses
│   └── build_notebook.py            ← Script that generated the .ipynb (for reference)
├── models/
│   ├── house_price_pipeline.pkl     ← Saved trained pipeline (preprocessing + model)
│   └── sample_input.csv             ← Example input row format
├── outputs/                          ← All generated plots (PNG) + result CSVs
│   ├── 01_price_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_by_neighborhood.png
│   ├── 04_area_vs_price.png
│   ├── 05_model_comparison.png
│   ├── 06_actual_vs_predicted.png
│   ├── 07_residuals.png
│   ├── 08_shap_summary.png
│   ├── model_comparison.csv
│   └── shap_feature_importance.csv
├── requirements.txt
└── README.md

How to Run Locally

1. Clone and set up environment


git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction

Create a virtual environment 
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

2. Generate the dataset


python src/generate_data.py

This creates `data/housing.csv` (1,500 rows, reproducible via random seed).

3. Run the full training pipeline

python src/train_pipeline.py

This will:
- Run EDA and save 8 plots to `outputs/`
- Train and compare 4 models
- Tune the best model with GridSearchCV
- Evaluate on the test set
- Generate SHAP explainability plots
- Save the final model to `models/house_price_pipeline.pkl`

**Expected runtime:** ~30–60 seconds on a laptop.

4. Make predictions on new houses


python src/predict.py

Edit the `new_houses` list inside `predict.py` to try your own property details.

5. (Optional) Explore the notebook

jupyter notebook notebooks/House_Price_Prediction.ipynb

The notebook is already pre-executed with all outputs, so you can read it directly on GitHub without running anything.

Results Summary

| Metric | Value |
|---|---|
| Dataset size | 1,500 houses, 14 features |
| Best model | Gradient Boosting (tuned via GridSearchCV) |
| Best params | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3` |
| Test R² | 0.872 |
| Test RMSE | ~$31,850 |
| Test MAE | ~$24,281 |

In plain English:the model explains about **87% of the variation in house prices, with an average prediction error of roughly $24,000 — on houses ranging from $70K to $870K.

Top features driving predictions (via SHAP)
1. `area_sqft` — square footage (by far the strongest driver)
2. `house_style_Single_Family` — single-family homes carry a premium
3. `neighborhood_Downtown` — location premium
4. `overall_qual` — quality score (1–10)
5. `neighborhood_Rural` / `neighborhood_Lakeside` — location effects

---

Sample Visualizations

| Plot | Description |
|---|---|
| `01_price_distribution.png` | Right-skewed price distribution + outlier boxplot |
| `02_correlation_heatmap.png` | Correlation of all numeric features with price |
| `03_price_by_neighborhood.png` | Price ranges across 8 neighborhoods |
| `04_area_vs_price.png` | Area vs price, colored by quality score |
| `05_model_comparison.png` | R² comparison across 4 models |
| `06_actual_vs_predicted.png` | Final model accuracy visualization |
| `07_residuals.png` | Error distribution (checks for bias) |
| `08_shap_summary.png` | Global feature importance via SHAP |

---

How The Project Works (For Class / Interview)

The data story:
1. `generate_data.py` creates 1,500 synthetic houses using a price formula based on area, quality, location, age, and amenities — then deliberately adds missing values and outliers to mimic real datasets.

2. EDA reveals that `area_sqft` and `overall_qual` correlate most strongly with price, and that location (`neighborhood`) creates clear price tiers — this guides what features matter.

3. Train/test split happens BEFORE preprocessing this is critical to avoid *data leakage* (the model must never see test data during fitting, including during imputation/scaling).

4. `ColumnTransformer` routes numeric columns through median-imputation + scaling, and categorical columns through most-frequent-imputation + one-hot encoding — all wrapped in one `Pipeline` object.

5. Four models are compared on identical preprocessing to find the best baseline algorithm family for this data.

6. `GridSearchCV` then searches over hyperparameter combinations using 5-fold cross-validation, picking the configuration with the best average R².

7. Evaluation uses RMSE (penalizes large errors), MAE (average absolute error, easy to interpret), and R² (% of variance explained) — plus visual diagnostics (actual vs predicted, residuals).

8. SHAP explains *individual* predictions by showing how much each feature pushed the prediction up or down from the baseline average — crucial for trust and debugging.

9. `joblib.dump()` saves the entire pipeline as one file, so `predict.py` can load it and call `.predict()` on raw new data with zero manual preprocessing.

Key interview talking point:"I built a leakage safe pipeline where preprocessing and modeling are inseparable this means the exact same transformations applied during training are guaranteed to apply during inference, which is a common production bug source when preprocessing is done manually outside a pipeline."*

What I Learned

- How to structure an ML project with reusable `Pipeline` + `ColumnTransformer` objects
- Why train/test split must happen before any preprocessing (avoiding data leakage)
- How to compare multiple algorithms fairly using identical preprocessing
- How `GridSearchCV` + cross-validation finds robust hyperparameters
- How to choose and interpret regression metrics (RMSE vs MAE vs R²)
- How to use **SHAP** to explain *why* a model makes specific predictions
- How to persist a full pipeline with `joblib` for reuse without re-training

Future Improvements

- [ ] Try XGBoost or LightGBM for potentially stronger performance
- [ ] Add engineered features (e.g., `price_per_sqft`, `age × quality` interaction)
- [ ] Use `RandomizedSearchCV` or Optuna for faster/larger hyperparameter search
- [ ] Add geographic coordinates and spatial clustering features
- [ ] Build a simple web form (Streamlit or a static HTML + serialized model via ONNX/TF.js) to serve live predictions
- [ ] Add unit tests for the preprocessing pipeline
- [ ] Experiment with log-transforming `sale_price` to handle skewness


License

This project is open source under the MIT License. The dataset is synthetically generated and contains no real property data.


Built as an intermediate data science portfolio project 2026
