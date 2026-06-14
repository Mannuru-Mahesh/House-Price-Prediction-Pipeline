"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC housing dataset for the
House Price Prediction project.

We create this ourselves (instead of downloading a real dataset)
so that:
  - The project works fully offline / without internet
  - There are no licensing issues
  - We can control feature relationships to make the ML story clear

The dataset mimics common real-estate datasets (like Ames Housing
or Kaggle's House Prices) with both numeric and categorical features,
some missing values, and a few outliers — so students practice
real data-cleaning steps.
"""

import numpy as np
import pandas as pd

# Set seed so the dataset is reproducible every time this script runs
np.random.seed(42)

N = 1500  # number of houses

# ---------------------------------------------------------------
# 1. NUMERIC FEATURES
# ---------------------------------------------------------------
area_sqft     = np.random.normal(1800, 600, N).clip(400, 6000)
bedrooms      = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.05, 0.20, 0.35, 0.25, 0.10, 0.05])
bathrooms     = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], N,
                                  p=[0.10, 0.10, 0.30, 0.20, 0.15, 0.10, 0.05])
year_built    = np.random.randint(1950, 2024, N)
garage_spaces = np.random.choice([0, 1, 2, 3], N, p=[0.15, 0.30, 0.45, 0.10])
lot_size      = np.random.normal(7000, 2500, N).clip(1000, 25000)
overall_qual  = np.random.choice(range(1, 11), N,
                                  p=[0.01,0.02,0.04,0.07,0.15,0.20,0.20,0.15,0.10,0.06])  # 1-10 quality score
distance_to_city_km = np.random.exponential(8, N).clip(0.5, 60)

# ---------------------------------------------------------------
# 2. CATEGORICAL FEATURES
# ---------------------------------------------------------------
neighborhoods = np.random.choice(
    ['Downtown', 'Suburb_North', 'Suburb_South', 'Suburb_East',
     'Suburb_West', 'Rural', 'Lakeside', 'Hillside'],
    N, p=[0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.15, 0.10]
)

house_style = np.random.choice(
    ['Single_Family', 'Townhouse', 'Condo', 'Duplex'],
    N, p=[0.55, 0.20, 0.15, 0.10]
)

heating_type = np.random.choice(
    ['Gas', 'Electric', 'Oil', 'Solar'],
    N, p=[0.55, 0.30, 0.10, 0.05]
)

has_pool      = np.random.choice([0, 1], N, p=[0.85, 0.15])
has_basement  = np.random.choice([0, 1], N, p=[0.40, 0.60])
central_air   = np.random.choice([0, 1], N, p=[0.20, 0.80])

# ---------------------------------------------------------------
# 3. BUILD TARGET VARIABLE: sale_price
#    We create a price formula based on the features above, then
#    add random noise so the relationships aren't perfectly linear
#    (more realistic for ML).
# ---------------------------------------------------------------

# Base price driven by square footage and quality
base_price = (
    area_sqft * 85
    + overall_qual * 9000
    + bedrooms * 4000
    + bathrooms * 6000
    + garage_spaces * 5000
    + lot_size * 1.5
    + has_pool * 15000
    + has_basement * 8000
    + central_air * 5000
)

# Age penalty — older houses lose value (unless heavily renovated, which we ignore for simplicity)
age = 2024 - year_built
age_penalty = age * 350

# Distance penalty — farther from city centre = cheaper, except 'Lakeside'/'Hillside' bonus
distance_penalty = distance_to_city_km * 900

# Neighborhood multiplier (location, location, location!)
neighborhood_multiplier = pd.Series(neighborhoods).map({
    'Downtown':       1.35,
    'Lakeside':       1.25,
    'Hillside':       1.20,
    'Suburb_North':   1.05,
    'Suburb_West':    1.05,
    'Suburb_East':    0.95,
    'Suburb_South':   0.90,
    'Rural':          0.75,
}).values

# House style adjustment
style_adjustment = pd.Series(house_style).map({
    'Single_Family': 1.10,
    'Hillside':      1.05,
    'Townhouse':     0.95,
    'Condo':         0.85,
    'Duplex':        0.90,
}).fillna(1.0).values

sale_price = (
    (base_price - age_penalty - distance_penalty)
    * neighborhood_multiplier
    * style_adjustment
)

# Add random noise (±10%) to simulate real-world unpredictability
noise = np.random.normal(1, 0.10, N)
sale_price = (sale_price * noise).clip(40000, None)  # floor price at $40,000

# ---------------------------------------------------------------
# 4. ASSEMBLE DATAFRAME
# ---------------------------------------------------------------
df = pd.DataFrame({
    'area_sqft':            area_sqft.round(0).astype(int),
    'bedrooms':             bedrooms,
    'bathrooms':            bathrooms,
    'year_built':           year_built,
    'garage_spaces':        garage_spaces,
    'lot_size':             lot_size.round(0).astype(int),
    'overall_qual':         overall_qual,
    'distance_to_city_km':  distance_to_city_km.round(1),
    'neighborhood':         neighborhoods,
    'house_style':          house_style,
    'heating_type':         heating_type,
    'has_pool':             has_pool,
    'has_basement':         has_basement,
    'central_air':          central_air,
    'sale_price':           sale_price.round(0).astype(int),
})

# ---------------------------------------------------------------
# 5. INTRODUCE REALISTIC MESSINESS (missing values + outliers)
#    Real datasets are never clean — this gives students practice
#    handling missing data in the pipeline.
# ---------------------------------------------------------------

# Randomly set ~3% of lot_size values to NaN (missing data)
missing_idx = np.random.choice(df.index, size=int(0.03 * N), replace=False)
df.loc[missing_idx, 'lot_size'] = np.nan

# Randomly set ~2% of heating_type to NaN
missing_idx2 = np.random.choice(df.index, size=int(0.02 * N), replace=False)
df.loc[missing_idx2, 'heating_type'] = np.nan

# Add a handful of extreme outliers (luxury mansions) for robustness testing
outlier_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_idx, 'area_sqft'] = (df.loc[outlier_idx, 'area_sqft'] * 3).astype(int)
df.loc[outlier_idx, 'sale_price'] = (df.loc[outlier_idx, 'sale_price'] * 2.5).astype(int)

# Shuffle rows so it doesn't look generated in order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------
# 6. SAVE TO CSV
# ---------------------------------------------------------------
output_path = 'data/housing.csv'
df.to_csv(output_path, index=False)

print(f"✅ Generated {len(df)} rows -> {output_path}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nPrice range: ${df['sale_price'].min():,} - ${df['sale_price'].max():,}")
print(f"Mean price: ${df['sale_price'].mean():,.0f}")
