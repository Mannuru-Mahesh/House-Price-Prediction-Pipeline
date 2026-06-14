"""
predict.py
-----------
Loads the trained pipeline (models/house_price_pipeline.pkl) and
predicts the sale price for one or more new houses.

Usage:
    python src/predict.py

Edit the `new_houses` list below to try your own inputs, or import
`predict_price()` into another script.
"""

import pandas as pd
import joblib

# Load the full pipeline (preprocessing + model) saved during training
MODEL_PATH = 'models/house_price_pipeline.pkl'
pipeline = joblib.load(MODEL_PATH)


def predict_price(house_dict):
    """
    Predict the sale price for a single house.

    Parameters
    ----------
    house_dict : dict
        Must contain all the feature columns the model was trained on:
        area_sqft, bedrooms, bathrooms, year_built, garage_spaces,
        lot_size, overall_qual, distance_to_city_km, neighborhood,
        house_style, heating_type, has_pool, has_basement, central_air

    Returns
    -------
    float : predicted sale price in dollars
    """
    df = pd.DataFrame([house_dict])
    prediction = pipeline.predict(df)[0]
    return prediction


if __name__ == '__main__':

    # ----- Example houses to predict -----
    new_houses = [
        {
            'area_sqft': 2200,
            'bedrooms': 4,
            'bathrooms': 2.5,
            'year_built': 2015,
            'garage_spaces': 2,
            'lot_size': 8000,
            'overall_qual': 8,
            'distance_to_city_km': 5.0,
            'neighborhood': 'Suburb_North',
            'house_style': 'Single_Family',
            'heating_type': 'Gas',
            'has_pool': 0,
            'has_basement': 1,
            'central_air': 1,
        },
        {
            'area_sqft': 950,
            'bedrooms': 2,
            'bathrooms': 1,
            'year_built': 1975,
            'garage_spaces': 1,
            'lot_size': 3000,
            'overall_qual': 4,
            'distance_to_city_km': 25.0,
            'neighborhood': 'Rural',
            'house_style': 'Duplex',
            'heating_type': 'Electric',
            'has_pool': 0,
            'has_basement': 0,
            'central_air': 0,
        },
        {
            'area_sqft': 4200,
            'bedrooms': 5,
            'bathrooms': 4,
            'year_built': 2020,
            'garage_spaces': 3,
            'lot_size': 15000,
            'overall_qual': 10,
            'distance_to_city_km': 1.5,
            'neighborhood': 'Downtown',
            'house_style': 'Single_Family',
            'heating_type': 'Solar',
            'has_pool': 1,
            'has_basement': 1,
            'central_air': 1,
        },
    ]

    print("=" * 50)
    print("HOUSE PRICE PREDICTIONS")
    print("=" * 50)

    for i, house in enumerate(new_houses, 1):
        price = predict_price(house)
        print(f"\nHouse {i}:")
        print(f"  {house['area_sqft']} sqft | {house['bedrooms']} bed | "
              f"{house['bathrooms']} bath | {house['neighborhood']} | "
              f"Quality {house['overall_qual']}/10")
        print(f"  --> Predicted Price: ${price:,.0f}")
