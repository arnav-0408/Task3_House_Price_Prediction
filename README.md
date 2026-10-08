# Task 3: House Price Prediction — QSkill AI/ML Internship

Predicts median house value from location, house age, room/bedroom counts,
population, households, income, and ocean proximity, using the California
Housing Prices dataset (1990 census block groups). Includes a Streamlit
web UI for interactive predictions.

## Files
- `housing.csv` — California Housing Prices dataset (20,640 rows)
- `house_price_prediction.py` — original standalone script (EDA, preprocessing, training, evaluation printed to console)
- `train_model.py` — trains Linear Regression + Random Forest, picks the better one by RMSE, saves `pipeline.pkl` for the app
- `app.py` — Streamlit frontend: enter house/location details, get an instant predicted price
- `pipeline.pkl`, `model_name.txt`, `defaults.pkl` — pre-trained artifacts (already included, ready to use)
- `housing_distributions.png`, `housing_geo_scatter.png` — EDA plots

## How to run

1. Install dependencies:
   ```
   pip install pandas scikit-learn matplotlib joblib streamlit
   ```

2. (Optional — artifacts are already included) Retrain the model:
   ```
   python3 train_model.py
   ```

3. Launch the app:
   ```
   streamlit run app.py
   ```
   Set the location, house age, rooms/bedrooms, population, households,
   income, and ocean proximity, then click "Predict price."

## Results
- Linear Regression: RMSE ≈ $70,059, R² ≈ 0.63
- Random Forest (selected): RMSE ≈ $49,026, R² ≈ 0.82

Random Forest is used in the app since it captures the non-linear
relationships between location/income and price much better than a
straight linear model.

## Note on the dataset
The classic "Boston Housing" dataset was removed from scikit-learn due
to ethical concerns about one of its features, and is no longer
distributed with the library. The California Housing Prices dataset
(also commonly used for this exact task) is used here instead — same
regression objective, similar size/location/price structure.
