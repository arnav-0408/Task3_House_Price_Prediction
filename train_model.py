"""
QSkill AI/ML Internship - Task 3: House Price Prediction
Training script — trains Linear Regression and Random Forest, picks the
better one by RMSE, and saves the full pipeline (preprocessing + model)
as pipeline.pkl so the Streamlit app can load it instantly.

Run this once before launching the app:
    python3 train_model.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]


def main():
    df = pd.read_csv("housing.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, max_depth=18, random_state=42, n_jobs=-1
        ),
    }

    best_name, best_pipeline, best_rmse = None, None, np.inf
    for name, regressor in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n=== {name} ===")
        print(f"RMSE: ${rmse:,.0f}  MAE: ${mae:,.0f}  R^2: {r2:.4f}")

        if rmse < best_rmse:
            best_name, best_pipeline, best_rmse = name, pipeline, rmse

    print(f"\nSelected model for the app: {best_name} (RMSE=${best_rmse:,.0f})")

    joblib.dump(best_pipeline, "pipeline.pkl")
    with open("model_name.txt", "w") as f:
        f.write(best_name)

    # Save median/typical values so the app can pre-fill sensible defaults
    defaults = df[NUMERIC_FEATURES].median().to_dict()
    defaults["ocean_proximity_options"] = sorted(df["ocean_proximity"].unique().tolist())
    joblib.dump(defaults, "defaults.pkl")

    print("Saved: pipeline.pkl, model_name.txt, defaults.pkl")


if __name__ == "__main__":
    main()
