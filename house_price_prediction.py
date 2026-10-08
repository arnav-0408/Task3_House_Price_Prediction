"""
QSkill AI/ML Internship - Task 3: House Price Prediction
Objective: Predict median house value from features like size, location,
and number of rooms/bedrooms.
Dataset: California Housing Prices (StatLib / Pace & Barry, 1997) - housing.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. Load the dataset and explore distributions
# ---------------------------------------------------------
df = pd.read_csv("housing.csv")

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

df.hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.savefig("housing_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["longitude"], df["latitude"], c=df["median_house_value"],
    cmap="viridis", s=df["population"] / 100, alpha=0.4
)
plt.colorbar(scatter, label="Median House Value ($)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("House Value by Location (size = population)")
plt.savefig("housing_geo_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved plots: housing_distributions.png, housing_geo_scatter.png")

# ---------------------------------------------------------
# 2. Handle missing data and preprocess (normalization)
# ---------------------------------------------------------
# total_bedrooms has some missing values -> impute with median.
# ocean_proximity is categorical -> one-hot encode.
# Numeric features are scaled (normalized) via StandardScaler.

numeric_features = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
categorical_features = ["ocean_proximity"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

X = df[numeric_features + categorical_features]
y = df["median_house_value"]

# ---------------------------------------------------------
# 3. Split into train/test sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. Train regression models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=18, random_state=42, n_jobs=-1),
}

results = {}

for name, regressor in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    print(f"\n=== {name} ===")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R^2:  {r2:.4f}")

# ---------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------
print("\n=== Model Comparison ===")
for name, res in results.items():
    print(f"{name}: RMSE=${res['rmse']:,.0f}, MAE=${res['mae']:,.0f}, R^2={res['r2']:.4f}")

best_model_name = min(results, key=lambda k: results[k]["rmse"])
print(f"\nBest performing model: {best_model_name}")
