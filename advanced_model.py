# advanced_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    "depth": np.random.randint(800, 2000, n_samples),
    "gas_methane": np.random.randint(80, 300, n_samples),
    "gas_co2": np.random.randint(50, 200, n_samples),
    "temperature": np.random.uniform(30, 55, n_samples),
    "humidity": np.random.uniform(60, 95, n_samples),
    "pressure": np.random.uniform(95, 105, n_samples),
    "accidents": np.random.randint(0, 10, n_samples),
    "soil_type": np.random.choice(["rocky", "sandy", "clay"], n_samples),
    "water_nearby": np.random.choice(["yes", "no"], n_samples),
    "mining_intensity": np.random.choice(["low", "medium", "high"], n_samples),
})

# Simulate target variable (lifetime)
df["lifetime"] = (
    30
    - (df["depth"] / 2000 * 10)
    - (df["gas_methane"] / 300 * 5)
    - (df["gas_co2"] / 200 * 2)
    - (df["temperature"] / 55 * 3)
    + (df["humidity"] / 100 * 2)
    - (df["accidents"] * 0.5)
    + np.random.normal(0, 1, n_samples)
).round(2)

# Features and labels
features = [
    "depth", "gas_methane", "gas_co2", "temperature",
    "humidity", "pressure", "accidents", "soil_type",
    "water_nearby", "mining_intensity"
]
X = df[features]
y = df["lifetime"]

# Preprocessing pipeline
cat_features = ["soil_type", "water_nearby", "mining_intensity"]
num_features = list(set(features) - set(cat_features))

preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(), cat_features)
])

# Full model pipeline
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save to disk
df.to_csv("advanced_coal_data.csv", index=False)
joblib.dump(pipeline, "advanced_model.pkl")

print("✅ Model and dataset saved: advanced_model.pkl, advanced_coal_data.csv")
