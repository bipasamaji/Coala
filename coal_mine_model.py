# coal_mine_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load dataset
df = pd.read_csv("coal_mine_dataset.csv")  # Replace with your real CSV

# 2. Define features and label
X = df.drop("lifetime", axis=1)
y = df["lifetime"]

# 3. Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}, R2 Score: {r2:.2f}")

# 6. Export model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
