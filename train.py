import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---- Generate synthetic data ----
np.random.seed(42)
n = 300

study_hours   = np.random.uniform(0, 10, n)
attendance    = np.random.uniform(50, 100, n)
assignments   = np.random.uniform(0, 100, n)

score = (
    study_hours  * 4.0
    + attendance * 0.3
    + assignments * 0.2
    + np.random.normal(0, 5, n)
).clip(0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance":  attendance,
    "assignments": assignments,
    "score":       score,
})
df.to_csv("data/students.csv", index=False)
print(f"Dataset saved  →  data/students.csv  ({len(df)} rows)")

# ---- Train ----
X = df[["study_hours", "attendance", "assignments"]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse    = mean_squared_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"MSE : {mse:.2f}")
print(f"R²  : {r2:.4f}")

joblib.dump(model, "model/model.pkl")
print("Model saved  →  model/model.pkl")