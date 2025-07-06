import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


# Load original dataset
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, '../Dataset/crop_recommendation.csv')
model_dir = os.path.join(base_dir, '../Models')
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(dataset_path)


# Simulate crop yield (in kg/ha) using a normal distribution
np.random.seed(42)
df['yield'] = np.random.normal(loc=2500, scale=400, size=len(df)).astype(int)


# Split data
X = df.drop(['label', 'yield'], axis=1)
y = df['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train regression model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)


# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# Save model and scaler
yield_path = os.path.join(model_dir, 'yield_model.pkl')
yield_scaler = os.path.join(model_dir, 'yield_scaler.pkl')

joblib.dump(model, yield_path)
joblib.dump(scaler, yield_scaler)