import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os


# Load data
df = pd.read_csv('crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)


# Save model and scaler
joblib.dump(model, 'crop_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")