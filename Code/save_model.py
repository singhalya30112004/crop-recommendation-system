import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, '../Dataset/crop_recommendation.csv')
model_dir = os.path.join(base_dir, '../Models')
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(dataset_path)
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)


# Save model and scaler
model_path = os.path.join(model_dir, 'crop_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("Model and scaler saved successfully.")