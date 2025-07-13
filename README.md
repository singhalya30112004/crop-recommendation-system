# Crop Recommendation System

An AI-powered app that recommends the best crop to grow based on soil and climate conditions — and predicts the expected yield (kg/ha). Built with machine learning and deployed using Streamlit Cloud, it supports real-time predictions with optional weather API integration.

**Live App:** [https://singhalya3011-crop-recommendation-system.streamlit.app](https://singhalya3011-crop-recommendation-system.streamlit.app)


## Dataset

Used the **Crop Recommendation Dataset** (publicly available) containing:
- Soil nutrients: `N`, `P`, `K`
- Climate conditions: `temperature`, `humidity`, `rainfall`
- Soil acidity: `pH`
- Target: `label` (crop name)

Simulated additional `yield (kg/ha)` column for regression modeling.


## Workflow

### 1. EDA & Preprocessing (`Code/eda.py`)
- Visualized feature distributions
- Checked for outliers, patterns, correlations

### 2. Model Training (`Code/model_training.py`)
- Scaled features using `StandardScaler`
- Trained models: `Random Forest`, `SVM`, `KNN`
- Best accuracy: **~99.32% (Random Forest Classifier)**

### 3. Yield Prediction (`Code/yield_model_training.py`)
- Simulated realistic crop yields
- Trained `Random Forest Regressor`
- RMSE ≈ **415.20 kg/ha**, R² ≈ **-0.11**

### 4. Weather API (`Utilities/weather_api.py`)
- Integrated OpenWeatherMap API to auto-fill real-time temperature, humidity, and rainfall

### 5. App Deployment (`Code/app.py`)
- Built with `Streamlit`
- Accepts user input or weather-based values
- Shows recommended crop and expected yield
- Deployed on Streamlit Cloud


## Sample Prediction

|       Input      | Example |
|------------------|---------|
| N                | 90      |
| P                | 40      |
| K                | 40      |
| Temperature (°C) | 25      |
| Humidity (%)     | 80      |
| pH               | 6.5     |
| Rainfall (mm)    | 200     |

Output:
- **Recommended Crop:** Rice  
- **Expected Yield:** ~2303 kg/ha


## Tech Stack

- `Python`, `Pandas`, `NumPy`
- `Scikit-learn`, `Joblib`
- `Matplotlib`, `Seaborn` (for EDA)
- `Streamlit` (for frontend)
- `OpenWeatherMap API` (weather data)


## How to Run

```bash
# Clone this repo
git clone https://github.com/singhalya30112004/crop-recommendation-system
cd crop-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
cd Code
streamlit run app.py
```

## Author

Alya Singh  
[LinkedIn](https://www.linkedin.com/in/alya-singh/)  
[GitHub: @singhalya30112004](https://github.com/singhalya30112004)


## License
MIT License – feel free to use, share, and modify!
