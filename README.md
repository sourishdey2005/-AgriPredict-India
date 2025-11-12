# 🌾 AgriPredict India: Optimal Crop Prediction

**AgriPredict India** is a machine learning-powered web application designed to assist Indian farmers and agricultural analysts in determining the most suitable crop variety for specific environmental and soil conditions.

The application leverages a trained classification model to instantly predict the **optimal crop** out of **250 diverse Indian varieties** based on **11 key input metrics**.

---

## ✨ Key Features

- **11 Input Metrics:** Accepts user input for Temperature, Humidity, Rainfall, Sunlight Hours, Soil Type, pH, Moisture, Elevation, and the NPK (Nitrogen, Phosphorus, Potassium) ratio.  
- **Real-Time Prediction:** Provides instant classification of the optimal crop variety upon submission.  
- **User-Friendly Interface:** Built using Streamlit with a clean, high-contrast **Black & White** theme for maximum clarity and accessibility.  
- **Descriptive Soil Types:** Uses real soil names (e.g., *Alluvial*, *Loamy*, *Red*) instead of numeric indices for intuitive user input.  

---




> ⚠️ **Note:** The `.pkl` model files must be present in the same directory as `app.py`.

---

## ⚙️ Installation and Setup

### Prerequisites

You need **Python 3.8+** installed on your system.

### 1️⃣ Create and Activate Virtual Environment

It is **highly recommended** to use a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.\.venv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source .venv/bin/activate
```



## 📁 Project Structure


├── app.py # Main Streamlit application and UI logic
├── requirements.txt # Python dependencies
├── crop_prediction_model.pkl # The trained Machine Learning model
├── preprocessor.pkl # Pipeline/Scaler object for data transformation
└── label_encoder.pkl # Encoder object for mapping prediction output back to crop names


