🌾 AgriPredict India: Optimal Crop Prediction

Overview

AgriPredict India is a machine learning-powered web application designed to assist Indian farmers and agricultural analysts in determining the most suitable crop variety for specific environmental and soil conditions.

The application leverages a trained classification model to instantly predict the optimal crop out of 250 diverse Indian varieties based on 11 key input metrics.

✨ Key Features

11 Input Metrics: Accepts user input for Temperature, Humidity, Rainfall, Sunlight Hours, Soil Type, pH, Moisture, Elevation, and the NPK (Nitrogen, Phosphorus, Potassium) ratio.

Real-Time Prediction: Provides instant classification of the optimal crop variety upon submission.

User-Friendly Interface: Built using Streamlit with a clean, high-contrast Black & White theme for maximum clarity and accessibility.

Descriptive Soil Types: Uses real soil names (e.g., Alluvial, Loamy, Red) instead of generic indices for intuitive user input.

📁 Project Structure

The project requires the following files to run correctly:

.
├── app.py                      # Main Streamlit application and UI logic
├── requirements.txt            # Python dependencies
├── crop_prediction_model.pkl   # The trained Machine Learning model
├── preprocessor.pkl            # Pipeline/Scaler object for data transformation
└── label_encoder.pkl           # Encoder object for mapping prediction output back to crop names


Note: The .pkl model files must be present in the same directory as app.py.

⚙️ Installation and Setup

Prerequisites

You need Python 3.8+ installed on your system.

1. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.\.venv\Scripts\activate
# Activate the virtual environment (macOS/Linux)
source .venv/bin/activate


2. Install Dependencies

Use the provided requirements.txt file to install all necessary Python packages, including Streamlit and the core machine learning libraries.

pip install -r requirements.txt


▶️ How to Run the Application

Once the dependencies are installed and the virtual environment is active, run the Streamlit application from your terminal:

streamlit run app.py


Streamlit will automatically open the application in your default web browser (usually at http://localhost:8501).

💻 Technology Stack

Component

Technology

Role

Frontend/App

Streamlit

Web framework for rapid data app development

Backend/Model

Python, Scikit-learn, Joblib

Core logic, model loading, and prediction

Data Handling

Pandas, NumPy

Data input, transformation, and processing

Styling

Custom CSS

Black and White high-contrast UI theme

📊 Model & Data

Model Type: Classification Model (The specific algorithm, e.g., RandomForestClassifier or XGBoost, is contained within crop_prediction_model.pkl).

Target: Predicts one of 250 specific crop varieties.

Input Features: 11 features including environmental (T, H, R, S) and soil attributes (N, P, K, pH, Moisture, Elevation, Soil Type).

📄 License

This project is open-sourced under the MIT License. (Add your specific license details here).
