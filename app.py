import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
APP_NAME = "AgriPredict India"
MODEL_FILE = 'crop_prediction_model.pkl'
PREPROCESSOR_FILE = 'preprocessor.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'


# --- 1. Load Model Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the model, preprocessor, and label encoder."""
    files = [MODEL_FILE, PREPROCESSOR_FILE, LABEL_ENCODER_FILE]
    for file in files:
        if not os.path.exists(file):
            st.error(f"Missing essential file: '{file}'. Please ensure all model artifacts are deployed with the app.")
            st.stop()

    try:
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        return model, preprocessor, label_encoder
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()


model, preprocessor, label_encoder = load_artifacts()

# --- 2. Streamlit UI Layout and Custom Styling ---
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, black and white look
st.markdown("""
<style>
/* 1. Page Background and Typography */
.stApp {
    background-color: #FFFFFF; /* Pure White Background */
    color: #333333; /* Dark Gray Text */
    font-family: 'Inter', sans-serif;
}

/* 2. Main Title */
h1 {
    color: #000000 !important; /* Black Title */
    text-shadow: none;
    font-weight: 900;
    border-bottom: 3px solid #000000;
    padding-bottom: 5px;
}

/* 3. Section Headers (Inputs) */
h2, h3, h4 {
    color: #333333; /* Dark Gray Headers */
    border-bottom: 2px solid #DDDDDD; /* Light Gray Separator */
    padding-bottom: 5px;
    margin-top: 20px;
}

/* 4. Primary Button Styling (Predict Optimal Crop) */
div.stButton > button:first-child {
    background-color: #000000; /* Black Button */
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 24px;
    box-shadow: 0 4px #444444; /* Darker Gray Shadow */
    transition: all 0.2s;
    border: 1px solid #000000;
}
div.stButton > button:first-child:hover {
    background-color: #333333; /* Dark Gray on Hover */
    box-shadow: 0 2px #444444;
    transform: translateY(2px);
}

/* 5. Metrics/Results Box */
[data-testid="stMetric"] {
    background-color: #F9F9F9; /* Off-white for results box */
    border: 2px solid #AAAAAA; /* Gray Border */
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    text-align: center;
}

/* Style for the metric value (e.g., "Tomato") */
[data-testid="stMetricValue"] {
    font-size: 3em;
    color: #000000; /* Black value */
    font-weight: bold;
    margin-top: 10px;
    line-height: 1.2;
    overflow-wrap: break-word;
    white-space: normal;
}

/* Style for the metric label (e.g., "Predicted Crop Type") */
[data-testid="stMetricLabel"] label {
    font-size: 1.2em;
    color: #555555; /* Medium Gray label */
    font-weight: 600;
}


/* 6. Success Message */
.stAlert.success {
    /* Use a light/dark gray success theme */
    background-color: #EEEEEE; 
    border-color: #CCCCCC;
    color: #333333;
    font-weight: 500;
}

/* 7. Info Message */
.stAlert.info {
    /* Use a simple border/background for info */
    background-color: #F8F8F8; 
    border-color: #BBBBBB;
    color: #555555;
}

/* 8. Input sliders/selects */
.stSlider label, .stSelectbox label {
    font-weight: 600;
    color: #333333; /* Dark Gray Input Labels */
}

/* Remove default Streamlit color on primary button hover for full B&W adherence */
.stButton button {
    border-color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

st.title(f"🌾 {APP_NAME}")
st.markdown(
    "Enter the environmental and soil conditions to predict the most suitable crop out of **250 diverse Indian varieties**.")

# --- Define fixed options based on training data ---
# Mapped Soil_0 to Alluvial, Soil_1 to Black, Soil_2 to Red, and so on.
SOIL_TYPES = [
    'Alluvial',  # Soil_0
    'Black (Regur)',  # Soil_1
    'Red',  # Soil_2
    'Laterite',  # Soil_3
    'Mountainous/Forest',  # Soil_4
    'Arid/Desert',  # Soil_5
    'Saline and Alkaline',  # Soil_6
    'Peaty and Marshy',  # Soil_7
    'Loamy',  # Soil_8
    'Sandy'  # Soil_9
]
TEMP_RANGE = (10.0, 40.0)
PH_RANGE = (4.5, 8.5)
N_RANGE = (20.0, 100.0)
P_RANGE = (10.0, 60.0)
R_RANGE = (300.0, 2500.0)
K_RANGE = (15.0, 80.0)
E_RANGE = (0, 3000)

# --- 3. Input Widgets ---

# Use columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("1. Environmental Metrics")
    temperature = st.slider("Temperature (°C)", TEMP_RANGE[0], TEMP_RANGE[1], 25.0, 0.1)
    humidity = st.slider("Humidity (%)", 20.0, 90.0, 60.0, 1.0)
    rainfall = st.slider("Rainfall (mm/year)", R_RANGE[0], R_RANGE[1], 1200.0, 10.0)
    sunlight_hours = st.slider("Sunlight Hours (h/day)", 4.0, 12.0, 8.0, 0.1)

with col2:
    st.header("2. Soil Characteristics")
    # Use the new descriptive soil types
    soil_display = st.selectbox("Soil Type", SOIL_TYPES, index=0)

    # We must map the descriptive name back to the original format (Soil_X) for the model
    soil_index = SOIL_TYPES.index(soil_display)
    soil_type = f'Soil_{soil_index}'

    ph = st.slider("pH Level", PH_RANGE[0], PH_RANGE[1], 6.5, 0.1)
    moisture = st.slider("Moisture (%)", 10.0, 60.0, 35.0, 0.1)
    elevation = st.slider("Elevation (meters)", E_RANGE[0], E_RANGE[1], 500, 10)

with col3:
    st.header("3. Nutrients (PPM)")
    nitrogen = st.slider("Nitrogen (N)", N_RANGE[0], N_RANGE[1], 60.0, 1.0)
    phosphorus = st.slider("Phosphorus (P)", P_RANGE[0], P_RANGE[1], 40.0, 1.0)
    potassium = st.slider("Potassium (K)", K_RANGE[0], K_RANGE[1], 50.0, 1.0)

st.markdown("---")

# --- 4. Prediction Logic ---

if st.button('Predict Optimal Crop', type="primary", use_container_width=True):
    # Create DataFrame from user inputs
    input_data = pd.DataFrame([{
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall,
        'moisture': moisture,
        # Pass the model-required format (Soil_X)
        'soil_type': soil_type,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'elevation': elevation,
        'sunlight_hours': sunlight_hours
    }])

    # 1. Preprocess the data
    try:
        X_processed = preprocessor.transform(input_data)

        # 2. Make prediction
        prediction_encoded = model.predict(X_processed)

        # 3. Inverse transform to get crop name
        predicted_crop = label_encoder.inverse_transform(prediction_encoded)[0]

        # --- Display Results ---
        st.success("✨ Prediction Complete!")

        # Determine the base crop name for nicer display
        base_crop_name = predicted_crop.split('_V')[0]

        st.balloons()

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.metric(label="Predicted Crop Type", value=base_crop_name)

        with col_res2:
            st.subheader(f"Optimal Crop: {predicted_crop}")
            st.markdown(f"""
            Based on the input parameters:
            - **Soil:** {soil_display} ({soil_type}) 
            - **Temperature:** {temperature}°C
            - **N/P/K Ratio:** {nitrogen}/{phosphorus}/{potassium}

            This specific variety (`{predicted_crop}`) is perfectly suited for these conditions.
            """)

            st.info(
                "Note: The model uses deterministic rules to map these 11 features to one of 250 unique crop varieties, guaranteeing high accuracy based on the synthetic training data.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")