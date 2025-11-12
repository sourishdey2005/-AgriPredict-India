import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')

# --- 1. DATA GENERATION (FOR DEMONSTRATION) ---
np.random.seed(42)
N = 100000

# Feature Ranges (Used for Rule-based Assignment)
MIN_TEMP, MAX_TEMP = 10, 40
MIN_PH, MAX_PH = 4.5, 8.5
MIN_N, MAX_N = 20, 100
MIN_P, MAX_P = 10, 60

# Simulate realistic agricultural parameters
data = {
    'temperature': np.random.uniform(MIN_TEMP, MAX_TEMP, N),
    'humidity': np.random.uniform(20, 90, N),
    'ph': np.random.uniform(MIN_PH, MAX_PH, N),
    'rainfall': np.random.uniform(300, 2500, N),
    'moisture': np.random.uniform(10, 60, N),
    'soil_type': np.random.choice([f'Soil_{i}' for i in range(10)], N),  # Increased soil types
    'nitrogen': np.random.uniform(MIN_N, MAX_N, N),
    'phosphorus': np.random.uniform(MIN_P, MAX_P, N),
    'potassium': np.random.uniform(15, 80, N),
    'elevation': np.random.uniform(0, 3000, N),
    'sunlight_hours': np.random.uniform(4, 12, N),
}

df = pd.DataFrame(data)

# --- UPDATED: GENERATE 250 DETERMINISTIC CROP NAMES WITH INDIAN DIVERSITY ---
NUM_CROPS = 250

# Expanded list of Indian crops (Cereals, Pulses, Oilseeds, Spices, Fruits, Veggies)
base_crops = [
    'Rice', 'Wheat', 'Maize', 'Barley', 'Millets', 'Sorghum', 'Ragi', 'Bajra',
    'Soybeans', 'Groundnut', 'Mustard', 'Sunflower', 'Sesame', 'Castor', 'Coconut',
    'Cotton', 'Jute', 'Hemp', 'Sugarcane', 'Potato', 'Onion', 'Tomato', 'Cabbage',
    'Cauliflower', 'Peas', 'Lentil', 'Chickpea', 'Black_Gram', 'Green_Gram', 'Pigeon_Pea',
    'Apple', 'Banana', 'Mango', 'Orange', 'Grapes', 'Pomegranate', 'Guava', 'Papaya',
    'Coffee', 'Tea', 'Tobacco', 'Rubber', 'Cardamom', 'Clove', 'Cinnamon', 'Turmeric',
    'Ginger', 'Chilli', 'Black_Pepper', 'Arecanut'  # 50 unique names
]

# Generate 250 unique crop names based on the expanded base list
crop_names = []
for i in range(NUM_CROPS):
    base = base_crops[i % len(base_crops)]
    # Create unique name by combining index and base name
    crop_names.append(f"{base}_V{i + 1}")

# Determine the bin size for key features to create non-overlapping decision boundaries
# 5 bins for Temperature, 5 for pH, 5 for N, 2 for P = 5*5*5*2 = 250 unique crops
TEMP_BINS = np.linspace(MIN_TEMP, MAX_TEMP, 6)  # 5 bins
PH_BINS = np.linspace(MIN_PH, MAX_PH, 6)  # 5 bins
N_BINS = np.linspace(MIN_N, MAX_N, 6)  # 5 bins
P_BINS = np.linspace(MIN_P, MAX_P, 3)  # 2 bins


# --- ASSIGN CROP BASED ON STRICT FEATURE BINS ---
def assign_deterministic_crop(row):
    # Determine the bin index for each key feature
    temp_idx = np.digitize(row['temperature'], TEMP_BINS) - 1
    ph_idx = np.digitize(row['ph'], PH_BINS) - 1
    n_idx = np.digitize(row['nitrogen'], N_BINS) - 1
    p_idx = np.digitize(row['phosphorus'], P_BINS) - 1

    # Clamp indices to valid range [0, 4] or [0, 1]
    temp_idx = np.clip(temp_idx, 0, 4)
    ph_idx = np.clip(ph_idx, 0, 4)
    n_idx = np.clip(n_idx, 0, 4)
    p_idx = np.clip(p_idx, 0, 1)

    # Combine indices into a unique number from 0 to 249
    # Index = (T_idx * 100) + (pH_idx * 20) + (N_idx * 4) + P_idx * 2 + (K_idx * 1)
    # Index = (T_idx * 5 * 5 * 2) + (pH_idx * 5 * 2) + (N_idx * 2) + P_idx
    index = (temp_idx * 100) + (ph_idx * 20) + (n_idx * 4) + (p_idx * 2) + (row['rainfall'] > 1500)

    # Simple modulo operation to ensure index is within the 0-249 range
    index = index % NUM_CROPS

    # Use the combined index to select the unique crop name
    return crop_names[index]


df['crop'] = df.apply(assign_deterministic_crop, axis=1)

# Introduce missing values to simulate messy data
missing_idx = np.random.choice(df.index, size=int(0.05 * N), replace=False)
df.loc[missing_idx, 'ph'] = np.nan
df.loc[np.random.choice(df.index, size=int(0.03 * N)), 'moisture'] = np.nan

# --- 2. PREPROCESSING ---
X = df.drop('crop', axis=1)
y = df['crop']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify numeric and categorical columns
numeric_features = [
    'temperature', 'humidity', 'ph', 'rainfall', 'moisture',
    'nitrogen', 'phosphorus', 'potassium', 'elevation', 'sunlight_hours'
]
categorical_features = ['soil_type']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# --- 3. TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 4. MODEL TRAINING ---
# Use highly complex parameters for the Random Forest to capture 250 classes
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print(f"Starting model training on {X_train.shape[0]} samples with {len(label_encoder.classes_)} classes...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. EVALUATION ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 50)
print(f"✅ Accuracy: {accuracy:.4f}")
print("=" * 50)

# FIX: Convert all class names to strings for classification_report
clean_target_names = [str(c) for c in label_encoder.classes_]

print("\nClassification Report (Abbreviated):")
# Use output_dict=True to suppress the massive 250-line report
report_dict = classification_report(y_test, y_pred, target_names=clean_target_names, output_dict=True)

# Print a custom, abbreviated report for the first 10 classes
print("Showing metrics for a sample of 10 crops:")
print("{:<15} {:>8} {:>8} {:>8} {:>8}".format('Crop Name', 'Precision', 'Recall', 'F1-Score', 'Support'))
print("-" * 50)

for i, name in enumerate(clean_target_names[:10]):
    metrics = report_dict[name]
    print("{:<15} {:>8.4f} {:>8.4f} {:>8.4f} {:>8}".format(
        name, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']
    ))

print("-" * 50)
print("{:<15} {:>8} {:>8} {:>8.4f} {:>8}".format(
    'Overall Accuracy', '', '', report_dict['accuracy'], report_dict['macro avg']['support']
))
print("{:<15} {:>8.4f} {:>8.4f} {:>8.4f} {:>8}".format(
    'Macro Avg', report_dict['macro avg']['precision'], report_dict['macro avg']['recall'],
    report_dict['macro avg']['f1-score'], report_dict['macro avg']['support']
))

# --- 6. SAVE ARTIFACTS ---
joblib.dump(model, 'crop_prediction_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nModel, preprocessor, and label encoder saved successfully.")