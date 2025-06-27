import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- Configuration ---
DATASET_PATH = 'Gestational Diabetes.csv'
TARGET_COLUMN = 'Prediction' # Identified target column from the provided dataset
# Features where 0 might represent a missing value, based on common medical datasets
# For Gestational Diabetes, Weight, Height, BMI are unlikely to be 0.
# Age and Pregnancy No are also unlikely to be 0 in a meaningful context for a pregnant individual.
FEATURES_WITH_ZERO_AS_MISSING = ['Weight', 'Height', 'BMI']
MODEL_DIR = 'models'

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATASET_PATH)
    print("Gestational Diabetes dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found. Please ensure the file is in the correct directory.")
    raise

print("\nOriginal Dataset Head:")
print(df.head())
print("\nOriginal Dataset Info:")
df.info()

# --- 2. Initial Cleaning/Renaming ---
# Standardize column names by stripping whitespace and replacing spaces with underscores
df.columns = df.columns.str.strip().str.replace(' ', '_')

# --- 3. Identify and Prepare Target Variable ---
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
    raise ValueError(f"Target column '{TARGET_COLUMN}' missing.")

y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN])

print(f"\nTarget variable: '{TARGET_COLUMN}'")

# --- 4. Handle Missing Values (including 0s for specific features) ---
print("\nHandling missing values (including 0s for specific features)...")

for col in FEATURES_WITH_ZERO_AS_MISSING:
    if col in X.columns:
        # Replace 0s with NaN for proper imputation
        X[col] = X[col].replace(0, np.nan)
        # Impute NaNs with the median of the column
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"  Replaced 0s/NaNs in '{col}' with median: {median_val:.2f}")

# Handle any other general NaNs that might exist in the dataset
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype in ['int64', 'float64']:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  Imputed general missing in numerical '{col}' with median: {median_val:.2f}")
        else: # For any unexpected categorical NaNs, impute with mode
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
            print(f"  Imputed general missing in categorical '{col}' with mode: {mode_val}")

print("\nMissing values after imputation:")
print(X.isnull().sum())
print("\nFeatures after imputation (head):")
print(X.head())

# --- 5. Encode Categorical Features (if any) ---
# The provided dataset 'Gestational Diabetes.csv' appears to have only numerical features
# and a binary 'Heredity' column which is already 0/1.
# This section is kept general in case other categorical features are introduced or misidentified.
encoded_features = []
label_encoders = {} # Store encoders for later use in prediction

for col in X.select_dtypes(include='object').columns:
    if X[col].nunique() > 1: # Only encode if there's more than one unique category
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        encoded_features.append(col)
        print(f"  Encoded categorical column '{col}'. Mapping: {list(le.classes_)}")
    else:
        print(f"  Skipping encoding for '{col}' as it has only one unique value.")

# --- 6. Scale Numerical Features ---
print("\nScaling numerical features...")
# Select numerical columns AFTER potential categorical encoding
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_features])
X[numerical_features] = X_scaled # Update the DataFrame with scaled values

print("Features after scaling (head):")
print(X.head())

# --- 7. Split Data into Training and Testing Sets ---
print("\nSplitting data into training and testing sets...")
# Stratify by 'Prediction' to ensure balanced representation of target classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Target prevalence in training set (1s): {y_train.sum()}/{len(y_train)} ({y_train.mean():.2f})")
print(f"Target prevalence in testing set (1s): {y_test.sum()}/{len(y_test)} ({y_test.mean():.2f})")

# --- 8. Train the ML Model ---
print("\nTraining the Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=300)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 9. Evaluate the Model ---
print("\nEvaluating the model...")
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_test_pred = model.predict(X_test)
validation_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

print("\nClassification Report (Validation Set):")
print(classification_report(y_test, model.predict(X_test)))

print("\nConfusion Matrix (Validation Set):")
print(confusion_matrix(y_test, model.predict(X_test)))

# --- 10. Save the Trained Model and Preprocessing Assets ---
os.makedirs(MODEL_DIR, exist_ok=True)

model_filename = os.path.join(MODEL_DIR, 'gestational_diabetes_logistic_regression_model.pkl')
scaler_filename = os.path.join(MODEL_DIR, 'gestational_diabetes_scaler.pkl')
label_encoders_filename = os.path.join(MODEL_DIR, 'gestational_diabetes_label_encoders.pkl') if label_encoders else None
features_list_filename = os.path.join(MODEL_DIR, 'gestational_diabetes_features.pkl')

print(f"\nSaving model to {model_filename}")
joblib.dump(model, model_filename)

print(f"Saving scaler to {scaler_filename}")
joblib.dump(scaler, scaler_filename)

if label_encoders_filename:
    print(f"Saving label encoders to {label_encoders_filename}")
    joblib.dump(label_encoders, label_encoders_filename)
else:
    print("No categorical features were encoded, so no label encoders saved.")

print(f"Saving feature names to {features_list_filename}")
joblib.dump(X.columns.tolist(), features_list_filename)

print("\nModel training, evaluation, and saving complete!")
print(f"The trained model, scaler, and features list are saved in the '{MODEL_DIR}' directory.")
if label_encoders_filename:
    print("Label encoders are also saved if categorical features were present.")
print("You can now load these .pkl files in your Flask backend for predictions.")
