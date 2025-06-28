# ovarian_cancer_ml_training.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- 1. Load Data ---
try:
    df = pd.read_csv('Ovarian_patient_data.csv')
    print("Ovarian Cancer dataset loaded successfully.")
except FileNotFoundError as e:  # Catch the exception as 'e'
    print("Error: Ovarian_patient_data.csv not found.")
    print("Please ensure the file is in the correct directory.")
    raise e  # Re-raise the caught exception 'e' to provide full traceback

print("\nOriginal Dataset Head:")
print(df.head())
print("\nOriginal Dataset Info:")
df.info()

# --- 2. Initial Cleaning/Renaming ---
# Standardize column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^0-9a-zA-Z_]', '', regex=True)
print("\nDataset Preview after column cleanup:")
print(df.head())

# --- 3. Identify and Prepare Target Variable ---
target_column = 'RiskLabel'

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found after all checks. Please specify it correctly.")

print(f"\nIdentified Target Column: '{target_column}'")

y = df[target_column]
X = df.drop(columns=[target_column])

# --- 4. Drop ID or irrelevant columns if present ---
columns_to_drop = ['Timestamp']
X = X.drop(columns=[col for col in columns_to_drop if col in X.columns], errors='ignore')
print("\nDropped Timestamp column from features if present.")

# --- 5. Handle Missing Values ---
print("\nHandling missing values...")

# Step 5.1: Convert all columns to numeric where possible, coercing errors
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Step 5.2: Check for any remaining NaN values before imputation
if X.isnull().sum().sum() > 0:
    print("Warning: There are NaN values in the dataset before imputation.")
    print(X.isnull().sum())

# Step 5.3: Impute numerical columns with median
for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"  Imputed missing in numerical '{col}' with median: {median_val}")

# Step 5.4: Check for any remaining NaN values after imputation
if X.isnull().sum().sum() > 0:
    print("Warning: There are still NaN values in the dataset after imputation.")
    print(X.isnull().sum())
else:
    print("No NaN values remain in the dataset after imputation.")

# --- 6. Encode Categorical Features ---
# --- 6. Encode Categorical Features ---
print("\nEncoding categorical features...")
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    if X[col].nunique() > 1:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  Encoded categorical column '{col}'. Mapping: {list(le.classes_)}")

# --- 💡 Final NaN check + fill ---
X.fillna(0, inplace=True)

# --- 7. Scale Numerical Features ---
print("\nScaling numerical features...")
numerical_features_to_scale = X.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X[numerical_features_to_scale] = scaler.fit_transform(X[numerical_features_to_scale])

print("Features after scaling (head):")
print(X.head())

# --- 8. Split Data into Training and Testing Sets ---
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"RiskLabel distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"RiskLabel distribution in testing set:\n{y_test.value_counts(normalize=True)}")

# --- 9. Train the ML Model ---
from sklearn.ensemble import RandomForestClassifier
print("\nTraining the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")

# --- 10. Evaluate the Model ---
print("\nEvaluating the model...")
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_test_pred = model.predict(X_test)
validation_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

print("\nClassification Report (Validation Set):")
print(classification_report(y_test, y_test_pred, zero_division=0))

print("\nConfusion Matrix (Validation Set):")
print(confusion_matrix(y_test, y_test_pred))

# --- 11. Save the Trained Model and Preprocessing Assets ---
os.makedirs('models', exist_ok=True)  # Create models directory if it doesn't exist

model_filename = 'models/ovarian_cancer_logistic_regression_model.pkl'
scaler_filename = 'models/ovarian_cancer_scaler.pkl' if scaler else None  # Only save scaler if it was used
label_encoders_filename = 'models/ovarian_cancer_label_encoders.pkl' if label_encoders else None
features_list_filename = 'models/ovarian_cancer_features.pkl'

print(f"\nSaving model to {model_filename}")
joblib.dump(model, model_filename)

if scaler_filename:
    print(f"Saving scaler to {scaler_filename}")
    joblib.dump(scaler, scaler_filename)

if label_encoders_filename:
    print(f"Saving label encoders to {label_encoders_filename}")
    joblib.dump(label_encoders, label_encoders_filename)

print(f"Saving feature names to {features_list_filename}")
joblib.dump(X.columns.tolist(), features_list_filename)  # Save feature names in correct order

print("\nModel training, evaluation, and saving complete!")
print("The trained model and relevant preprocessing assets are saved in the 'models' directory.")
