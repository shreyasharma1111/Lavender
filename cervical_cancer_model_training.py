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
    df = pd.read_csv('kag_risk_factors_cervical_cancer.csv')
    print("Cervical Cancer dataset loaded successfully.")
except FileNotFoundError:
    print("Error: kag_risk_factors_cervical_cancer.csv not found.")
    print("Please ensure the file is in the correct directory.")
    raise

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
# IMPORTANT: 'Biopsy' is typically used as the main diagnosis target.
# If you want to predict other columns (Hinselmann, Schiller, Citology), adjust here.
target_column = 'Biopsy'

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found. Please verify the column name.")

# Convert target to integer (0 or 1) - it should already be 0 or 1 in this dataset
y = df[target_column].astype(int)
X = df.drop(columns=[target_column])

print(f"\nIdentified Target Column: '{target_column}'")
print(f"Target variable unique values: {y.unique()}")

# --- 4. Drop ID or irrelevant columns if present ---
# This dataset doesn't usually have explicit 'id' columns but 'Unnamed' can appear.
df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()], inplace=True, errors='ignore')
print("\nDropped ID or Unnamed columns.")

# --- 5. Handle Missing Values (specifically '?' and then NaNs) ---
print("\nHandling missing values (replacing '?' with NaN and then imputing)...")

# Replace '?' with NaN across the entire DataFrame for proper numerical conversion/imputation
X = X.replace('?', np.nan)

# Convert all columns that should be numerical to numeric, coercing errors
for col in X.columns:
    if X[col].dtype == 'object':
        # Try to convert to numeric, if fails, it's truly categorical
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Impute numerical columns with median after conversion to numeric
for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"  Imputed missing in numerical '{col}' with median: {median_val}")

# Impute remaining categorical (object/string) columns with mode
for col in X.select_dtypes(include='object').columns:
    if X[col].isnull().any():
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        print(f"  Imputed missing in categorical '{col}' with mode: {mode_val}")

print("\nMissing values after imputation:")
print(X.isnull().sum())
print("\nFeatures after imputation (head):")
print(X.head())

# --- 6. Encode Categorical Features ---
print("\nEncoding categorical features...")
label_encoders = {} # Store encoders for later use in prediction

for col in X.select_dtypes(include='object').columns:
    if X[col].nunique() > 1: # Only encode if more than one unique value
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  Encoded categorical column '{col}'. Mapping: {list(le.classes_)}")
    else:
        print(f"  Skipping encoding for '{col}' as it has only one unique value.")
        # If it has only one unique value, it's a constant. It will be kept as its numerical value.

print("\nFeatures after categorical encoding (head):")
print(X.head())

# --- 7. Scale Numerical Features ---
print("\nScaling numerical features...")
numerical_features_to_scale = X.select_dtypes(include=np.number).columns.tolist()

if numerical_features_to_scale:
    scaler = StandardScaler()
    X_scaled_part = scaler.fit_transform(X[numerical_features_to_scale])
    X[numerical_features_to_scale] = X_scaled_part
    print(f"  Scaled numerical features: {numerical_features_to_scale}")
else:
    scaler = None # No scaler needed if no numerical features to scale
    print("  No continuous numerical features found to scale.")

print("Features after scaling (head):")
print(X.head())

# --- 8. Split Data into Training and Testing Sets ---
print("\nSplitting data into training and testing sets...")
# Ensure target variable 'y' is binary (0 or 1) before stratifying
if y.nunique() > 2:
    print("Warning: Target variable has more than 2 unique values, cannot stratify. Proceeding without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
if y.nunique() <= 2:
    print(f"Cervical Cancer prevalence in training set: {y_train.sum()/len(y_train):.2f}")
    print(f"Cervical Cancer prevalence in testing set: {y_test.sum()/len(y_test):.2f}")


# --- 9. Train the ML Model ---
print("\nTraining the Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=500) # Increased max_iter for this dataset
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
print(classification_report(y_test, model.predict(X_test)))

print("\nConfusion Matrix (Validation Set):")
print(confusion_matrix(y_test, model.predict(X_test)))

# --- 11. Save the Trained Model and Preprocessing Assets ---
os.makedirs('models', exist_ok=True) # Create models directory if it doesn't exist

model_filename = 'models/cervical_cancer_logistic_regression_model.pkl'
scaler_filename = 'models/cervical_cancer_scaler.pkl' if scaler else None # Only save scaler if it was used
label_encoders_filename = 'models/cervical_cancer_label_encoders.pkl' if label_encoders else None
features_list_filename = 'models/cervical_cancer_features.pkl'
# No target encoder saved as Biopsy is already 0/1 in this specific dataset

print(f"\nSaving model to {model_filename}")
joblib.dump(model, model_filename)

if scaler_filename:
    print(f"Saving scaler to {scaler_filename}")
    joblib.dump(scaler, scaler_filename)
else:
    print("No scaler saved as no numerical features were scaled.")

if label_encoders_filename:
    print(f"Saving label encoders to {label_encoders_filename}")
    joblib.dump(label_encoders, label_encoders_filename)
else:
    print("No categorical features were encoded, so no label encoders saved.")

print(f"Saving feature names to {features_list_filename}")
joblib.dump(X.columns.tolist(), features_list_filename) # Save feature names in correct order

print("\nModel training, evaluation, and saving complete!")
print("The trained model and relevant preprocessing assets are saved in the 'models' directory.")
print("You can now load these .pkl files in your Flask backend for predictions.")
