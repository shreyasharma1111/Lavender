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
    df = pd.read_csv('lung_cancer_mortality_data_large_v2.csv')
    print("Lung Cancer dataset loaded successfully.")
except FileNotFoundError:
    print("Error: lung_cancer_mortality_data_large_v2.csv not found.")
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
target_column_candidates = ['Lung_Cancer_Mortality', 'Diagnosis', 'Outcome', 'Label', 'Target']
target_column = None
for col_candidate in target_column_candidates:
    if col_candidate.lower() in df.columns.str.lower():
        target_column = df.columns[df.columns.str.lower() == col_candidate.lower()].tolist()[0]
        break

if target_column is None:
    print("\nWARNING: No common target column found.")
    print("Assuming the last column as target for demonstration. Adjust 'target_column' variable if incorrect.")
    target_column = df.columns[-1]  # Fallback to last column for demo if specific not found

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found after all checks. Please specify it correctly.")

print(f"\nIdentified Target Column: '{target_column}'")

# Convert target labels if needed (e.g., string 'Yes'/'No' or 'Positive'/'Negative' to 1/0)
le_target = None
if df[target_column].dtype == 'object' or df[target_column].nunique() > 2:
    unique_target_values = df[target_column].unique()
    if 'positive' in [str(x).lower() for x in unique_target_values] and 'negative' in [str(x).lower() for x in unique_target_values]:
        df[target_column] = df[target_column].astype(str).str.lower().map({'positive': 1, 'negative': 0})
        print("Mapped 'positive' to 1 and 'negative' to 0 for target.")
    elif 'yes' in [str(x).lower() for x in unique_target_values] and 'no' in [str(x).lower() for x in unique_target_values]:
        df[target_column] = df[target_column].astype(str).str.lower().map({'yes': 1, 'no': 0})
        print("Mapped 'yes' to 1 and 'no' to 0 for target.")
    elif df[target_column].nunique() == 2:  # If 2 unique non-numeric values, use LabelEncoder
        le_target = LabelEncoder()
        df[target_column] = le_target.fit_transform(df[target_column])
        print(f"Target encoding using LabelEncoder: {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")
    else:
        print(f"Warning: Target column '{target_column}' has non-binary object type or multiple unique values not handled. Attempting numeric conversion.")
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(0).astype(int)
        print("Attempted to convert target to numeric, filling NaNs with 0.")

y = df[target_column]
X = df.drop(columns=[target_column])

# --- 4. Drop ID or irrelevant columns if present ---
df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()], inplace=True, errors='ignore')
print("\nDropped ID or Unnamed columns.")

# --- 5. Handle Missing Values ---
print("\nHandling missing values...")

# Convert all columns that should be numerical to numeric, coercing errors
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Check for columns that are entirely NaN
for col in X.columns:
    if X[col].isnull().all():
        print(f"Warning: Column '{col}' contains all NaN values and will be dropped.")
        X.drop(columns=[col], inplace=True)

# Impute numerical columns with median
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

# Check for any remaining NaNs
if X.isnull().sum().sum() > 0:
    raise ValueError("There are still NaN values in the feature set after imputation.")

print("\nMissing values after imputation:")
print(X.isnull().sum())
print("\nFeatures after imputation (head):")
print(X.head())

# --- 6. Encode Categorical Features ---
print("\nEncoding categorical features...")
label_encoders = {}  # Store encoders for later use in prediction

for col in X.select_dtypes(include='object').columns:
    if X[col].nunique() > 1:  # Only encode if more than one unique value
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  Encoded categorical column '{col}'. Mapping: {list(le.classes_)}")
    else:
        print(f"  Skipping encoding for '{col}' as it has only one unique value.")

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
    scaler = None  # No scaler needed if no numerical features to scale
    print("  No continuous numerical features found to scale.")

print("Features after scaling (head):")
print(X.head())

# --- 8. Split Data into Training and Testing Sets ---
print("\nSplitting data into training and testing sets...")
if y.nunique() > 2:
    print("Warning: Target variable has more than 2 unique values, cannot stratify. Proceeding without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
if y.nunique() <= 2:
    print(f"Lung Cancer prevalence in training set: {y_train.sum()/len(y_train):.2f}")
    print(f"Lung Cancer prevalence in testing set: {y_test.sum()/len(y_test):.2f}")

# --- 9. Train the ML Model ---
print("\nTraining the Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=500)  # Increased max_iter for this dataset
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
os.makedirs('models', exist_ok=True)  # Create models directory if it doesn't exist

model_filename = 'models/lung_cancer_logistic_regression_model.pkl'
scaler_filename = 'models/lung_cancer_scaler.pkl' if scaler else None  # Only save scaler if it was used
label_encoders_filename = 'models/lung_cancer_label_encoders.pkl' if label_encoders else None
features_list_filename = 'models/lung_cancer_features.pkl'
target_encoder_filename = 'models/lung_cancer_target_encoder.pkl' if le_target else None

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
joblib.dump(X.columns.tolist(), features_list_filename)  # Save feature names in correct order

if target_encoder_filename:
    print(f"Saving target encoder to {target_encoder_filename}")
    joblib.dump(le_target, target_encoder_filename)
else:
    print("No target encoder saved as target was already numerical or not multi-class categorical.")

print("\nModel training, evaluation, and saving complete!")
print("The trained model and relevant preprocessing assets are saved in the 'models' directory.")
