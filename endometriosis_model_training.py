import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# --- 1. Load Data ---
try:
    df = pd.read_csv('structured_endometriosis_data.csv')
    print("Endometriosis dataset loaded successfully.")
except FileNotFoundError:
    print("Error: structured_endometriosis_data.csv not found.")
    print("Please ensure the file is in the correct directory.")
    raise

print("\nOriginal Dataset Head:")
print(df.head())
print("\nOriginal Dataset Info:")
df.info()

# --- 2. Initial Cleaning/Renaming ---
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^0-9a-zA-Z_]', '', regex=True)
print("\nDataset Preview after column cleanup:")
print(df.head())

# --- 3. Identify and Prepare Target Variable ---
target_column_candidates = ['Endometriosis_Diagnosis', 'Diagnosis', 'Outcome', 'Target', 'Label']
target_column = None
for col_candidate in target_column_candidates:
    if col_candidate.lower() in df.columns.str.lower():
        target_column = df.columns[df.columns.str.lower() == col_candidate.lower()].tolist()[0]
        break

if target_column is None:
    print("\nWARNING: No common target column found.")
    target_column = df.columns[-1]  # Fallback to last column for demo if specific not found

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found after all checks. Please specify it correctly.")

print(f"\nIdentified Target Column: '{target_column}'")

# Convert target labels if needed
le_target = None
if df[target_column].dtype == 'object' or df[target_column].nunique() > 2:
    le_target = LabelEncoder()
    df[target_column] = le_target.fit_transform(df[target_column])
    print(f"Target encoding using LabelEncoder: {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")

y = df[target_column]
X = df.drop(columns=[target_column])

# --- 4. Drop ID or irrelevant columns if present ---
df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()], inplace=True, errors='ignore')
print("\nDropped ID or Unnamed columns.")

# --- 5. Handle Missing Values ---
print("\nHandling missing values...")
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)

for col in X.select_dtypes(include='object').columns:
    if X[col].isnull().any():
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)

print("\nMissing values after imputation:")
print(X.isnull().sum())
print("\nFeatures after imputation (head):")
print(X.head())

# --- 6. Encode Categorical Features ---
print("\nEncoding categorical features...")
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    if X[col].nunique() > 1:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

print("\nFeatures after categorical encoding (head):")
print(X.head())

# --- 7. Scale Numerical Features ---
print("\nScaling numerical features...")
numerical_features_to_scale = X.select_dtypes(include=np.number).columns.tolist()
if numerical_features_to_scale:
    scaler = StandardScaler()
    X[numerical_features_to_scale] = scaler.fit_transform(X[numerical_features_to_scale])
    print(f"  Scaled numerical features: {numerical_features_to_scale}")

print("Features after scaling (head):")
print(X.head())

# --- 8. Handle Class Imbalance ---
print("\nHandling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Original dataset shape: {X.shape}, Resampled dataset shape: {X_resampled.shape}")

# --- 9. Split Data into Training and Testing Sets ---
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 10. Train the ML Model with Hyperparameter Tuning ---
print("\nTraining the Random Forest model with hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# --- 11. Evaluate the Model ---
print("\nEvaluating the model...")
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_test_pred = best_model.predict(X_test)
validation_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

print("\nClassification Report (Validation Set):")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix (Validation Set):")
print(confusion_matrix(y_test, y_test_pred))

# --- 12. Save the Trained Model and Preprocessing Assets ---
os.makedirs('models', exist_ok=True)

model_filename = 'models/endometriosis_random_forest_model.pkl'
scaler_filename = 'models/endometriosis_scaler.pkl' if scaler else None
label_encoders_filename = 'models/endometriosis_label_encoders.pkl' if label_encoders else None
features_list_filename = 'models/endometriosis_features.pkl'
target_encoder_filename = 'models/endometriosis_target_encoder.pkl' if le_target else None

print(f"\nSaving model to {model_filename}")
joblib.dump(best_model, model_filename)

if scaler_filename:
    print(f"Saving scaler to {scaler_filename}")
    joblib.dump(scaler, scaler_filename)

if label_encoders_filename:
    print(f"Saving label encoders to {label_encoders_filename}")
    joblib.dump(label_encoders, label_encoders_filename)

print(f"Saving feature names to {features_list_filename}")
joblib.dump(X.columns.tolist(), features_list_filename)

if target_encoder_filename:
    print(f"Saving target encoder to {target_encoder_filename}")
    joblib.dump(le_target, target_encoder_filename)

print("\nModel training, evaluation, and saving complete!")
print("The trained model and relevant preprocessing assets are saved in the 'models' directory.")
