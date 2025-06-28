import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- Configuration ---
DATASET_PATH = 'breast_cancer_dataset.csv'
TARGET_COLUMN = 'Survival_Status' # Updated target column based on the provided dataset
ID_COLUMN = 'Patient_ID' # Updated ID column based on the provided dataset
MODEL_DIR = 'models'

# --- Load Dataset ---
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found.")
    raise

# --- Preprocessing ---

# Standardize column names
df.columns = df.columns.str.strip().str.replace(' ', '_')
print("\nDataset Preview:")
print(df.head())

# Check if target column exists
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")

# Separate target variable
y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors='ignore') # Drop ID column and target column

# Handle missing values (if any)
print("\nHandling missing values...")
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype == 'object':
            # For categorical, fill with mode
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
            print(f"  Imputed missing in categorical column '{col}' with mode: {mode_val}")
        else:
            # For numerical, fill with median
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  Imputed missing in numerical column '{col}' with median: {median_val}")

# Encode categorical features
encoded_features = []
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    encoded_features.append(col)
    print(f"Encoded '{col}' → {list(le.classes_)}")

# Encode target variable if it's categorical
le_target = None
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"Target encoding: {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled array back to DataFrame to retain column names for splitting
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model Training ---
model = LogisticRegression(solver='liblinear', max_iter=300, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_test, model.predict(X_test))

print(f"\n* Training Accuracy: {train_acc*100:.2f}%")
print(f"* Validation Accuracy: {val_acc*100:.2f}%")

print("\nClassification Report:")
# If target was encoded, use inverse_transform for class names in report
if le_target:
    print(classification_report(y_test, model.predict(X_test), target_names=le_target.classes_))
else:
    print(classification_report(y_test, model.predict(X_test)))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

# --- Save Model and Preprocessing Objects ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, 'breast_cancer_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'breast_cancer_scaler.pkl'))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'breast_cancer_label_encoders.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'breast_cancer_features.pkl'))

if le_target:
    joblib.dump(le_target, os.path.join(MODEL_DIR, 'breast_cancer_target_encoder.pkl'))

print(f"\nModel, scaler, and encoders saved in '{MODEL_DIR}/' folder.")
