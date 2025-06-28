import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- Configuration ---
DATASET_PATH = 'Hemochromatosis - hemochromatosis-DOID_2352-genes-2023-11-17.csv'
# In this dataset, the 'Disease_Name' column contains the specific hemochromatosis types.
# We will use 'Disease_Name' as the target, and filter for 'hemochromatosis' vs. specific types.
# For a binary classification, we might classify 'hemochromatosis' as one class and
# specific types (e.g., 'hemochromatosis type 1', 'hemochromatosis type 2A') as another,
# or focus on predicting the specific type given the gene.
# Given the prompt, let's aim to predict the specific 'Disease_Name'. This makes it a multi-class classification.
# If a binary classification (hemochromatosis vs. not hemochromatosis) is desired,
# further logic would be needed to group disease names.
TARGET_COLUMN = 'Disease_Name'
# Columns to drop that are identifiers or redundant for direct ML training
IRRELEVANT_COLUMNS = ['Species_ID', 'Gene_ID', 'Genetic_Entity_ID', 'Genetic_Entity_Name',
                      'Evidence_Code', 'Evidence_Code_Name', 'Based_On_ID', 'Based_On_Name',
                      'Source', 'Reference', 'Date', 'Disease_ID']
MODEL_DIR = 'models'

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATASET_PATH)
    print("Hemochromatosis dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found.")
    print("Please ensure the file is in the correct directory.")
    raise

print("\nOriginal Dataset Head:")
print(df.head())
print("\nOriginal Dataset Info:")
df.info()

# --- 2. Initial Cleaning/Renaming ---
# Standardize column names: strip whitespace, replace spaces with underscores, remove special characters
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^0-9a-zA-Z_]', '', regex=True)
print("\nDataset Preview after column cleanup:")
print(df.head())

# --- 3. Identify and Prepare Target Variable ---
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset after column cleanup.")

print(f"\nIdentified Target Column: '{TARGET_COLUMN}'")

# Encode the target variable (Disease_Name) as it's categorical and multi-class
le_target = LabelEncoder()
df[TARGET_COLUMN] = le_target.fit_transform(df[TARGET_COLUMN])
print(f"Target encoding using LabelEncoder: {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")

y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN])

# --- 4. Drop irrelevant columns ---
# Drop columns identified as irrelevant for direct ML training
columns_to_drop_actual = [col for col in IRRELEVANT_COLUMNS if col in X.columns]
X.drop(columns=columns_to_drop_actual, inplace=True, errors='ignore')
print(f"\nDropped irrelevant columns: {columns_to_drop_actual}")

# --- 5. Handle Missing Values ---
print("\nHandling missing values in features...")
# Impute numerical columns with median
for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"  Imputed missing in numerical '{col}' with median: {median_val}")

# Impute categorical (object/string) columns with mode
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
# In this dataset, most "features" (like Gene_Symbol, Genetic_Entity_Type, Species_Name)
# are categorical or identifiers. After encoding, they become numerical.
# 'Species_Name' and 'Gene_Symbol' are now numerical representations of categories.
# 'Genetic_Entity_Type' is also categorical.
# Scaling these encoded categorical features is generally not recommended for tree-based models,
# but for Logistic Regression, it can sometimes help.
# Let's scale all numerical features that are not binary (0/1) if any exist.
print("\nScaling numerical features...")
numerical_features_to_scale = X.select_dtypes(include=np.number).columns.tolist()

# Filter out features that are already binary (e.g., from previous encoding of 2 categories)
# This is a heuristic; adjust if specific columns need to be excluded/included.
features_to_actually_scale = []
for col in numerical_features_to_scale:
    if X[col].nunique() > 2 or X[col].min() < 0 or X[col].max() > 1: # Not binary 0/1
        features_to_actually_scale.append(col)

if features_to_actually_scale:
    scaler = StandardScaler()
    X_scaled_part = scaler.fit_transform(X[features_to_actually_scale])
    X[features_to_actually_scale] = X_scaled_part
    print(f"  Scaled numerical features: {features_to_actually_scale}")
else:
    scaler = None # No scaler needed if no numerical features to scale
    print("  No continuous numerical features found to scale.")

print("Features after scaling (head):")
print(X.head())

# --- 8. Split Data into Training and Testing Sets ---
print("\nSplitting data into training and testing sets...")
# Stratify by 'y' (Disease_Name) to ensure balanced representation of different disease types
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Disease_Name distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Disease_Name distribution in testing set:\n{y_test.value_counts(normalize=True)}")

# --- 9. Train the ML Model ---
print("\nTraining the Logistic Regression model...")
# For multi-class classification, LogisticRegression uses 'ovr' (one-vs-rest) by default.
# 'liblinear' solver supports 'ovr' for multi-class.
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=500) # Increased max_iter for convergence
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
# Use target_names from le_target to show actual disease names in the report
print(classification_report(y_test, model.predict(X_test), target_names=le_target.classes_))

print("\nConfusion Matrix (Validation Set):")
print(confusion_matrix(y_test, model.predict(X_test)))

# --- 11. Save the Trained Model and Preprocessing Assets ---
os.makedirs(MODEL_DIR, exist_ok=True) # Create models directory if it doesn't exist

model_filename = os.path.join(MODEL_DIR, 'hemochromatosis_logistic_regression_model.pkl')
scaler_filename = os.path.join(MODEL_DIR, 'hemochromatosis_scaler.pkl') if scaler else None
label_encoders_filename = os.path.join(MODEL_DIR, 'hemochromatosis_label_encoders.pkl') if label_encoders else None
features_list_filename = os.path.join(MODEL_DIR, 'hemochromatosis_features.pkl')
target_encoder_filename = os.path.join(MODEL_DIR, 'hemochromatosis_target_encoder.pkl')

print(f"\nSaving model to {model_filename}")
joblib.dump(model, model_filename)

if scaler_filename:
    print(f"Saving scaler to {scaler_filename}")
    joblib.dump(scaler, scaler_filename)
else:
    print("No scaler saved as no continuous numerical features were scaled.")

if label_encoders_filename:
    print(f"Saving label encoders to {label_encoders_filename}")
    joblib.dump(label_encoders, label_encoders_filename)
else:
    print("No categorical features were encoded, so no label encoders saved.")

print(f"Saving feature names to {features_list_filename}")
joblib.dump(X.columns.tolist(), features_list_filename)

print(f"Saving target encoder to {target_encoder_filename}")
joblib.dump(le_target, target_encoder_filename)

print("\nModel training, evaluation, and saving complete!")
print(f"The trained model and relevant preprocessing assets are saved in the '{MODEL_DIR}' directory.")
from sklearn.ensemble import RandomForestClassifier

print("\nTraining a Random Forest Classifier...")
# Initialize Random Forest model
# n_estimators: number of trees in the forest
# class_weight: 'balanced' automatically adjusts weights inversely proportional to class frequencies
# You would tune these parameters further with GridSearchCV/RandomizedSearchCV
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train_resampled if 'X_train_resampled' in locals() else X_train,
          y_train_resampled if 'y_train_resampled' in locals() else y_train)
print("Random Forest model training complete.")

# Evaluate as before
y_train_pred = model.predict(X_train) # Use original X_train for evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy (RF): {train_accuracy * 100:.2f}%")

y_test_pred = model.predict(X_test)
validation_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy (RF): {validation_accuracy * 100:.2f}%")

print("\nClassification Report (RF Validation Set):")
print(classification_report(y_test, model.predict(X_test), target_names=le_target.classes_))

from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution before SMOTE
print("Class distribution before SMOTE:", Counter(y_train))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_resampled))
