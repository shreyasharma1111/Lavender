import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving and loading the model and scaler


try:
    df = pd.read_csv('pcos_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: pcos_dataset.csv not found. Please ensure the file is in the correct directory.")
    raise

print("\nOriginal Dataset Head:")
print(df.head())
print("\nOriginal Dataset Info:")
df.info()

# Standardize column names by stripping whitespace and replacing spaces with underscores
df.columns = df.columns.str.strip().str.replace(' ', '_')
# Further clean column names by removing special characters like parentheses and their content
df.columns = df.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()


# Check for the target variable 'PCOS_Diagnosis' and convert it to numerical
if 'PCOS_Diagnosis' in df.columns:
    # Assuming 'PCOS_Diagnosis' is already numerical (0 or 1) based on the provided dataset sample
    # If it were 'Y'/'N', the original mapping would be needed: df['PCOS'] = df['PCOS_Diagnosis'].map({'Y': 1, 'N': 0})
    df['PCOS'] = df['PCOS_Diagnosis'] # Directly use the column as it's already 0 or 1
    df = df.drop(columns=['PCOS_Diagnosis']) # Drop the original column
    print("\nTarget variable 'PCOS' created from 'PCOS_Diagnosis'.")
else:
    print("Warning: 'PCOS_Diagnosis' column not found. Assuming 'PCOS' is already numerical.")
    if 'PCOS' not in df.columns:
        print("Error: Target 'PCOS' column not found.")
        raise ValueError("Target 'PCOS' column not found in the dataset.")

# Define columns to drop based on the dataset provided (no 'Sl_No', 'Patient_File_No' in this specific dataset)
columns_to_drop = [col for col in ['Sl_No', 'Patient_File_No'] if col in df.columns]
df = df.drop(columns=columns_to_drop, errors='ignore') # errors='ignore' prevents error if column not found

# Identify numerical and categorical features based on the provided dataset
# The dataset has 'Age', 'BMI', 'Menstrual_Irregularity', 'Testosterone_Level(ng/dL)', 'Antral_Follicle_Count', 'PCOS_Diagnosis'
# After cleaning, column names become 'Age', 'BMI', 'Menstrual_Irregularity', 'Testosterone_Level', 'Antral_Follicle_Count', 'PCOS'

numerical_features = ['Age', 'BMI', 'Testosterone_Level', 'Antral_Follicle_Count']
categorical_features = ['Menstrual_Irregularity'] # This column appears to be binary (0/1) but can be treated as categorical for encoding if needed, or directly as numerical. Given the context, it's likely a binary categorical.

# Filter existing columns from dataset to ensure they are present
numerical_features = [f for f in numerical_features if f in df.columns]
categorical_features = [f for f in categorical_features if f in df.columns]

print(f"\nNumerical features selected: {numerical_features}")
print(f"Categorical features selected: {categorical_features}")

print("\nHandling missing values...")
for col in numerical_features:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Imputed missing values in numerical column '{col}' with median: {median_val}")

for col in categorical_features:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Imputed missing values in categorical column '{col}' with mode: {mode_val}")

print("\nMissing values after imputation:")
print(df.isnull().sum())

print("\nEncoding categorical features...")
encoded_features = []
label_encoders = {} # Store encoders for later use in prediction

for col in categorical_features:
    le = LabelEncoder()
    # Check if the column exists and has values to encode
    if col in df.columns and len(df[col].unique()) > 1:
        df[f'{col}_Encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        encoded_features.append(f'{col}_Encoded')
        print(f"  Encoded '{col}'. Mapping: {list(le.classes_)} -> {list(range(len(le.classes_)))}")
    elif col in df.columns and len(df[col].unique()) == 1:
         # If only one unique value, treat as constant, set encoded value to 0
         df[f'{col}_Encoded'] = 0
         encoded_features.append(f'{col}_Encoded')
         print(f"  '{col}' has only one unique value, encoded to 0.")


# Combine all features for X
all_features = numerical_features + encoded_features
X = df[all_features]
y = df['PCOS'] # Target variable

print("\nFeatures after encoding (head):")
print(X.head())

print("\nScaling numerical features...")
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("Features after scaling (head):")
print(X.head())


print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"PCOS prevalence in training set: {y_train.sum()/len(y_train):.2f}")
print(f"PCOS prevalence in testing set: {y_test.sum()/len(y_test):.2f}")


print("\nTraining the Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=200) # Increased max_iter for convergence
model.fit(X_train, y_train)
print("Model training complete.")

print("\nEvaluating the model...")
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_test_pred = model.predict(X_test)
validation_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

print("\nClassification Report (Validation Set):")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix (Validation Set):")
print(confusion_matrix(y_test, y_test_pred))

os.makedirs('models', exist_ok=True)

model_filename = 'models/pcos_logistic_regression_model.pkl'
scaler_filename = 'models/pcos_scaler.pkl'
label_encoders_filename = 'models/pcos_label_encoders.pkl'
features_list_filename = 'models/pcos_features.pkl' # To save the order of features

print(f"\nSaving model to {model_filename}")
joblib.dump(model, model_filename)

print(f"Saving scaler to {scaler_filename}")
joblib.dump(scaler, scaler_filename)

print(f"Saving label encoders to {label_encoders_filename}")
joblib.dump(label_encoders, label_encoders_filename)

print(f"Saving feature names to {features_list_filename}")
joblib.dump(X.columns.tolist(), features_list_filename) # Save feature names in correct order


print("\nModel training, evaluation, and saving complete!")
print("The trained model, scaler, and label encoders are saved in the 'models' directory.")
