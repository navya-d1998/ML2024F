# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load the datasets
train_data_path = '/mnt/data/train_final.csv'
test_data_path = '/mnt/data/test_final.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Replace '?' with NaN for better handling of missing values
train_data.replace('?', pd.NA, inplace=True)
test_data.replace('?', pd.NA, inplace=True)

# Fill missing values with "Unknown" for categorical columns
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                    'relationship', 'race', 'sex', 'native.country']

for col in categorical_cols:
    train_data[col] = train_data[col].fillna("Unknown")
    test_data[col] = test_data[col].fillna("Unknown")

# Encode categorical features with LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_data = pd.concat([train_data[col], test_data[col]], axis=0).astype(str)
    le.fit(combined_data)
    train_data[col] = le.transform(train_data[col].astype(str))
    test_data[col] = le.transform(test_data[col].astype(str))
    label_encoders[col] = le

# Extract features and target variable from training data
X = train_data.drop(columns=['income>50K'])
y = train_data['income>50K']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=100,    # Number of boosting stages
    learning_rate=0.1,   # Shrinkage rate
    max_depth=5,         # Maximum depth of the individual estimators
    random_state=42
)

gb_model.fit(X_train, y_train)

# Validate the model
val_preds_gb = gb_model.predict_proba(X_val)[:, 1]
auc_score_gb = roc_auc_score(y_val, val_preds_gb)

# Drop the "Prediction" column from the test set to avoid feature mismatch
test_features = test_data.drop(columns=['ID', 'Prediction'], errors='ignore')

# Predict on the cleaned test set
test_preds_gb = gb_model.predict_proba(test_features)[:, 1]
test_data['Prediction'] = test_preds_gb

# Save the predictions to a CSV file
submission_path_gb_corrected = '/mnt/data/gradient_boost_submission_corrected.csv'
test_data[['ID', 'Prediction']].to_csv(submission_path_gb_corrected, index=False)

# Output validation AUC score and submission file path
print(f"Validation AUC Score: {auc_score_gb}")
print(f"Submission File Path: {submission_path_gb_corrected}")
