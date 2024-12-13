import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# Load the datasets
train_data = pd.read_csv('/mnt/data/train_final.csv')
test_data = pd.read_csv('/mnt/data/test_final.csv')

# Replace '?' with NaN and handle missing values
train_data.replace('?', pd.NA, inplace=True)
test_data.replace('?', pd.NA, inplace=True)

# Categorical columns
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

# Extract features and target variable
X = train_data.drop(columns=['income>50K'])
y = train_data['income>50K']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a tuned Gradient Boosting model
gb_model_tuned = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    random_state=42
)
gb_model_tuned.fit(X_train, y_train)

# Validate the model
val_preds_gb_tuned = gb_model_tuned.predict_proba(X_val)[:, 1]
auc_score_gb_tuned = roc_auc_score(y_val, val_preds_gb_tuned)
print(f"Validation AUC Score: {auc_score_gb_tuned}")

# Predict on the test set
test_features = test_data.drop(columns=['ID'], errors='ignore')
test_preds_gb_tuned = gb_model_tuned.predict_proba(test_features)[:, 1]
test_data['Prediction'] = test_preds_gb_tuned

# Save the predictions to a CSV file
submission_path_gb_tuned = '/mnt/data/gradient_boost_submission_tuned.csv'
test_data[['ID', 'Prediction']].to_csv(submission_path_gb_tuned, index=False)
print(f"Submission File Saved: {submission_path_gb_tuned}")
