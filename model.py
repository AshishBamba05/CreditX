# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

# Load and preprocess your data
df = pd.read_csv("Loan_default.csv")
df = df.fillna(df.median(numeric_only=True)).round(0)


def preprocess_data(df):
    df["HasCoSigner"] = df["HasCoSigner"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    df["HasMortgage"] = df["HasMortgage"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    df["Education"] = df["Education"].map({"Bachelor's": 0, "High School": 1, "Other": 2}).fillna(0).astype(int)
    df["HasDependents"] = df["HasDependents"].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    df["LoanPurpose"] = df["LoanPurpose"].map({'Business': 0, 'Home': 1, 'Other': 2}).fillna(0).astype(int)
    df["MaritalStatus"] = df["MaritalStatus"].map({'Married': 0, 'Divorced': 1, 'Other': 2}).fillna(0).astype(int)
    return df

df = preprocess_data(df)

print(df["Default"].describe())

continuous_features = [
    'Age',
    'Income',
    'LoanAmount',
    'CreditScore',
    'MonthsEmployed',
    'NumCreditLines',
    'InterestRate',
    'LoanTerm',
    'DTIRatio'
]

categorical_features = [
    'Education',
    'MaritalStatus',
    'HasMortgage',
    'HasDependents',
    'LoanPurpose',
    'HasCoSigner'
]

feature_names = continuous_features + categorical_features


X = df[feature_names]
y = df["Default"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.188, random_state=42)

# Preprocessing
scaler = StandardScaler()
preprocessor = ColumnTransformer([
    ('cont', scaler, continuous_features),
    ('cat', 'passthrough', categorical_features)
])

X_train_scaled = preprocessor.fit_transform(X_train)

# SMOTEENN balancing
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_scaled, y_train)

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    scale_pos_weight=len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_resampled, y_train_resampled)

# Save
joblib.dump(model, "model.joblib")
joblib.dump(preprocessor, "preprocessor.joblib")

# Evaluate performance
X_test_scaled = preprocessor.transform(X_test)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("=== Test Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred):.3f}")

# Optional: AUC
y_probs = model.predict_proba(X_test_scaled)[:, 1]
print(f"AUC Score: {roc_auc_score(y_test, y_probs):.3f}")