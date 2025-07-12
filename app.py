import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from db_utils import insert_prediction, fetch_predictions
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# --- Load & Prepare Dataset ---
df = pd.read_csv("credit_score.csv")

df = df.fillna(df.median(numeric_only=True)).round(0)

FEATURES = [
    'R_DEBT_INCOME',
    'T_EXPENDITURE_12',
    'R_DEBT_SAVINGS'
]

X = df[FEATURES]
y = df["CREDIT_SCORE"]

# --- Train-Test Split + Scaling ---
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train Model ---
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Feature Importance Plot ---
importances = model.feature_importances_
feat_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feat_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

st.subheader("Feature Importance")
st.pyplot(plt)

# --- Evaluation Metrics ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MAE: {mae:.2f}")
print(f"✅ MSE: {mse:.2f}")
print(f"✅ R² Score: {r2:.3f}")

# --- Streamlit UI ---
st.title("Credit Score Predictor")

income = st.number_input("Annual Income", min_value=0.0)
debt = st.number_input("Total Debt", min_value=0.0)
savings = st.number_input("Total Savings", min_value=0.0)
expenditure = st.number_input("Annual Expenditure", min_value=0.0)

# --- Feature Engineering ---
r_debt_income = debt / (income + 1)
t_expenditure_12 = expenditure
r_debt_savings = debt / (savings + 1)

user_input = [[
    r_debt_income,
    t_expenditure_12,
    r_debt_savings
]]

def classify_score(score):
    if score >= 800:
        return "🟢 Excellent"
    elif score >= 740:
        return "🟢 Very Good"
    elif score >= 670:
        return "🟡 Good"
    elif score >= 580:
        return "🟠 Fair"
    else:
        return "🔴 Poor"

if st.button("Predict My Credit Score"):
    user_input_scaled = scaler.transform(user_input)
    prediction = int(model.predict(user_input_scaled)[0])
    
    label = classify_score(prediction)
    st.success(f"Estimated Credit Score: {prediction}  \nCategory: {label}")

    insert_prediction(income, debt, savings, expenditure, prediction)


# --- Display Past Predictions ---
st.subheader("Recent Predictions")
try:
    prediction_df = fetch_predictions()
    st.dataframe(prediction_df)
except Exception as e:
    st.error(f"Could not load prediction history: {e}")


