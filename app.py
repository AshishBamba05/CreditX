import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR


from db_utils import (
    insert_prediction,
    fetch_predictions,
    run_sql_query_from_file,
    get_connection
)

# --- DB Connection ---
conn = get_connection()
cursor = conn.cursor()

# --- Load & Prepare Dataset ---
df = pd.read_csv("credit_score.csv")
df = df.fillna(df.median(numeric_only=True)).round(0)

FEATURES = ['R_DEBT_INCOME', 'T_EXPENDITURE_12', 'R_DEBT_SAVINGS']
X = df[FEATURES]
y = df["CREDIT_SCORE"]

#rus = RandomUnderSampler(random_state=42)
#X_resampled, y_resampled = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVR(kernel='rbf', C=150, epsilon=17)
svm_model.fit(X_train, y_train)

# --- Feature Importance ---
#importances = model.feature_importances_
#feat_names = X.columns

#plt.figure(figsize=(10, 6))
#plt.barh(feat_names, importances)
#plt.xlabel("Feature Importance")
#plt.title("Random Forest Feature Importance")
#plt.tight_layout()
#st.subheader("Feature Importance")
#st.pyplot(plt)

# --- Model Evaluation ---
y_pred = svm_model.predict(X_test)
#y_pred = model.predict(X_test)
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
    # --- Feature Engineering ---
    r_debt_income = debt / (income + 1)
    t_expenditure_12 = expenditure
    r_debt_savings = debt / (savings + 1)

    user_input = [[r_debt_income, t_expenditure_12, r_debt_savings]]
    user_input_scaled = scaler.transform(user_input)

    prediction = int(round(svm_model.predict(user_input_scaled)[0]))
    label = classify_score(prediction)

    st.success(f"Estimated Credit Score: {prediction}  \nCategory: {label}")

    insert_prediction(income, debt, savings, expenditure, prediction)

    # --- Bracket Classification ---
    sql = run_sql_query_from_file("ex_queries.sql", "classify_user",
                                  income=income,
                                  debt=debt,
                                  savings=savings,
                                  expenditure=expenditure)

    if sql:
        bracket_df = pd.read_sql_query(sql, conn)
        st.subheader("Your Bracket Classification")
        st.write("Income Bracket:", bracket_df["income_bracket"][0])
        st.write("Debt Bracket:", bracket_df["debt_bracket"][0])
        st.write("Savings Bracket:", bracket_df["savings_bracket"][0])
        st.write("Expenditure Bracket:", bracket_df["expenditure_bracket"][0])
    else:
        st.error("⚠️ Couldn't find classification query in ex_queries.sql.")


st.subheader("Recent Scores with Category Labels")
recent_scores_sql = run_sql_query_from_file("ex_queries.sql", "recent_scores_with_labels")

if recent_scores_sql:
    df = pd.read_sql_query(recent_scores_sql, conn)
    st.dataframe(df)
else:
    st.error("⚠️ Failed to load recent score classifications.")


# --- Prediction History ---
st.subheader("Prediction History")
prediction_df = fetch_predictions()

if not prediction_df.empty:
    st.dataframe(prediction_df)
else:
    st.info("No predictions found yet.")