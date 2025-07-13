import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVR(kernel='rbf', C=150, epsilon=17)
svm_model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = svm_model.predict(X_test)
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

if st.button("Predict My Credit Score"):
    # --- Feature Engineering ---
    r_debt_income = debt / (income + 1)
    t_expenditure_12 = expenditure
    r_debt_savings = debt / (savings + 1)

    user_input = [[r_debt_income, t_expenditure_12, r_debt_savings]]
    user_input_scaled = scaler.transform(user_input)

    prediction = int(round(svm_model.predict(user_input_scaled)[0]))

    # Store prediction
    insert_prediction(
        income, debt, savings, expenditure,
        r_debt_income, t_expenditure_12,
        r_debt_savings, prediction
    )

    st.success(f"Estimated Credit Score: {prediction}")

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

# --- Prediction History ---
st.subheader("Prediction History")
prediction_df = fetch_predictions()

# Apply color to score column using score_category
def style_by_category(category):
    if 'Excellent' in category:
        return 'color: green'
    elif 'Very Good' in category:
        return 'color: limegreen'
    elif 'Good' in category:
        return 'color: goldenrod'
    elif 'Fair' in category:
        return 'color: darkorange'
    elif 'Poor' in category:
        return 'color: red'
    return ''

if not prediction_df.empty:
    styled_df = prediction_df.style.apply(
        lambda row: [
            style_by_category(row['score_category']) if col == 'score' else ''
            for col in prediction_df.columns
        ],
        axis=1
    )
    st.dataframe(styled_df)
else:
    st.info("No predictions yet.")

# --- Drop & Recreate Table ---
if st.button("Drop & Recreate Prediction Table"):
    conn = sqlite3.connect("creditx_predictions.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS predictions")
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            income FLOAT,
            debt FLOAT,
            savings FLOAT,
            expenditure FLOAT,
            r_debt_income FLOAT,
            t_expenditure_12 FLOAT,
            r_debt_savings FLOAT,
            score FLOAT
        )
    """)
    conn.commit()
    conn.close()

    st.success("✅ Table has been dropped and recreated.")
    st.dataframe(pd.DataFrame(columns=[
        "id", "timestamp", "income", "debt", "savings",
        "expenditure", "r_debt_income", "t_expenditure_12",
        "r_debt_savings", "score"
    ]))
