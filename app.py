import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
import sqlite3
import plotly.graph_objects as go

from db_utils import (
    insert_prediction,
    fetch_predictions,
    fetch_latest_score_with_label,
    run_sql_query_from_file,
    get_connection
)

# --- DB Connection ---
conn = get_connection()
cursor = conn.cursor()

# --- Load & Prepare Dataset ---
df = pd.read_csv("credit_score.csv")
df = df.fillna(df.median(numeric_only=True)).round(0)

FEATURES = [
    'R_DEBT_INCOME', 'T_EXPENDITURE_12',
    'T_HEALTH_12', 'T_GAMBLING_12',
    'CAT_SAVINGS_ACCOUNT', 'R_HOUSING_DEBT'
]

X = df[FEATURES]
y = df["CREDIT_SCORE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVR(kernel='rbf', C=95, epsilon=23, gamma=0.1, verbose=False)
svm_model.fit(X_train, y_train)

# --- Model Evaluation ---
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Test MAE:  {mae_test:.2f}, Test MSE:  {mse_test:.2f}")
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"Test R² Score:  {r2_test:.3f}")

# --- Streamlit UI ---
st.title("Credit Score Predictor | By Ashish V Bamba")

income = st.number_input("Annual Income", min_value=0.0)
debt = st.number_input("Total Debt", min_value=0.0)
expenditure = st.number_input("Annual Expenditure", min_value=0.0)
health = st.number_input("Annual Health Spend", min_value=0.0)
gambling = st.number_input("Annual Gambling Spend", min_value=0.0)
housing = st.number_input("Annual Housing Spend", min_value=0.0)
has_savings = st.checkbox("Do you have a savings account?")

if st.button("Predict My Credit Score"):
    r_debt_income = debt / (income + 1)
    t_expenditure_12 = expenditure
    t_health_12 = health
    t_gambling_12 = gambling
    r_housing_debt = housing / (debt + 1)
    cat_savings_account = 1 if has_savings else 0

    user_input = [[
        r_debt_income, t_expenditure_12,
        t_health_12, t_gambling_12,
        cat_savings_account, r_housing_debt
    ]]
    user_input_scaled = scaler.transform(user_input)

    prediction = int(round(svm_model.predict(user_input_scaled)[0]))
    prediction = max(300, min(850, prediction))

    insert_prediction(
        income, debt, expenditure,
        r_debt_income, t_expenditure_12,
        t_health_12, t_gambling_12,
        cat_savings_account, r_housing_debt,
        prediction
    )

    latest = fetch_latest_score_with_label()
    score = int(latest['score'][0])
    label = latest['score_category'][0]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "green"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 580], 'color': "#FF4B4B"},
                {'range': [580, 670], 'color': "#FFA500"},
                {'range': [670, 740], 'color': "#FFD700"},
                {'range': [740, 800], 'color': "#90EE90"},
                {'range': [800, 850], 'color': "#00FF00"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Estimated Credit Score: {score}  \nCategory: {label}")

    sql = run_sql_query_from_file("ex_queries.sql", "classify_user",
                                  income=income,
                                  debt=debt,
                                  expenditure=expenditure)

    if sql:
        bracket_df = pd.read_sql_query(sql, conn)
        st.subheader("Your Bracket Classification")
        st.write("Income Bracket:", bracket_df["income_bracket"][0])
        st.write("Debt Bracket:", bracket_df["debt_bracket"][0])
        st.write("Expenditure Bracket:", bracket_df["expenditure_bracket"][0])
    else:
        st.error("⚠️ Couldn't find classification query in ex_queries.sql.")

# --- Prediction History ---
st.subheader("Prediction History")
prediction_df = fetch_predictions()

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
            expenditure FLOAT,
            r_debt_income FLOAT,
            t_expenditure_12 FLOAT,
            t_health_12 FLOAT,
            t_gambling_12 FLOAT,
            cat_savings_account INTEGER,
            r_housing_debt FLOAT,
            score FLOAT
        )
    """)

    conn.commit()
    conn.close()

    st.success("✅ Table has been dropped and recreated.")
    st.dataframe(pd.DataFrame(columns=[
        "id", "timestamp", "income", "debt", "expenditure",
        "r_debt_income", "t_expenditure_12", "t_health_12",
        "t_gambling_12", "cat_savings_account", "r_housing_debt", "score"
    ]))
