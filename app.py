import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import numpy as np

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

df["R_EXPENDITURE"] = df["T_EXPENDITURE_6"] / (df["T_EXPENDITURE_12"] + 1)
df["R_EDUCATION"] = df["T_EDUCATION_6"] / (df["T_EDUCATION_12"] + 1)
df["R_EDUCATION"] = df["R_EDUCATION"].clip(upper=1.5)


FEATURES = [
    'R_DEBT_INCOME', 
    'T_EXPENDITURE_12',
    'T_GAMBLING_12',
    'CAT_SAVINGS_ACCOUNT', 'R_HOUSING_DEBT',
    'R_EXPENDITURE', 'R_EDUCATION'
]

# Oversample high-credit samples
df_high = df[df["CREDIT_SCORE"] >= 700]
df_balanced = pd.concat([
    df,
    df_high.sample(n=200, replace=True, random_state=42)
])

np.random.seed(42)

highCred_samples = []
for _ in range(10):
    highCred_samples.append({
        "R_DEBT_INCOME": np.random.uniform(0.0, 0.02),
        "T_EXPENDITURE_12": np.random.uniform(35000, 48000),
        "T_GAMBLING_12": 0,
        "CAT_SAVINGS_ACCOUNT": 1,
        "R_HOUSING_DEBT": np.random.uniform(0.65, 0.78),
        "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
        "R_EDUCATION": np.random.uniform(0.48, 0.52),
        "CREDIT_SCORE": np.random.randint(810, 850)
    })
df_highCred = pd.DataFrame(highCred_samples)
df_balanced = pd.concat([df_balanced, df_highCred], ignore_index=True)


goodSavings_samples = []
for _ in range(10):
    goodSavings_samples.append({
        "R_DEBT_INCOME": 0.0,
        "T_EXPENDITURE_12": 0.0,
        "T_GAMBLING_12": 0.0,
        "CAT_SAVINGS_ACCOUNT": 1,
        "R_HOUSING_DEBT": 0.0,
        "R_EXPENDITURE": 0.0,
        "R_EDUCATION": 0.0,
        "CREDIT_SCORE": 835
    })

df_goodSavings = pd.DataFrame(goodSavings_samples)
df_balanced = pd.concat([df_balanced, df_goodSavings], ignore_index=True)


badSavings_samples = []
for _ in range(20):
    badSavings_samples.append({
        "R_DEBT_INCOME": np.random.uniform(2.5, 4.5),
        "T_EXPENDITURE_12": np.random.uniform(5000, 12000),
        "T_GAMBLING_12": np.random.uniform(3000, 6000),
        "CAT_SAVINGS_ACCOUNT": 1,
        "R_HOUSING_DEBT": np.random.uniform(0.9, 1.2),
        "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
        "R_EDUCATION": np.random.uniform(0.2, 0.4),
        "CREDIT_SCORE": np.random.randint(300, 520)
    })

df_badSavings = pd.DataFrame(badSavings_samples)
df_balanced = pd.concat([df_balanced, df_badSavings], ignore_index=True)


midCredit_samples = []
for _ in range(20):
    midCredit_samples.append({
        "R_DEBT_INCOME": np.random.uniform(0.01, 0.03),
        "T_EXPENDITURE_12": np.random.uniform(25000, 40000),
        "T_GAMBLING_12": 0,
        "CAT_SAVINGS_ACCOUNT": 0,
        "R_HOUSING_DEBT": np.random.uniform(0.6, 0.7),
        "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
        "R_EDUCATION": np.random.uniform(0.5, 0.6),
        "CREDIT_SCORE": np.random.randint(740, 800)
    })

df_midCredit_samples = pd.DataFrame(midCredit_samples)
df_balanced = pd.concat([df_balanced, df_midCredit_samples], ignore_index=True)

# Add 15 synthetics for very risky profiles
poor_samples = []
for _ in range(35):
    poor_samples.append({
        "R_DEBT_INCOME": np.random.uniform(1.5, 5.0),  # Very high ratio
        "T_EXPENDITURE_12": np.random.uniform(5000, 15000),
        "T_GAMBLING_12": np.random.uniform(2000, 5000),
        "CAT_SAVINGS_ACCOUNT": 0,
        "R_HOUSING_DEBT": np.random.uniform(0.8, 1.2),
        "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
        "R_EDUCATION": np.random.uniform(0.48, 0.52),
        "CREDIT_SCORE": np.random.randint(300, 520)  # Force into poor
    })

df_poor = pd.DataFrame(poor_samples)
df_balanced = pd.concat([df_balanced, df_poor], ignore_index=True)

debt_only_samples = []
for _ in range(35):
    income = 0
    debt = np.random.uniform(50000, 150000)
    debt_only_samples.append({
        "R_DEBT_INCOME": min(debt / (income + 1), 5),
        "T_EXPENDITURE_12": 0,
        "T_GAMBLING_12": 0,
        "CAT_SAVINGS_ACCOUNT": 0,
        "R_HOUSING_DEBT": np.log1p(min(np.random.uniform(0, 1.5), 1.24)),
        "R_EXPENDITURE": 0,
        "R_EDUCATION": 0,
        "CREDIT_SCORE": np.random.randint(300, 520)
    })
df_debt_only = pd.DataFrame(debt_only_samples)
df_balanced = pd.concat([df_balanced, df_debt_only], ignore_index=True)

# --- Synthetic: Gambling-Heavy Risk Profiles ---
gambling_risk_samples = []
for _ in range(30):
    gambling_risk_samples.append({
        "R_DEBT_INCOME": np.random.uniform(1.0, 3.5),
        "T_EXPENDITURE_12": np.random.uniform(15000, 30000),
        "T_GAMBLING_12": np.random.uniform(8000, 20000),
        "CAT_SAVINGS_ACCOUNT": 0,
        "R_HOUSING_DEBT": np.random.uniform(0.85, 1.3),
        "R_EXPENDITURE": np.random.uniform(0.45, 0.55),
        "R_EDUCATION": np.random.uniform(0.35, 0.50),
        "CREDIT_SCORE": np.random.randint(300, 600)
    })

df_gambling = pd.DataFrame(gambling_risk_samples)
df_balanced = pd.concat([df_balanced, df_gambling], ignore_index=True)

excellent_samples = []
for _ in range(30):
    excellent_samples.append({
        "R_DEBT_INCOME": np.random.uniform(0.01, 0.05),
        "T_EXPENDITURE_12": np.random.uniform(38000, 50000),
        "T_GAMBLING_12": 0,
        "CAT_SAVINGS_ACCOUNT": 1,
        "R_HOUSING_DEBT": np.random.uniform(0.60, 0.75),
        "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
        "R_EDUCATION": np.random.uniform(0.45, 0.55),
        "CREDIT_SCORE": np.random.randint(810, 850)
    })

df_excellent = pd.DataFrame(excellent_samples)
df_balanced = pd.concat([df_balanced, df_excellent], ignore_index=True)


X = df_balanced[FEATURES]
y = df_balanced["CREDIT_SCORE"]
print(df_balanced["CREDIT_SCORE"].describe())

# Separate continuous vs. binary categorical feature
X_continuous = X.drop(columns=["CAT_SAVINGS_ACCOUNT"])
X_cat = X[["CAT_SAVINGS_ACCOUNT"]].values  # keep as raw 0/1

# Train/test split
X_train_cont, X_test_cont, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_continuous, X_cat, y, test_size=0.3, random_state=42
)

# Scale only continuous features
scaler = StandardScaler()
X_train_scaled_cont = scaler.fit_transform(X_train_cont)
X_test_scaled_cont = scaler.transform(X_test_cont)

# Combine scaled continuous + unscaled categorical
X_train_final = np.hstack([X_train_scaled_cont, X_train_cat])
X_test_final = np.hstack([X_test_scaled_cont, X_test_cat])

rf_model = RandomForestRegressor(n_estimators=115, max_depth=8, random_state=42)
rf_model.fit(X_train_final, y_train)

# --- Model Evaluation ---
y_train_pred = rf_model.predict(X_train_final)
y_test_pred = rf_model.predict(X_test_final)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test MAE:  {mae_test:.2f}, Test MSE:  {mse_test:.2f}")
print(f"Test R² Score:  {r2_test:.3f}")

# --- Streamlit UI ---
st.title("Credit Score Predictor | By Ashish V Bamba")

income = st.number_input("Annual Income", min_value=0.0)
debt = st.number_input("Total Debt", min_value=0.0)
expenditure_12 = st.number_input("12-Month Expenditure", min_value=0.0)
expenditure_6 = st.number_input("6-Month Expenditure", min_value=0.0)
gambling = st.number_input("Annual Gambling Spend", min_value=0.0)
housing = st.number_input("Annual Housing Spend", min_value=0.0)
education_12 = st.number_input("12-Month Education Spend", min_value=0.0)
education_6 = st.number_input("6-Month Education Spend", min_value=0.0)
has_savings = st.checkbox("Do you have a savings account?")

if st.button("Predict My Credit Score"):
    r_debt_income = np.log1p(debt / (income + 1))
    t_expenditure_12 = expenditure_12
    t_expenditure_6 = expenditure_6
    t_gambling_12 = gambling
    r_housing_debt = np.log1p(min(housing / (debt + 1), 1.24))
    cat_savings_account = 1 if has_savings else 0
    r_expenditure = t_expenditure_6 / (t_expenditure_12 + 1)
    r_education = education_6 / (education_12 + 1)

    user_input_cont = [[
    r_debt_income, t_expenditure_12,
    t_gambling_12,
    r_housing_debt,
    r_expenditure, r_education
]]
    user_input_cat = [[cat_savings_account]]

    user_input_scaled_cont = scaler.transform(user_input_cont)
    user_input_final = np.hstack([user_input_scaled_cont, user_input_cat])

    print("Raw input (cont):", user_input_cont)
    print("Raw input (cat):", user_input_cat)
    print("Full input:", user_input_final)

    prediction = int(round(rf_model.predict(user_input_final)[0]))

    prediction = max(300, min(850, prediction))

    insert_prediction(
        income, debt, 
        r_debt_income, t_expenditure_12,
        t_gambling_12,
        cat_savings_account, r_housing_debt,
        r_expenditure, r_education, prediction
    )

    zero_fields = 0

    fields_to_check = [
        income, debt, expenditure_12, expenditure_6,  
        gambling, housing, education_12, education_6, has_savings
    ]

    for val in fields_to_check:
        if val == 0:
            zero_fields += 1

    if zero_fields >= 5:
        st.warning("⚠️ Your inputs contain many zero values. This may result in an inaccurate prediction. Try entering realistic estimates for typical expenses (e.g. health, housing, education).")


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
                                  expenditure=expenditure_12)

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
            t_expenditure_12 FLOAT,
            r_debt_income FLOAT,
            t_gambling_12 FLOAT,
            cat_savings_account INTEGER,
            r_housing_debt FLOAT,
            r_expenditure FLOAT,
            r_education FLOAT,
            score FLOAT
        )
    """)

    conn.commit()
    conn.close()

    st.success("✅ Table has been dropped and recreated.")
    st.dataframe(pd.DataFrame(columns=[
        "id", "timestamp", "income", "debt",
        "r_debt_income", "t_expenditure_12",
        "t_gambling_12", "cat_savings_account", "r_housing_debt",
        "r_expenditure", "r_education", "score"
    ]))