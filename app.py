import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
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
df = df.fillna(df.mean(numeric_only=True)).round(0)

df["R_EXPENDITURE"] = df["T_EXPENDITURE_6"] / (df["T_EXPENDITURE_12"] + 1)
df["R_EDUCATION"] = df["T_EDUCATION_6"] / (df["T_EDUCATION_12"] + 1)

FEATURES = [
    'R_DEBT_INCOME', 
    'T_GAMBLING_12',
    'SAVINGS',
    'CAT_CREDIT_CARD',      
    'R_EXPENDITURE',
    'R_EDUCATION',
]

poor_samples = []
for _ in range(35):
   poor_samples.append({
       "R_DEBT_INCOME": np.random.uniform(1.5, 5.0),  # Very high ratio
       "T_GAMBLING_12": np.random.uniform(2000, 5000),
       "SAVINGS_ACCOUNT": 0,
       "CAT_CREDIT_CARD": np.random.choice([0, 1]),
       "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
       "R_EDUCATION": np.random.uniform(0.3, 0.6),
       "DEFAULT": 1
   })

df_poor = pd.DataFrame(poor_samples)
df_balanced = pd.concat([df, df_poor], ignore_index=True)

debt_only_samples = []
for _ in range(35):
   income = 0
   debt = np.random.uniform(50000, 150000)
   debt_only_samples.append({
       "R_DEBT_INCOME": min(debt / (income + 1), 5),
       "T_GAMBLING_12": 0,
       "SAVINGS": 0,
       "CAT_CREDIT_CARD": np.random.choice([0, 1]),
       "R_EXPENDITURE": 0,
       "R_EDUCATION": np.random.uniform(0.3, 0.6),
       "DEFAULT": 1
   })
df_debt_only = pd.DataFrame(debt_only_samples)
df_balanced = pd.concat([df_balanced, df_debt_only], ignore_index=True)

# --- Synthetic: Gambling-Heavy Risk Profiles ---
gambling_risk_samples = []
for _ in range(30):
   gambling_risk_samples.append({
       "R_DEBT_INCOME": np.random.uniform(1.0, 3.5),
       "T_GAMBLING_12": np.random.uniform(8000, 20000),
       "SAVINGS": 0,
       "CAT_CREDIT_CARD": np.random.choice([0, 1]),
       "R_EXPENDITURE": np.random.uniform(0.45, 0.55),
       "R_EDUCATION": np.random.uniform(0.3, 0.6),
       "DEFAULT": 1
   })

df_gambling = pd.DataFrame(gambling_risk_samples)
df_balanced = pd.concat([df_balanced, df_gambling], ignore_index=True)

midCredit_defaulters = []
for _ in range(20):
    midCredit_defaulters.append({
        "R_DEBT_INCOME": np.random.uniform(1.2, 2.0),  # Not extreme, but risky
        "T_EXPENDITURE_12": np.random.uniform(22000, 35000),
        "T_GAMBLING_12": np.random.uniform(500, 1500),
        "SAVINGS": np.random.uniform(5000, 25000),
        "CAT_CREDIT_CARD": np.random.choice([0, 1]),
        "R_EXPENDITURE": np.random.uniform(0.50, 0.65),  # High spend rate
        "R_EDUCATION": np.random.uniform(0.3, 0.5),
        "DEFAULT": 1
    })

df_midCredit_defaulters = pd.DataFrame(midCredit_defaulters)
df_balanced = pd.concat([df_balanced, df_midCredit_defaulters], ignore_index=True)

excellent_samples = []
for _ in range(30):
    excellent_samples.append({
        "R_DEBT_INCOME": np.random.uniform(0.01, 0.05),
        "T_EXPENDITURE_12": np.random.uniform(38000, 50000),
        "T_GAMBLING_12": 0,
        "SAVINGS": np.random.uniform(5000, 25000),
        "CAT_CREDIT_CARD": 1,
        "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
        "R_EDUCATION": np.random.uniform(0.5, 0.7),
        "DEFAULT": 0  # important for classification
    })

df_excellent = pd.DataFrame(excellent_samples)
df_balanced = pd.concat([df_balanced, df_excellent], ignore_index=True)

highCred_samples = []
for _ in range(10):
    highCred_samples.append({
        "R_DEBT_INCOME": np.random.uniform(0.0, 0.02),
        "T_EXPENDITURE_12": np.random.uniform(35000, 48000),
        "T_GAMBLING_12": 0,
        "SAVINGS": np.random.uniform(5000, 25000),
        "CAT_CREDIT_CARD": 1,
        "R_EXPENDITURE": np.random.uniform(0.48, 0.52),
        "R_EDUCATION": np.random.uniform(0.5, 0.7),
        "R_HEALTH": np.random.uniform(0.5, 0.7),
        "DEFAULT": 0
    })

df_highCred = pd.DataFrame(highCred_samples)
df_balanced = pd.concat([df_balanced, df_highCred], ignore_index=True)


X = df_balanced[FEATURES]
y = df_balanced["DEFAULT"]

print(df_balanced["DEFAULT"].describe())

# Separate continuous vs. binary categorical feature
X_continuous = X.drop(columns=["CAT_CREDIT_CARD"])
X_cat = X[["CAT_CREDIT_CARD"]].values

# Train/test split
X_train_cont, X_test_cont, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_continuous, X_cat, y, test_size=0.25, random_state=42
)

# Scale only continuous features
scaler = StandardScaler()
X_train_scaled_cont = scaler.fit_transform(X_train_cont)
X_test_scaled_cont = scaler.transform(X_test_cont)

# Combine scaled continuous + unscaled categorical
X_train_final = np.hstack([X_train_scaled_cont, X_train_cat])
X_test_final = np.hstack([X_test_scaled_cont, X_test_cat])

rf_model = RandomForestClassifier(n_estimators=115, max_depth=8, class_weight='balanced', random_state=42)
rf_model.fit(X_train_final, y_train)

# --- Feature Importance ---
importances = rf_model.feature_importances_
feature_names = FEATURES
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)


# --- Model Evaluation ---
y_train_pred = rf_model.predict(X_train_final)
y_test_pred = rf_model.predict(X_test_final)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")


# --- Streamlit UI ---
st.title("Credit Score Predictor | By Ashish V Bamba")

income = st.number_input("Annual Income", min_value=0.0)
debt = st.number_input("Total Debt", min_value=0.0)
expenditure_12 = st.number_input("12-Month Expenditure", min_value=0.0)
expenditure_6 = st.number_input("6-Month Expenditure", min_value=0.0)
gambling = st.number_input("Annual Gambling Spend", min_value=0.0)
savings = st.number_input("Savings Account Balance ($)", min_value=0.0)
education_12 = st.number_input("12-Month Education Spend", min_value=0.0)
education_6 = st.number_input("6-Month Education Spend", min_value=0.0)
has_credit_card = st.checkbox("Do you have a credit card?")


if st.button("Predict Default Status"):
    r_debt_income = np.log1p(debt / (income + 1))
    t_expenditure_12 = expenditure_12
    t_expenditure_6 = expenditure_6
    t_gambling_12 = gambling
    savings_amount = savings
    r_expenditure = t_expenditure_6 / (t_expenditure_12 + 1)
    r_education = education_6 / (education_12 + 1)
    cat_credit_card = 1 if has_credit_card else 0

    user_input_cont = [[
    r_debt_income, 
    t_gambling_12,
    savings,
    r_expenditure, 
    r_education
]]

    user_input_cat = [[cat_credit_card]]

    user_input_scaled_cont = scaler.transform(user_input_cont)
    user_input_final = np.hstack([user_input_scaled_cont, user_input_cat])

    print("Raw input (cont):", user_input_cont)
    print("Raw input (cat):", user_input_cat)
    print("Full input:", user_input_final)

    prediction = rf_model.predict(user_input_final)[0]
    prob = rf_model.predict_proba(user_input_final)[0][1]  # Probability of default


    insert_prediction(
        income, debt, 
        r_debt_income, 
        t_gambling_12,
        savings_amount, 
        r_expenditure, 
        education_12,
        education_6,
        r_education,
        cat_credit_card,
        prediction
    )

    zero_fields = 0

    fields_to_check = [
        income, debt, expenditure_12, expenditure_6,  
        gambling, savings_amount, has_credit_card,
        education_12,
        education_6,
    ]

    for val in fields_to_check:
        if val == 0:
            zero_fields += 1

    if zero_fields >= 5:
        st.warning("⚠️ Your inputs contain many zero values. This may result in an inaccurate prediction. Try entering realistic estimates for typical expenses")


    latest = fetch_latest_score_with_label()
    raw_score = latest['score'][0]
    score = int.from_bytes(raw_score, byteorder='little') if isinstance(raw_score, bytes) else int(raw_score)

    label = "Default" if score == 1 else "No Default"


    fig = go.Figure(go.Indicator(
    mode="number+gauge",
    value=prob * 100,  # convert to percent
    number={'suffix': "%", 'font': {'size': 36}},
    title={'text': "Probability of Default", 'font': {'size': 24}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': "red" if prediction == 1 else "green"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 25], 'color': "#00FF00"},
            {'range': [25, 50], 'color': "#FFD700"},
            {'range': [50, 75], 'color': "#FFA500"},
            {'range': [75, 100], 'color': "#FF4B4B"},
        ]
    },
    domain={'x': [0, 1], 'y': [0, 1]}
    ))

    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.error(f"⚠️ This user is likely to **default**.\nProbability: {prob:.2%}")
    else:
        st.success(f"✅ This user is likely **not** to default.\nProbability: {prob:.2%}")



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

def style_by_default_flag(row):
    return [
        'color: red' if row['score'] == 1 else 'color: green'
        if row['score'] == 0 else ''
    ] * len(row)

styled_df = prediction_df.style.apply(style_by_default_flag, axis=1)

st.subheader("Feature Importance")

fig = go.Figure(go.Bar(
    x=importance_df['Importance'],
    y=importance_df['Feature'],
    orientation='h',
    marker=dict(color='lightskyblue')
))

fig.update_layout(
    xaxis_title="Importance",
    yaxis_title="Feature",
    title="Random Forest Feature Importances",
    title_font_size=20,
    margin=dict(l=100, r=20, t=40, b=20),
    height=400
)

st.plotly_chart(fig, use_container_width=True)



def style_score_cell(val):
    if val == 1:
        return 'color: red'
    elif val == 0:
        return 'color: green'
    return ''

if not prediction_df.empty:
    styled_df = prediction_df.style.applymap(style_score_cell, subset=['score'])
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
            r_debt_income FLOAT,
            t_gambling_12 FLOAT,
            savings_amount FLOAT,
            r_expenditure FLOAT,
            r_education FLOAT,
            education_12 FLOAT,
            education_6 FLOAT,
            cat_credit_card INTEGER,
            score INTEGER
        )
    """)

    conn.commit()
    conn.close()

    st.success("✅ Table has been dropped and recreated.")
    st.dataframe(pd.DataFrame(columns=[
        "id", "timestamp", "income", "debt",
        "r_debt_income", 
        "t_gambling_12", "savings_amount",
        "r_expenditure", 
        "education_12",
        "education_6",
        "r_education",
        "cat_credit_card",
        "score"
    ]))