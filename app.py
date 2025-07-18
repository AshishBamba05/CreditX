import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import numpy as np
from collections import Counter
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier


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



@st.cache_data
def load_data():
    df = pd.read_csv("Loan_default.csv")
    return df.fillna(df.median(numeric_only=True)).round(0)

df = load_data()


@st.cache_data
def preprocess_data(df):
    #df["R_LOAN_INCOME"] = df["LoanAmount"] / (df["Income"] + 1)
    df["R_INTEREST_BURDEN"] = df["InterestRate"] * df["LoanTerm"]
    df["R_CREDIT_UTIL"] = df["LoanAmount"] / (df["CreditScore"] + 1)
    df["HasCoSigner"] = df["HasCoSigner"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    df["HasMortgage"] = df["HasMortgage"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    df["Education"] = df["Education"].map({"Bachelor's": 0, "High School": 1, "Other": 2}).fillna(0).astype(int)
    df["R_SCORE_PER_LINE"] = df["CreditScore"] / (df["NumCreditLines"] + 1)
    df["Income_Age"] = (df["Income"] / df["Age"])
    df["R_LINES_AGE"] = df["LoanTerm"] / (12 * df["Age"] + 1)
    return df

df = preprocess_data(df)

print(df["Default"].describe())

continuous_features = [
    'Income',
    'InterestRate',
    'R_CREDIT_UTIL',
    'MonthsEmployed',
    'R_SCORE_PER_LINE',
    'LoanAmount',
    'Income_Age',
    #'R_LINES_AGE'
]

categorical_features = ['Education']
feature_names = continuous_features + categorical_features

# 1. Start with original, unbalanced data
X = df[continuous_features + categorical_features]
y = df["Default"]

# 2. Split before balancing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.188, random_state=42
)

# 4. Preprocess as before
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', scaler, continuous_features),
        ('cat', 'passthrough', categorical_features)
    ]
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

smote_enn = SMOTEENN(random_state=42)
X_train_scaled, y_train = smote_enn.fit_resample(X_train_scaled, y_train)

# 5. Train model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=1,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# --- Feature Importance ---
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)


# --- Model Evaluation ---
y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")


# --- Streamlit UI ---
st.title("FinRisk.AI")
st.markdown("###### ⚡️ Flags high-risk financial default users.")
st.markdown("###### 💵 Uses financial behavior data.")
st.markdown("💥 Provides ML-driven insights.")

st.caption("By Ashish V Bamba | [GitHub](https://github.com/AshishBamba05/FinRisk.AI) | [LinkedIn](https://www.linkedin.com/in/ashishbamba/)")

st.subheader("Enter Applicant Financial Details")

age = st.number_input("Age", min_value=0)
months_employed = st.number_input("Months Employed", min_value=0)
income = st.number_input("Annual Income ($)", min_value=0.0)
loan_amount = st.number_input("Loan Amount ($)", min_value=0.0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0)
loan_term = st.number_input("Loan Term (in months)", min_value=1)
credit_score = st.number_input("Credit Score", min_value=0)
has_coSigner = st.checkbox("Do you have a co-signer?", help="Select if another person is legally responsible for this loan with you.")
has_mortgage = st.checkbox("Do you have a mortgage?", help="Select if you have a mortgage on this loan.")
credit_lines = st.number_input("Number of Credit Lines", min_value=0)
education = st.selectbox(
    "Highest Education Level",
    ["Bachelor's", "High School", "Other"]
)
loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Home", "Other"]
)

st.write()


if st.button("Predict Default Status"):
    r_loan_income = loan_amount / (income + 1)
    r_interest_burden = interest_rate * loan_term
    r_credit_util = loan_amount / (credit_score + 1)
    r_months_employed = months_employed / (age + 1)
    has_coSigner = 1 if has_coSigner else 0
    has_mortgage = 1 if has_mortgage else 0
    r_score_per_line = credit_score / (credit_lines + 1)

    loan_purpose_map = {
        "Business": 0,
        "Home": 1,
        "Other": 2
    }
    loan_purpose = loan_purpose_map.get(loan_purpose, 2)

    education_map = {
        "Bachelor's": 0,
        "High School": 1,
        "Other": 2
    }
    education = education_map.get(education, 2)

    user_input = [[  
    r_loan_income, 
    r_interest_burden,
    r_credit_util,
    r_months_employed,
    has_coSigner,
    has_mortgage,
    loan_purpose,
    education,
    r_loan_per_line,
    r_credit_lines_income
]]

    user_input_scaled = scaler.transform(user_input)

    print("Raw input:", user_input)

    prediction = rf_model.predict(user_input_scaled)[0]
    prob = rf_model.predict_proba(user_input_scaled)[0][1]  # Probability of default


    insert_prediction(
        income,
        loan_amount, interest_rate,  
        loan_term,
        age,
        months_employed,
        credit_score,
        has_coSigner,
        has_mortgage,
        loan_purpose,
        education,
        credit_lines,
        prediction
    )

    zero_fields = 0

    fields_to_check = [
        income, age, months_employed, loan_amount, interest_rate,  
        has_coSigner, has_mortgage, loan_term, credit_score, education,
        loan_purpose, credit_lines
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
                                  debt=debt
                                 # expenditure=expenditure_12
                                  )

    if sql:
        bracket_df = pd.read_sql_query(sql, conn)
        st.subheader("Your Bracket Classification")
        st.write("Income Bracket:", bracket_df["income_bracket"][0])
       # st.write("Debt Bracket:", bracket_df["debt_bracket"][0])
        #st.write("Expenditure Bracket:", bracket_df["expenditure_bracket"][0])
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
            age INT,
            months_employed INT,
            loan_amount FLOAT,
            interest_rate FLOAT,
            loan_term FLOAT,
            credit_score INT,
            has_coSigner INT,
            has_mortgage INT,
            education INT,
            loan_purpose INT,
            numCreditLines INT,
            score INTEGER
        )
    """)

    conn.commit()
    conn.close()

    st.success("✅ Table has been dropped and recreated.")
    st.dataframe(pd.DataFrame(columns=[
        "id", "timestamp", "income", "loan_amount",
        "age", "months_employed", 
        "interest_rate", "loan_term", "credit_score", "has_coSigner",
        "education", "loan_purpose", "numCreditLines",
         "has_mortgage", "score"
    ]))