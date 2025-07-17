import sqlite3
import pandas as pd
import re

DB_PATH = "creditx_predictions.db"

def get_connection():
    return sqlite3.connect(DB_PATH)


def insert_prediction(income, age, months_employed,
                      loan_amount, interest_rate,
                      loan_term, credit_score,
                      education, 
                      loan_purpose,
                      has_coSigner, has_mortgage,
                      score):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            income,
            age,
            months_employed,
            loan_amount,
            interest_rate,
            loan_term,
            credit_score,
            education,
            loan_purpose,
            has_coSigner,
            has_mortgage,
            score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        income,
        age,
        months_employed,
        loan_amount,
        interest_rate,
        loan_term,
        education,
        loan_purpose,
        credit_score,
        has_coSigner,
        has_mortgage,
        score
    ))

    conn.commit()
    conn.close()


def run_sql_query_from_file(filename, query_label, **kwargs):
    with open(filename, "r") as file:
        sql_script = file.read()

    pattern = rf"-- \[{query_label}\]\s*(.*?)(?=\n-- \[|$)"
    match = re.search(pattern, sql_script, re.DOTALL)

    if not match:
        print(f"❌ Query label not found: {query_label}")
        return ""

    sql = match.group(1).strip()

    for key, val in kwargs.items():
        sql = sql.replace(f"{{{key}}}", str(val))

    return sql


def fetch_predictions():
    conn = get_connection()
    query = """
    SELECT 
        id,
        timestamp,
        income,  
        age,
        months_employed,
        loan_amount,
        interest_rate,
        loan_term,
        credit_score,
        education,
        loan_purpose,
        has_coSigner,
        has_mortgage,
        score,
        CASE
            WHEN score = 1 THEN '🔴 Default'
            WHEN score = 0 THEN '🟢 No Default'
            ELSE '⚪ Unknown'
        END AS score_category
    FROM predictions
    ORDER BY timestamp DESC
    """
    return pd.read_sql_query(query, conn)


def fetch_latest_score_with_label():
    sql = run_sql_query_from_file("ex_queries.sql", "latest_score_with_label")
    conn = get_connection()
    return pd.read_sql_query(sql, conn)