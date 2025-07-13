import sqlite3
import pandas as pd
import re

DB_PATH = "predictions.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def insert_prediction(income, debt, savings, expenditure,
                      r_debt_income, t_expenditure_12,
                      r_debt_savings, score):
    conn = sqlite3.connect("creditx_predictions.db")
    cursor = conn.cursor()

    # Auto-create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
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

    cursor.execute("""
        INSERT INTO predictions (
            income, debt, savings, expenditure,
            r_debt_income, t_expenditure_12, r_debt_savings, score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        income, debt, savings, expenditure,
        r_debt_income, t_expenditure_12, r_debt_savings, score
    ))

    conn.commit()
    conn.close()


def run_sql_query_from_file(filename, query_label, **kwargs):
    with open(filename, "r") as file:
        sql_script = file.read()

    # Match -- [label] until next label or end of file
    pattern = rf"-- \[{query_label}\]\s*(.*?)(?=\n-- \[|$)"
    match = re.search(pattern, sql_script, re.DOTALL)

    if not match:
        print(f"❌ Query label not found: {query_label}")
        return ""

    sql = match.group(1).strip()

    # Replace any {placeholders} with actual values
    for key, val in kwargs.items():
        sql = sql.replace(f"{{{key}}}", str(val))

    return sql

def fetch_predictions():
    conn = sqlite3.connect("creditx_predictions.db")
    query = """
    SELECT 
        id,
        timestamp,
        income,
        debt,
        savings,
        expenditure,
        r_debt_income,
        t_expenditure_12,
        r_debt_savings,
        score,
        CASE
            WHEN score >= 800 THEN '🟢 Excellent'
            WHEN score >= 740 THEN '🟢 Very Good'
            WHEN score >= 670 THEN '🟡 Good'
            WHEN score >= 580 THEN '🟠 Fair'
            ELSE '🔴 Poor'
        END AS score_category
    FROM predictions
    ORDER BY timestamp DESC
    """
    return pd.read_sql_query(query, conn)
