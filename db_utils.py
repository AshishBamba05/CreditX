import sqlite3
import pandas as pd
import re

DB_PATH = "creditx_predictions.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def insert_prediction(income, debt, savings, expenditure,
                      r_debt_income, t_expenditure_12,
                      t_health_12, t_gambling_12,
                      score):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (
        income, debt, savings, expenditure,
        r_debt_income, t_expenditure_12, 
        t_health_12, t_gambling_12, score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        income, debt, savings, expenditure,
        r_debt_income, t_expenditure_12,
        t_health_12, t_gambling_12, score
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
        debt,
        savings,
        expenditure,
        r_debt_income,
        t_health_12,
        t_gambling_12,
        t_expenditure_12,
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


def fetch_latest_score_with_label():
    sql = run_sql_query_from_file("ex_queries.sql", "latest_score_with_label")
    conn = get_connection()
    return pd.read_sql_query(sql, conn)
