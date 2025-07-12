import sqlite3
import pandas as pd
import re

DB_PATH = "predictions.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def insert_prediction(income, debt, savings, expenditure, score):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (income, debt, savings, expenditure, score)
        VALUES (?, ?, ?, ?, ?)
    """, (income, debt, savings, expenditure, score))

    conn.commit()
    conn.close()

def fetch_predictions():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def run_sql_query_from_file(filename, query_label, **kwargs):
    with open(filename, "r") as file:
        sql_script = file.read()

    # Improved pattern: match the block even if it's the last one in the file
    pattern = rf"-- \[{query_label}\]\s*(.*?)(?:\n-- \[|$)"
    match = re.search(pattern, sql_script, re.DOTALL)

    if not match:
        print(f"❌ Query label not found: {query_label}")
        return ""

    raw_query = match.group(1).strip()

    # Replace placeholders with user input
    for key, value in kwargs.items():
        raw_query = raw_query.replace(f"{{{key}}}", str(value))

    return raw_query