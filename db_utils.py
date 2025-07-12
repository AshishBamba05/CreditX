import sqlite3
import pandas as pd

DB_PATH = "predictions.db"

def insert_prediction(income, debt, savings, expenditure, score):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (income, debt, savings, expenditure, score)
        VALUES (?, ?, ?, ?, ?)
    """, (income, debt, savings, expenditure, score))

    conn.commit()
    conn.close()

def fetch_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df
