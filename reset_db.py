import sqlite3

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
        r_expenditure FLOAT,
        score FLOAT
    )
""")

conn.commit()
conn.close()

print("✅ New DB created with correct columns for model features and predictions.")
