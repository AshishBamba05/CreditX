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
        r_debt_income FLOAT,
        t_gambling_12 FLOAT,
        savings_amount INTEGER,
        cat_credit_card INTEGER,
        r_expenditure FLOAT,
        r_education FLOAT,
        score INTEGER
    )
""")

conn.commit()
conn.close()

print("✅ Recreated table with all required columns.")
