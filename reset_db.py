import sqlite3

conn = sqlite3.connect("creditx_predictions.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS predictions")

cursor.execute("""
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        loan_amount FLOAT,
        income FLOAT,
        age INTEGER,
        months_employed INTEGER,
        interest_rate FLOAT,
        loan_term INTEGER,
        credit_score INTEGER,
        r_loan_income FLOAT,
        r_interest_burden FLOAT,
        r_credit_util FLOAT,
        has_coSigner INTEGER,
        has_mortgage INTEGER,
        score INTEGER
    )
""")

conn.commit()
conn.close()

print("✅ Recreated table with updated feature set.")
