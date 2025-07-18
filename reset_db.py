import sqlite3

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
        loan_term INT,
        credit_score INT,
        has_coSigner INT,
        has_mortgage INT,
        education INT,
        loan_purpose INT,
        numCreditLines INT,
        has_dependents INT,
        marital_status INT,
        dti_ratio FLOAT,
        score INTEGER
    )
""")

conn.commit()
conn.close()

print("✅ Recreated table with updated feature set.")
