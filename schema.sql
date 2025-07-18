DROP TABLE IF EXISTS predictions;

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
  dti_ratio INT,
  numCreditLines INT,
  loan_purpose INT,
  has_dependents INT,
  marital_status INT,
  score INTEGER
);
