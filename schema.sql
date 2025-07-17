DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  loan_amount FLOAT,
  income FLOAT,
  interest_rate FLOAT,
  loan_term INTEGER,
  credit_score INTEGER,
  dti_ratio FLOAT,

  r_loan_income FLOAT,
  r_interest_burden FLOAT,
  r_credit_util FLOAT,
  flag_high_dti INTEGER,
  credit_bin TEXT,

  score INTEGER
);
