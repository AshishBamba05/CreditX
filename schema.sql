DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  income FLOAT,
  debt FLOAT,
  r_debt_income FLOAT,
  t_gambling_12 FLOAT,
  savings_amount INTEGER,
  r_expenditure FLOAT,
  r_education FLOAT,
  cat_credit_card INTEGER,
  score INTEGER
);
