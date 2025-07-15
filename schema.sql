DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  income FLOAT,
  debt FLOAT,
  r_debt_income FLOAT,
  t_expenditure_12 FLOAT,
  t_gambling_12 FLOAT,
  cat_savings_account INTEGER,
  r_expenditure FLOAT,
  education_12 FLOAT,
  education_6 FLOAT,
  r_education FLOAT,
  cat_credit_card INTEGER,
  score INTEGER
);
