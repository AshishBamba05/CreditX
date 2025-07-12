DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  income FLOAT,
  debt FLOAT,
  savings FLOAT,
  expenditure FLOAT,
  r_debt_income FLOAT,
  t_expenditure_12 FLOAT,
  r_debt_savings FLOAT,
  score FLOAT
);
