DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  income FLOAT,
  debt FLOAT,
  r_debt_income FLOAT,
  t_expenditure_12 FLOAT,
  t_health_12 FLOAT, 
  t_gambling_12 FLOAT,
  cat_savings_account INTEGER,
  r_housing_debt FLOAT,
  r_expenditure FLOAT,
  r_education FLOAT,
  score INTEGER
);
