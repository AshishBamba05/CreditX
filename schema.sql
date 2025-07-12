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



-- Retrieve top 5 recent predictions
SELECT * FROM predictions
ORDER BY timestamp DESC
LIMIT 5;

-- Insert a new prediction
INSERT INTO predictions (income, debt, savings, expenditure, score)
VALUES (50000, 10000, 2000, 15000, 720);

-- Check average credit score by income bracket
SELECT
  CASE
    WHEN income < 30000 THEN 'Low Income'
    WHEN income BETWEEN 30000 AND 70000 THEN 'Mid Income'
    ELSE 'High Income'
  END AS income_bracket,
  AVG(score) AS avg_score
FROM predictions
GROUP BY income_bracket;
