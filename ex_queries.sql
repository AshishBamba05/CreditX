-- [classify_user]
SELECT
  CASE
    WHEN {income} < 30000 THEN 'Low Income'
    WHEN {income} BETWEEN 30000 AND 70000 THEN 'Mid Income'
    ELSE 'High Income'
  END AS income_bracket,

  CASE
    WHEN {debt} < 5000 THEN 'Low Debt'
    WHEN {debt} BETWEEN 5000 AND 10000 THEN 'Mid Debt'
    ELSE 'High Debt'
  END AS debt_bracket,

  CASE
    WHEN {savings} < 5000 THEN 'Low Savings'
    WHEN {savings} BETWEEN 5000 AND 10000 THEN 'Mid Savings'
    ELSE 'High Savings'
  END AS savings_bracket,

  CASE
    WHEN {expenditure} < 10000 THEN 'Low Expenditure'
    WHEN {expenditure} BETWEEN 10000 AND 20000 THEN 'Mid Expenditure'
    ELSE 'High Expenditure'
  END AS expenditure_bracket;
