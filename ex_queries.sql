-- [classify_user]
SELECT
  CASE
    WHEN {income} < 30000 THEN 'Low Income'
    WHEN {income} BETWEEN 30000 AND 70000 THEN 'Mid Income'
    ELSE 'High Income'
  END AS income_bracket,

  SELECT
    CASE
      WHEN {debt} < 5000 THEN 'Low Debt'
      WHEN {debt} BETWEEN 5000 AND 10000 THEN 'Mid Debt'
      ELSE 'High Debt'
    END AS debt_bracket


-- [latest_score_with_label]
SELECT 
  score,
  CASE
    WHEN score = 1 THEN '🔴 Default'
    WHEN score = 0 THEN '🟢 No Default'
    ELSE '⚪ Unknown'
  END AS score_category,
  timestamp
FROM predictions
ORDER BY timestamp DESC
LIMIT 1
