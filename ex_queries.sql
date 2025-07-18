-- [classify_user]
SELECT
  CASE
    WHEN {income} < 30000 THEN 'Low Income'
    WHEN {income} BETWEEN 30000 AND 70000 THEN 'Mid Income'
    ELSE 'High Income'
  END AS income_bracket,

    CASE
      WHEN {dti_ratio} < 0.2 THEN 'Low Debt'
      WHEN {dti_ratio} BETWEEN 0.2 AND 0.35 THEN 'Mid Debt'
      ELSE 'High Debt'
    END AS debt_bracket,

    CASE
      WHEN {credit_score} < 580 THEN 'Poor Credit'
      WHEN {credit_score} BETWEEN 580 AND 669 THEN 'Fair Credit'
      WHEN {credit_score} BETWEEN 670 AND 739 THEN 'Good Credit'
      WHEN {credit_score} BETWEEN 740 AND 799 THEN 'Very Good Credit'
      ELSE 'Excellent Credit'
  END AS credit_bracket



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
