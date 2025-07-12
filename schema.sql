DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  income FLOAT,
  debt FLOAT,
  savings FLOAT,
  expenditure FLOAT,
  score FLOAT
);
