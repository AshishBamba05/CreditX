# 💳 CreditX

**CreditX** is a machine learning web app that predicts a user's credit score based on financial data including income, savings, debt, and annual expenditure.

## 🔗 Launch the App

👉 [Launch CreditX](https://creditx-nyywptbpkg9gkmtym5qeam.streamlit.app/) (Recommended: use **Chrome**)  
⚠️ *Safari may have display issues due to JavaScript compatibility.*


## 🚀 How It Works

- The model is trained on a [Kaggle dataset]((https://www.kaggle.com/datasets/conorsully1/credit-score)) and split using an 80/20 train-test ratio.
Once the input features are extracted, the model applies feature engineering to derive additional signals:

- `r_debt_income = debt / (income + 1)` — normalized debt-to-income ratio  
- `t_expenditure_12 = expenditure` — annual expenditure baseline  
- `debt_savings = debt / (savings + 1)` — debt-to-savings ratio with smoothing

- I used **Support Vector Machines (SVMs)** from **Scikit-Learn** to build the predictive model.
- The application is deployed using **Streamlit**.

## 📊 Model Performance

- ✅ **Mean Absolute Error (MAE):** 26.32  
- ✅ **Mean Squared Error (MSE):** 1181.19  
- ✅ **R² Score:** 0.697
---

