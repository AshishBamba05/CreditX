# 💳 CreditX

**CreditX** is a machine learning web app that predicts a user's credit score based on financial data including income, savings, debt, and annual expenditure.

## 🔗 Launch the App

👉 [Launch CreditX](https://creditx-nyywptbpkg9gkmtym5qeam.streamlit.app/) (Recommended: use **Chrome**)  
⚠️ *Safari may have display issues due to JavaScript compatibility.*


## 🚀 How It Works

- The model is trained on a [Kaggle dataset](https://www.kaggle.com/datasets/conorsully1/credit-score) and split using an 75/25 train-test ratio.
Once the input features are extracted, the model applies feature engineering to derive additional signals:

- `r_debt_income = debt / (income + 1)` — normalized debt-to-income ratio  
- `t_expenditure_12 = expenditure` — annual expenditure baseline  
- `t_gambling_12 = gambling` - annual gambling expenditure
- `t_health_12 = gambling` - annual health expenditure

- I used **Support Vector Machines (SVMs)** from **Scikit-Learn** to build the predictive model.
- The application is deployed using **Streamlit**.

## 📊 Model Performance

- ✅ **Mean Absolute Error (MAE):** 20.4  
- ✅ **Mean Squared Error (MSE):** 760.34  
- ✅ **R² Score:** 0.817
---

