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
<img width="139" height="64" alt="Screen Shot 2025-07-14 at 11 59 56 PM" src="https://github.com/user-attachments/assets/d3a312b3-b509-4283-a18d-611c18279cca" />

- ✅ **Accuracy**: 0.756
- ✅ **Precision**: 0.6
- ✅ **Recall**: 0.5
- ✅ **F1 Score**: 0.561
---

