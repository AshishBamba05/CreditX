# 💳 CreditX

**CreditX** is a machine learning web app that predicts whether a user will financially default based on financial data including income, savings, debt, and annual expenditure.

## 🔗 Launch the App

👉 [Launch CreditX](https://creditx-nyywptbpkg9gkmtym5qeam.streamlit.app/) (Recommended: use **Chrome**)  
⚠️ *Safari may have display issues due to JavaScript compatibility.*


## 🚀 How It Works

- The model is trained on a [Kaggle dataset](https://www.kaggle.com/datasets/conorsully1/credit-score) and split using an 75/25 train-test ratio.
Once the input features are extracted, the model applies feature engineering to derive additional signals:

- `r_debt_income = debt / (income + 1)` — normalized debt-to-income ratio  
- `r_expenditure = t_expenditure_12 / (t_expenditure_6 + 1)` — normalized full- to mid - year expenditure 
- `savings = savings_amount` - total annual savings
- `r_education = t_education_12 / (t_education_6 + 1)` - normalized full to mid year education spending
- `cat_credit_card = 1 if has_credit_card else 0` - user's credit card status
- `t_gambling_12 = gambling` - annual gambling expenditure

- I used **Random Forests (RFs)** from **Scikit-Learn** to build the predictive model.
- The application is deployed using **Streamlit**.

## 📊 Model Performance

- ✅ **Accuracy**: 0.855
- ✅ **Precision**: 0.791
- ✅ **Recall**: 0.909
- ✅ **F1 Score**: 0.846
---

