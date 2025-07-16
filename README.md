# 💳 CreditX

**CreditX** is a machine learning web app that predicts whether a user will financially default based on financial data including income, savings, debt, and annual expenditure.

## 🔗 Launch the App

👉 [Launch CreditX](https://creditx-nyywptbpkg9gkmtym5qeam.streamlit.app/) (Recommended: use **Chrome**)  
⚠️ *Safari may have display issues due to JavaScript compatibility.*

## 🖥 Installation
- `git clone https://github.com/yourusername/creditx.git`
- `cd creditx`
- `pip install -r requirements.txt`

NOTE: Make sure you have Python 3.9+ and streamlit installed.

To run the app locally,
- `streamlit run app.py`


## 💸 FinTech Background & What Model Does
This project tackles a core challenge in financial technology: predicting credit default risk using customer behavior and spending patterns rather than traditional credit bureau data. In real-world fintech applications, accurate default prediction helps institutions:

- Reduce financial losses

- Make smarter lending decisions

- Serve underbanked populations without formal credit history

The model is trained on a structured dataset containing customer-level financial indicators—such as income, debt, savings, and categorical flags like credit card ownership. It leverages classic machine learning (Random Forest Classifier via Scikit-learn) to predict the likelihood of default with high precision and recall.

The system prioritizes recall to capture as many potential defaulters as possible while maintaining solid overall performance (F1 score and accuracy). It's an interpretable, production-ready approach, well-suited for real-world financial screening workflows.


## 🚀 How It Works

- The model is trained on a [Kaggle dataset](https://www.kaggle.com/datasets/conorsully1/credit-score) and split using an 80/20 train-test ratio.
Once the input features are extracted, the model applies feature engineering to derive additional signals:

  - `r_debt_income = debt / (income + 1)` — debt-to-income ratio  
  - `r_expenditure = t_expenditure_12 / (t_expenditure_6 + 1)` — full- to mid- year expenditure ratio
  - `savings = savings_amount` - total annual savings
  - `r_education = t_education_12 / (t_education_6 + 1)` - full- to mid- year education spending
  - `cat_credit_card = 1 if has_credit_card else 0` - user's credit card status
  - `t_gambling_12 = gambling` - annual gambling expenditure
    
- Since the dataset initially featured heavy class imbalance (Non-default > default), I added 373 randomly generated rows (~25%) where `default == 1`. Using `Panndas` library to concatenate the initialband altered dataframes, here is the dataframe description of the new dataframe:
  - `print(df_balanced["DEFAULT"].describe())`
    - ```count    1373.000000```
      
      ```mean        0.478514
      
      std         0.499720
      
      min         0.000000
      
      25%         0.000000
      
      50%         0.000000
      
      75%         1.000000
      
      max         1.000000 ```

To account for disproportionate impact, I used z-score normalization via `StandardScaler` to normalize all **continuous** variables

 All things considered, here is how my model ranked each features in terms of predictive value:

  <img width="704" height="401" alt="Screen Shot 2025-07-16 at 2 29 34 AM" src="https://github.com/user-attachments/assets/5d9ab709-5823-454a-b5cf-297d90eff482" />

- I used **Random Forests (RFs)** from **Scikit-Learn** to build the predictive model, with the following hyperparemeters
  - `n_estimators = 95`
  - `max_depth = 12`
- The application is deployed using **Streamlit**.


## 📊 Model Performance

- ✅ **Accuracy**: 0.855
- ✅ **Precision**: 0.791
- ✅ **Recall**: 0.909
- ✅ **F1 Score**: 0.846
---

