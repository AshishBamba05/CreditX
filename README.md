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
Once raw input features (like `debt, income, mid-/full- year expenditure funds, mid-/full- year education funds, etc.`) are extracted, the model applies **feature engineering** to derive additional signals:

  - `r_debt_income = debt / (income + 1)` — debt-to-income ratio  
  - `r_expenditure = t_expenditure_12 / (t_expenditure_6 + 1)` — full- to mid- year expenditure ratio
  - `savings = savings_amount` - total annual savings
  - `r_education = t_education_12 / (t_education_6 + 1)` - full- to mid- year education spending
  - `cat_credit_card = 1 if has_credit_card else 0` - user's credit card status
  - `t_gambling_12 = gambling` - annual gambling expenditure
    
- Since the dataset initially featured heavy class imbalance (Non-default > default), I inserted 373 randomly synthesized rows (~25%) where `default == 1` into my dataframe

- Using `Pandas` library to concatenate the initial and altered dataframes, here is the description of the new dataframe:
  - `print(df_balanced["DEFAULT"].describe())`
    - ```
      count    1373.000000
      
      mean        0.478514
      
      std         0.499720
      
      min         0.000000
      
      25%         0.000000
      
      50%         0.000000
      
      75%         1.000000
      
      max         1.000000
      ```

To account for disproportionate impact, I used z-score normalization via the `StandardScaler` library to normalize all **continuous** variables

  ```
# Separate continuous vs. binary categorical feature
  X_continuous = X.drop(columns=["CAT_CREDIT_CARD"])
  X_cat = X[["CAT_CREDIT_CARD"]].values

# Train/test split
  X_train_cont, X_test_cont, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_continuous, X_cat, y, test_size=0.2, random_state=42
  )

# Scale only continuous features
  scaler = StandardScaler()
  X_train_scaled_cont = scaler.fit_transform(X_train_cont)
  X_test_scaled_cont = scaler.transform(X_test_cont)

# Combine scaled continuous + unscaled categorical
  X_train_final = np.hstack([X_train_scaled_cont, X_train_cat])
  X_test_final = np.hstack([X_test_scaled_cont, X_test_cat])
  
  ```

The same procedure was applied for values the user inserted.
    ```
    user_input_cont = [[ r_debt_income, t_gambling_12, savings, r_expenditure, r_education]]
    user_input_cat = [[cat_credit_card]]
    user_input_scaled_cont = scaler.transform(user_input_cont)
    user_input_final = np.hstack([user_input_scaled_cont, user_input_cat])
    ```

 All things considered, here is how my model ranked each feature in terms of predictive value across the board: 

  <img width="704" height="401" alt="Screen Shot 2025-07-16 at 2 29 34 AM" src="https://github.com/user-attachments/assets/5d9ab709-5823-454a-b5cf-297d90eff482" />

- I used **Random Forests (RFs)** from **Scikit-Learn** to build the predictive model, with the following hyperparemeters
  - `n_estimators = 95`
  - `max_depth = 12`
  - `class_weight = balanced`
  - `random_state = 42`
 
  - ```
    rf_model = RandomForestClassifier(n_estimators=95, max_depth=12, class_weight='balanced', random_state=42)
    rf_model.fit(X_train_final, y_train)
    ```
    
- The application is deployed using **Streamlit**.


## 📊 Model Performance

- ✅ **Accuracy**: 0.855
- ✅ **Precision**: 0.791
- ✅ **Recall**: 0.909
- ✅ **F1 Score**: 0.846
---

