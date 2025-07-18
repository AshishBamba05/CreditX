# 💳 FinRisk.AI

**FinRisk** is a machine learning web app that predicts whether a user will financially default based on financial data including income, savings, debt, and annual expenditure.

## 🔗 Launch the App

👉 [Launch FinRisk.AI](https://creditx-nyywptbpkg9gkmtym5qeam.streamlit.app/) (Recommended: use **Chrome**)  
⚠️ *Safari may have display issues due to JavaScript compatibility.*

## 🖥 Installation
- `git clone https://github.com/yourusername/creditx.git`
- `cd creditx`
- `pip install -r requirements.txt`

NOTE: Make sure you have Python 3.9+ and streamlit installed.

To run the app locally,
- `streamlit run app.py`


## 💸 Overview
A 10-second pitch: Where financial intelligence meets machine learning, meet FinRisk, a FinIntel tool that helps institutions flag high default-risk users using customer behavior and spending patterns rather than traditional credit bureau data. In real-world fintech applications, accurate default prediction helps institutions:

- 📈 Reduce financial losses

- 💥 Make smarter lending decisions

- 🏦 Serve underbanked populations without formal credit history
  

## 🚀 How It Works

- The model is trained on a [Kaggle dataset](https://www.kaggle.com/datasets/conorsully1/credit-score) and split using an 81/19 train-test ratio.
- The model asks users to answer the following questions to extract features.
- Here are instances of raw features the model extracts and directly uses in the ML model:
  -  **Raw Features**
      - `InterestRate`
      - `LoanAmount` 
      - `Income`
      - `MonthsEmployed`
      - `Education`
      - `HasCoSigner`
      
  - **Engineered Features**
    - `R_CREDIT_UTIL = LoanAmount / (CreditScore + 1)` — loan amount to credit score ratio
    - `R_SCORE_PER_LINE= CreditScore / (NumCreditLines + 1)` — credit score to credit lines ratio
    - `R_Income_Age = Income / (Age + 1) ` - income-to-age ratio
  
    
- Since the dataset initially featured heavy class imbalance (Non-default > default), I resorted using SMOTEEN to avoid the risk of underfitting:
  - ```
      smote_enn = SMOTEENN(random_state=42)
      X_train_scaled, y_train = smote_enn.fit_resample(X_train_scaled, y_train)
    ```
  To ensure accurate metrics, this was only applied to **Training** partition after split.

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


<img width="704" height="404" alt="Screen Shot 2025-07-18 at 3 32 58 AM" src="https://github.com/user-attachments/assets/b08b0ce1-a5c4-4c56-bdcd-25abe7cf3365" />



- I used **XGBoost** from **Scikit-Learn** to build the predictive model, with fine-tuned hyperparemeters for optimal accuracy:
 
  - ```
    xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=1,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
    )

    xgb_model.fit(X_train_scaled, y_train)
    ```
    
- The application is deployed using **Streamlit**.


## 📊 Model Performance
- ✅ **AUC**: 0.751
- ✅ **Precision**: 0.283
- ✅ **Recall**: 0.506
- ✅ **F1 Score**: 0.363
---

