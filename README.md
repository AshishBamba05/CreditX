# 💳 FinRisk.AI

**FinRisk** is a machine learning web app that predicts whether a user will default on a loan based on financial data including income, debt, and annual expenditure.

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

  -  ### The Dataset
        - The model is trained on a [Kaggle dataset]([(https://www.kaggle.com/datasets/nikhil1e9/loan-default/data)), containing **250K+** financial profiles and split using an 81/19 train-test ratio.
    
        - The model collects the following input features to render predictions in the ML model:
            -  **Continuous Features**
                  - `'Age'`,
                  - `'Income'`,
                  - `'LoanAmount'`,
                  - `'CreditScore'`,
                  - `'MonthsEmployed'`,
                  - `'NumCreditLines'`,
                  - `'InterestRate'`,
                  - `'LoanTerm'`,
                  - `'DTIRatio'`
        
            - **Categorical Features**
                - `'Education',`
                - `'MaritalStatus',`
                - `'HasMortgage',`
                - `'HasDependents',`
                - `'LoanPurpose',`
                - `'HasCoSigner'`
     
      - Since strings alone can break the compilation of the ML model, I mapped categorical variables to integer representation:
        ```
        - df["HasCoSigner"] = df["HasCoSigner"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        - df["HasMortgage"] = df["HasMortgage"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        - df["Education"] = df["Education"].map({"Bachelor's": 0, "High School": 1, "Other": 2}).fillna(0).astype(int)
        - df["HasDependents"] = df["HasDependents"].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        - df["LoanPurpose"] = df["LoanPurpose"].map({'Business': 0, 'Home': 1, 'Other': 2}).fillna(0).astype(int)
        -  df["MaritalStatus"] = df["MaritalStatus"].map({'Married': 0, 'Divorced': 1, 'Other': 2}).fillna(0).astype(int)
        ```
  
  - ### StandardScaler()
      - To account for disproportionate impact, I used z-score normalization via the `StandardScaler` library to normalize all **continuous** variables

          ```
          scaler = StandardScaler()
          preprocessor = ColumnTransformer([
          ('cont', scaler, continuous_features),
          ('cat', 'passthrough', categorical_features)
          ])

          X_train_scaled = preprocessor.fit_transform(X_train)
          ```

  - ### Applying SMOTE + EN   
      - Since the dataset initially featured heavy class imbalance (Non-default > default), I resorted using SMOTEEN to avoid the risk of overfitting:
      - ```
        smote_enn = SMOTEENN(random_state=42)
        X_train_scaled, y_train = smote_enn.fit_resample(X_train_scaled, y_train)
        ```

The same procedure was applied for values the user inserted.

    ```
    user_input_cont = [[ r_debt_income, t_gambling_12, savings, r_expenditure, r_education]]
    user_input_cat = [[cat_credit_card]]
    user_input_scaled_cont = scaler.transform(user_input_cont)
    user_input_final = np.hstack([user_input_scaled_cont, user_input_cat])
    ```


  - ### XGBoost Model 
      - I used **XGBoost** from **Scikit-Learn** to build the predictive model, with fine-tuned hyperparemeters for optimal accuracy:
 
        - ```
          xgb_model = XGBClassifier(
          n_estimators=200,
          max_depth=3,
          scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
          use_label_encoder=False,
          eval_metric='logloss',
          random_state=42
          n_jobs=-1
          )

          xgb_model.fit(X_train_scaled, y_train)
          ```
     -  All things considered, here is how my model ranked each feature in terms of predictive value across the board:
         - <img width="704" height="404" alt="Screen Shot 2025-07-18 at 3 32 58 AM" src="https://github.com/user-attachments/assets/b08b0ce1-a5c4-4c56-bdcd-25abe7cf3365" />
    
- The application is deployed using **Streamlit**.


## 📊 Model Performance
- ✅ **AUC**: 0.751
- ✅ **Precision**: 0.283
- ✅ **Recall**: 0.506
- ✅ **F1 Score**: 0.363
---

