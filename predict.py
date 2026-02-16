import pandas as pd
import joblib

model = joblib.load("churn_xgb_model.pkl")
model_columns = joblib.load("model_columns.pkl")

def preprocess(df):

    # binary columns
    binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1,'No':0})

    # numeric fix
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # one hot encoding
    df = pd.get_dummies(df)

    # match training columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    return df


def predict_risk(uploaded_file):

    raw_df = pd.read_excel(uploaded_file)

    processed = preprocess(raw_df.copy())

    probs = model.predict_proba(processed)[:,1]

    raw_df["Churn_Risk"] = probs

    # keep only required columns
    output_df = raw_df[["customerID", "Churn_Risk"]]

    # sort by risk
    output_df = output_df.sort_values(by="Churn_Risk", ascending=False).reset_index(drop=True)

    return output_df
