import streamlit as st
import pandas as pd
from predict import predict_risk

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Risk Ranking")

st.write("Upload an Excel file to rank customers by churn risk.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:

    with st.spinner("Analyzing customers..."):

        result = predict_risk(uploaded_file)

    st.success("Prediction complete!")

    st.subheader("Top High Risk Customers")
    st.dataframe(result.head(50), use_container_width=True)

    # download button
    csv = result.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Ranked Customers",
        data=csv,
        file_name="ranked_customers.csv",
        mime="text/csv"
    )
