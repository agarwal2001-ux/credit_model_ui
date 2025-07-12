import streamlit as st
import pandas as pd
import numpy as np
from joblib import load,dump
import os

# Load your trained model (update the path if necessary)
encoder = load(os.path.join("models","encoder","label_encoder.joblib"))
model = load(os.path.join("models","pipeline","final_model.joblib"))


st.set_page_config(page_title="Credit Worthiness Predictor", layout="wide")
st.title("Credit Worthiness Prediction App")

st.markdown("---")

# Define input columns
col1, col2, col3 = st.columns(3)

# Percentage Inputs (0 to 1)
with col1:
    pct_tl_open_l6m = st.slider("% TL Open in Last 6 Months", 0.0, 1.0, 0.5)
    pct_tl_closed_l6m = st.slider("% TL Closed in Last 6 Months", 0.0, 1.0, 0.5)
    pct_tl_open_l12m = st.slider("% TL Open in Last 12 Months", 0.0, 1.0, 0.5)
    pct_tl_closed_l12m = st.slider("% TL Closed in Last 12 Months", 0.0, 1.0, 0.5)
    cc_utilization = st.slider("Credit Card Utilization", 0.0, 1.0, 0.5)

with col2:
    tot_tl_closed_l12m = st.number_input("Total TL Closed in Last 12 Months", 0)
    tot_missed_pmnt = st.number_input("Total Missed Payments", 0)
    cc_tl = st.number_input("Credit Card TL", 0)
    home_tl = st.number_input("Home Loan TL", 0)
    pl_tl = st.number_input("Personal Loan TL", 0)
    secured_tl = st.number_input("Secured TL", 0)
    unsecured_tl = st.number_input("Unsecured TL", 0)
    other_tl = st.number_input("Other TL", 0)

with col3:
    age_oldest_tl = st.number_input("Age of Oldest TL (months)", 0)
    age_newest_tl = st.number_input("Age of Newest TL (months)", 0)
    time_since_recent_payment = st.number_input("Time Since Recent Payment (days)", 0)
    max_delinquency_level = st.number_input("Max Delinquency Level", 0)
    num_deliq_6_12mts = st.number_input("Number of Delinquencies (6-12 months)", 0)
    max_deliq_6mts = st.number_input("Max Delinquency in 6 Months", 0)
    max_deliq_12mts = st.number_input("Max Delinquency in 12 Months", 0)
    num_times_60p_dpd = st.number_input("Times 60+ DPD", 0)

# Row 2 Inputs
with col1:
    num_std_12mts = st.number_input("Standard in Last 12 Months", 0)
    num_sub = st.number_input("Number of Substandard TLs", 0)
    num_sub_12mts = st.number_input("Substandard TLs in 12 Months", 0)
    num_dbt = st.number_input("Number of Doubtful TLs", 0)
    num_dbt_12mts = st.number_input("Doubtful TLs in 12 Months", 0)
    num_lss = st.number_input("Number of Loss TLs", 0)
    recent_level_of_deliq = st.number_input("Recent Level of Delinquency", 0)

with col2:
    cc_enq_l12m = st.number_input("Credit Card Enquiries (Last 12 Months)", 0)
    pl_enq_l12m = st.number_input("Personal Loan Enquiries (Last 12 Months)", 0)
    time_since_recent_enq = st.number_input("Time Since Recent Enquiry (days)", 0)
    enq_l3m = st.number_input("Enquiries in Last 3 Months", 0)
    netmonthlyincome = st.number_input("Net Monthly Income")
    time_with_curr_empr = st.number_input("Time with Current Employer (months)", 0)

with col3:
    cc_flag = st.selectbox("Has Credit Card?", [0, 1])
    pl_flag = st.selectbox("Has Personal Loan?", [0, 1])
    pct_pl_enq_l6m_of_ever = st.slider("% PL Enq in 6M of Ever", 0.0, 1.0, 0.5)
    pct_cc_enq_l6m_of_ever = st.slider("% CC Enq in 6M of Ever", 0.0, 1.0, 0.5)
    hl_flag = st.selectbox("Has Home Loan?", [0, 1])
    gl_flag = st.selectbox("Has Gold Loan?", [0, 1])

# Categorical Inputs
with col1:
    maritalstatus = st.selectbox("Marital Status", ["Married","Single"])
    education = st.selectbox("Education", ['12TH', 'GRADUATE', 'SSC', 'POST-GRADUATE', 'UNDER GRADUATE','OTHERS', 'PROFESSIONAL'])
    gender = st.selectbox("Gender", ["M", "F"])
    last_prod_enq2 = st.selectbox("Last Product Enquired", ['PL', 'ConsumerLoan', 'others', 'AL', 'CC', 'HL'])
    first_prod_enq2 = st.selectbox("First Product Enquired", ['PL', 'ConsumerLoan', 'others', 'AL', 'HL', 'CC'])

# Prediction Button
if st.button("Predict Credit Worthiness"):
    # Create input DataFrame for prediction (dummy here)
    input_data = pd.DataFrame({
        'pct_tl_open_l6m': [pct_tl_open_l6m],
        'pct_tl_closed_l6m': [pct_tl_closed_l6m],
        'tot_tl_closed_l12m': [tot_tl_closed_l12m],
        'pct_tl_open_l12m': [pct_tl_open_l12m],
        'pct_tl_closed_l12m': [pct_tl_closed_l12m],
        'tot_missed_pmnt': [tot_missed_pmnt],
        'cc_tl': [cc_tl],
        'home_tl': [home_tl],
        'pl_tl': [pl_tl],
        'secured_tl': [secured_tl],
        'unsecured_tl': [unsecured_tl],
        'other_tl': [other_tl],
        'age_oldest_tl': [age_oldest_tl],
        'age_newest_tl': [age_newest_tl],
        'time_since_recent_payment': [time_since_recent_payment],
        'max_delinquency_level': [max_delinquency_level],
        'num_deliq_6_12mts': [num_deliq_6_12mts],
        'max_deliq_6mts': [max_deliq_6mts],
        'max_deliq_12mts': [max_deliq_12mts],
        'num_times_60p_dpd': [num_times_60p_dpd],
        'num_std_12mts': [num_std_12mts],
        'num_sub': [num_sub],
        'num_sub_12mts': [num_sub_12mts],
        'num_dbt': [num_dbt],
        'num_dbt_12mts': [num_dbt_12mts],
        'num_lss': [num_lss],
        'recent_level_of_deliq': [recent_level_of_deliq],
        'cc_enq_l12m': [cc_enq_l12m],
        'pl_enq_l12m': [pl_enq_l12m],
        'time_since_recent_enq': [time_since_recent_enq],
        'enq_l3m': [enq_l3m],
        'netmonthlyincome': [netmonthlyincome],
        'time_with_curr_empr': [time_with_curr_empr],
        'cc_utilization': [cc_utilization],
        'cc_flag': [cc_flag],
        'pl_flag': [pl_flag],
        'pct_pl_enq_l6m_of_ever': [pct_pl_enq_l6m_of_ever],
        'pct_cc_enq_l6m_of_ever': [pct_cc_enq_l6m_of_ever],
        'hl_flag': [hl_flag],
        'gl_flag': [gl_flag],
        'maritalstatus': [maritalstatus],
        'education': [education],
        'gender': [gender],
        'last_prod_enq2': [last_prod_enq2],
        'first_prod_enq2': [first_prod_enq2]
    })

    st.dataframe(input_data)

    prediction = model.predict(input_data)
    tranformed = encoder.inverse_transform(prediction)
    st.success(f"Predicted Credit Worthiness: {tranformed}")

    st.info("Model prediction have appeared successfully")
