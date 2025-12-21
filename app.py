import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL AND SCALER ----------------
try:
    model = joblib.load("loan_default_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Loan Default Prediction", layout="centered")
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk")

st.divider()

# ---------------- INPUT FIELDS ----------------
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Income", 0, 1_000_000, 50000)
loan_amount = st.number_input("Loan Amount", 0, 1_000_000, 15000)
credit_score = st.number_input("Credit Score", 300, 900, 700)
months_employed = st.number_input("Months Employed", 0, 500, 24)
num_credit_lines = st.number_input("Number of Credit Lines", 0, 20, 3)
interest_rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 5.5)
loan_term = st.number_input("Loan Term (months)", 6, 360, 36)
dti_ratio = st.number_input("DTI Ratio", 0.0, 100.0, 20.0)

education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate"])
employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Other"])
has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

st.divider()

# ---------------- ENCODING ----------------
education_map = {"High School": 0, "Graduate": 1, "Post Graduate": 2}
employment_map = {"Salaried": 0, "Self-Employed": 1, "Unemployed": 2}
marital_map = {"Single": 0, "Married": 1}
yes_no_map = {"No": 0, "Yes": 1}
purpose_map = {"Auto": 0, "Business": 1, "Other": 2}

input_data = np.array([[ 
    age,
    income,
    loan_amount,
    credit_score,
    months_employed,
    num_credit_lines,
    interest_rate,
    loan_term,
    dti_ratio,
    education_map[education],
    employment_map[employment],
    marital_map[marital_status],
    yes_no_map[has_mortgage],
    yes_no_map[has_dependents],
    purpose_map[loan_purpose],
    yes_no_map[has_cosigner]
]])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Loan Default"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = int(model.predict(input_scaled)[0])

        st.subheader("📊 Prediction Result")
        st.write("🔢 Model Output (Binary Prediction):", prediction)

        if prediction == 1:
            st.error("🔴 Loan is LIKELY to DEFAULT")
        else:
            st.success("🟢 Loan is NOT likely to default")

    except Exception as e:
        st.warning(f"⚠️ Prediction failed: {e}")
