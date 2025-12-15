import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Admission Success Predictor",
    layout="centered"
)

# Loading our trained model
model = joblib.load("models/logistics_regression.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Admission Success Predictor")
st.write("Enter application details to predicts admission outcome.")
# ---- Input Fields ----
gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5)
research = st.selectbox("Research Experience", [0, 1])

# ---- Prepare Input Data ----
input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
input_scaled = scaler.transform(input_data)

st.divider()

# ---- Prediction ----
if st.button("Predict Admission"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"Likely to be admitted (Probability: {probability:.2%})")
    else:
        st.error(f"Unlikely to be admitted (Probability: {probability:.2%})")


        # ---- Explanation ----
    st.subheader("Why this result?")

    if cgpa >= 8.5:
        st.write(" Strong CGPA significantly improves admission chances.")
    elif cgpa < 7.0:
        st.write("Low CGPA reduces admission likelihood.")

    if gre >= 320:
        st.write("High GRE score strengthens the application.")

    if toefl >= 105:
        st.write("Strong TOEFL score indicates good language proficiency.")

    if research == 1:
        st.write("Research experience positively influences admission.")
