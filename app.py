import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained XGBoost model
model = joblib.load('final_xgboost_model.joblib')

# Define the Streamlit app
def main():
    # Display the header image
    st.image('hospital-appointment.jpg', use_column_width=True)

    # Set the title with a custom color
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Medical Appointment Predictor App</h1>", unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <p style='text-align: center; font-size: 18px; color: #555;'>This app predicts whether a patient will show up for their medical appointment.</p>
    <p style='text-align: center; font-size: 18px; color: #555;'>Please input the required details below.</p>
    """, unsafe_allow_html=True)

    # Input fields for the user
    gender = st.selectbox('Gender', options=['Female', 'Male'])
    age = st.slider('Age', 0, 100, 25)
    scholarship = st.selectbox('Scholarship', options=['No', 'Yes'])
    hipertension = st.selectbox('Hypertension', options=['No', 'Yes'])
    diabetes = st.selectbox('Diabetes', options=['No', 'Yes'])
    alcoholism = st.selectbox('Alcoholism', options=['No', 'Yes'])
    handcap = st.selectbox('Handicap', options=['No', 'Yes'])
    sms_received = st.selectbox('SMS Received', options=['No', 'Yes'])
    date_diff = st.slider('Days between Scheduling and Appointment', 0, 100, 10)

    # Preprocess the inputs
    gender = 0 if gender == 'Female' else 1
    scholarship = 1 if scholarship == 'Yes' else 0
    hipertension = 1 if hipertension == 'Yes' else 0
    diabetes = 1 if diabetes == 'Yes' else 0
    alcoholism = 1 if alcoholism == 'Yes' else 0
    handcap = 1 if handcap == 'Yes' else 0
    sms_received = 1 if sms_received == 'Yes' else 0

    # Prepare the input data for prediction
    input_data = np.array([[gender, age, scholarship, hipertension, diabetes, alcoholism, handcap, sms_received, date_diff]])

    # Predict the outcome
    if st.button('Predict'):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0]

        # Show the probability of showing up
        show_up_prob = prediction_proba[1] * 100
        not_show_up_prob = prediction_proba[0] * 100

        st.markdown(f"**Probability of showing up:** <span style='color: #4CAF50;'>{show_up_prob:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"**Probability of not showing up:** <span style='color: #FF5722;'>{not_show_up_prob:.2f}%</span>", unsafe_allow_html=True)

        # Conclusion based on prediction
        if prediction == 1:
            st.success("This suggests that the patient will **likely show up** for the appointment.")
        else:
            st.warning("This suggests that the patient will **likely not show up** for the appointment.")

if __name__ == "__main__":
    main()
