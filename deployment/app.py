import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# Download model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="wellness-tourism-model",
        filename="best_model.joblib"
    )
    return joblib.load(model_path)

def main():
    st.set_page_config(
        page_title="Wellness Tourism Package Predictor",
        page_icon="✈️",
        layout="wide"
    )

    st.title("🌿 Wellness Tourism Package Predictor")
    st.markdown("---")

    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        st.header("Customer Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])

    with col2:
        st.header("Travel Preferences")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        monthly_income = st.number_input("Monthly Income", min_value=0, value=50000)
        number_of_trips = st.number_input("Number of Trips per Year", min_value=0, value=2)
        passport = st.selectbox("Has Passport", [0, 1])

    col3, col4 = st.columns(2)

    with col3:
        st.header("Additional Details")
        own_car = st.selectbox("Owns Car", [0, 1])
        number_of_person_visiting = st.number_input("Number of People", min_value=1, value=2)
        preferred_property_star = st.selectbox("Preferred Hotel Rating", [3, 4, 5])

    with col4:
        st.header("Interaction Data")
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    # Prediction button
    if st.button("Predict Purchase Probability", type="primary"):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [1 if gender == 'Male' else 0],
                'CityTier': [city_tier],
                'Occupation': [{"Salaried": 0, "Freelancer": 1, "Small Business": 2, "Large Business": 3}[occupation]],
                'MaritalStatus': [{"Single": 0, "Married": 1, "Divorced": 2}[marital_status]],
                'MonthlyIncome': [monthly_income],
                'NumberOfTrips': [number_of_trips],
                'Passport': [passport],
                'OwnCar': [own_car],
                'NumberOfPersonVisiting': [number_of_person_visiting],
                'PreferredPropertyStar': [preferred_property_star],
                'TypeofContact': [1 if type_of_contact == "Company Invited" else 0],
                'PitchSatisfactionScore': [pitch_satisfaction_score],
                'ProductPitched': [{"Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3, "King": 4}[product_pitched]]
            })

            # Make prediction
            prediction_proba = model.predict_proba(input_data)[0]
            prediction = model.predict(input_data)[0]

            # Display results
            st.markdown("---")
            st.header("Prediction Results")

            col_result1, col_result2 = st.columns(2)

            with col_result1:
                if prediction == 1:
                    st.success(f"✅ High likelihood to purchase! ({prediction_proba[1]:.2%})")
                else:
                    st.warning(f"❌ Low likelihood to purchase ({prediction_proba[1]:.2%})")

            with col_result2:
                st.metric("Purchase Probability", f"{prediction_proba[1]:.2%}")

            # Probability visualization
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Will Not Purchase', 'Will Purchase'],
                'Probability': prediction_proba
            })
            st.bar_chart(prob_df.set_index('Outcome'))

        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
