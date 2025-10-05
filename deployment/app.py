import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# Download model from Hugging Face
@st.cache_resource
def load_model():
    """Load model from Hugging Face with fallback options"""
    model_repo = os.getenv("HF_MODEL_REPO", "wellness-tourism-model")
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        # Try to download from Hugging Face
        st.info(f"Downloading model from {model_repo}...")
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename="best_model.joblib",
            token=hf_token
        )
        model = joblib.load(model_path)
        st.success(f"✓ Model loaded from Hugging Face!")
        return model
    except Exception as e:
        st.error(f"Could not load model from Hugging Face: {e}")
        
        # Try local file as fallback
        try:
            model = joblib.load("best_model.joblib")
            st.warning(" Loaded model from local file")
            return model
        except Exception as e2:
            st.error(f"Could not load local model: {e2}")
            st.error("Please ensure the model is uploaded to Hugging Face Model Hub")
            return None

def main():
    st.set_page_config(
        page_title="Wellness Tourism Package Predictor",
        page_icon="✈️",
        layout="wide"
    )
    
    st.title(" Wellness Tourism Package Predictor")
    st.markdown("### Predict customer likelihood to purchase wellness tourism packages")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Create input form
    st.sidebar.header(" Instructions")
    st.sidebar.markdown("""
    1. Fill in customer details
    2. Enter travel preferences
    3. Add interaction data
    4. Click 'Predict' to see results
    """)
    
    # Main input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.header(" Customer Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Customer's age")
        gender = st.selectbox("Gender", ["Male", "Female"])
        city_tier = st.selectbox("City Tier", [1, 2, 3], help="City development level (1=highest)")
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        
    with col2:
        st.header(" Financial & Travel Details")
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
        number_of_trips = st.number_input("Annual Trips", min_value=0, max_value=50, value=2)
        passport = st.selectbox("Has Passport", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = st.selectbox("Owns Car", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.header(" Travel Preferences")
        number_of_person_visiting = st.number_input("Number of People", min_value=1, max_value=10, value=2)
        number_of_children = st.number_input("Children (<5 years)", min_value=0, max_value=5, value=0)
        preferred_property_star = st.selectbox("Preferred Hotel Rating", [3, 4, 5])
        
    with col4:
        st.header(" Interaction Data")
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
        duration_of_pitch = st.number_input("Pitch Duration (minutes)", min_value=0, max_value=120, value=15)
    
    # Additional details
    with st.expander(" Additional Information"):
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    
    # Prediction button
    st.markdown("---")
    if st.button(" Predict Purchase Probability", type="primary", use_container_width=True):
        try:
            # Encode categorical variables (must match training encoding)
            gender_encoded = 1 if gender == "Male" else 0
            
            occupation_map = {
                "Salaried": 0, 
                "Small Business": 1, 
                "Large Business": 2, 
                "Freelancer": 3
            }
            occupation_encoded = occupation_map.get(occupation, 0)
            
            marital_status_map = {
                "Single": 0, 
                "Married": 1, 
                "Divorced": 2, 
                "Unmarried": 3
            }
            marital_status_encoded = marital_status_map.get(marital_status, 0)
            
            type_of_contact_encoded = 0 if type_of_contact == "Company Invited" else 1
            
            product_pitched_map = {
                "Basic": 0, 
                "Standard": 1, 
                "Deluxe": 2, 
                "Super Deluxe": 3, 
                "King": 4
            }
            product_pitched_encoded = product_pitched_map.get(product_pitched, 0)
            
            designation_map = {
                "Executive": 0,
                "Manager": 1,
                "Senior Manager": 2,
                "AVP": 3,
                "VP": 4
            }
            designation_encoded = designation_map.get(designation, 0)
            
            # Create input dataframe - order matters! Match training column order
            input_data = pd.DataFrame({
                'Age': [age],
                'TypeofContact': [type_of_contact_encoded],
                'CityTier': [city_tier],
                'DurationOfPitch': [duration_of_pitch],
                'Occupation': [occupation_encoded],
                'Gender': [gender_encoded],
                'NumberOfPersonVisiting': [number_of_person_visiting],
                'NumberOfFollowups': [number_of_followups],
                'ProductPitched': [product_pitched_encoded],
                'PreferredPropertyStar': [preferred_property_star],
                'MaritalStatus': [marital_status_encoded],
                'NumberOfTrips': [number_of_trips],
                'Passport': [passport],
                'PitchSatisfactionScore': [pitch_satisfaction_score],
                'OwnCar': [own_car],
                'NumberOfChildrenVisiting': [number_of_children],
                'Designation': [designation_encoded],
                'MonthlyIncome': [monthly_income]
            })
            
            # Add dummy column if model expects it (temporary fix)
            try:
                model.predict(input_data)
            except ValueError as e:
                st.warning("Note: Model expects an extra column. Please retrain the model with cleaned data." + str(e))
            
            # Make prediction
            prediction_proba = model.predict_proba(input_data)[0]
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.header(" Prediction Results")
            
            # Main prediction result
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                if prediction == 1:
                    st.success("###  Likely to Purchase")
                else:
                    st.warning("###  Unlikely to Purchase")
            
            with col_result2:
                st.metric(
                    "Purchase Probability", 
                    f"{prediction_proba[1]:.1%}",
                    delta=f"{prediction_proba[1] - 0.5:.1%}" if prediction_proba[1] > 0.5 else None
                )
            
            with col_result3:
                confidence = max(prediction_proba)
                st.metric("Confidence Level", f"{confidence:.1%}")
            
            # Probability visualization
            st.subheader(" Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Will Not Purchase', 'Will Purchase'],
                'Probability': prediction_proba
            })
            
            import plotly.express as px
            fig = px.bar(
                prob_df, 
                x='Outcome', 
                y='Probability',
                color='Probability',
                color_continuous_scale=['red', 'green'],
                text=prob_df['Probability'].apply(lambda x: f'{x:.1%}')
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.subheader(" Recommendation")
            if prediction == 1:
                st.success("""
                **HIGH PRIORITY CUSTOMER**
                - Schedule a follow-up call within 24 hours
                - Offer personalized package deals
                - Highlight wellness benefits matching their profile
                - Consider offering early booking discounts
                """)
            else:
                if prediction_proba[1] > 0.3:
                    st.info("""
                    **MEDIUM PRIORITY CUSTOMER**
                    - Nurture with targeted email campaigns
                    - Share customer testimonials and success stories
                    - Offer limited-time promotions
                    - Schedule follow-up after 1 week
                    """)
                else:
                    st.warning("""
                    **LOW PRIORITY CUSTOMER**
                    - Add to general marketing list
                    - Send periodic newsletters
                    - Re-evaluate after 1 month
                    - Focus resources on higher probability leads
                    """)
            
            # Feature contribution (if available)
            if hasattr(model, 'feature_importances_'):
                with st.expander(" Model Insights"):
                    st.write("Top factors influencing this prediction:")
                    feature_importance = pd.DataFrame({
                        'Feature': input_data.columns,
                        'Value': input_data.iloc[0].values
                    })
                    st.dataframe(feature_importance, use_container_width=True)
            
        except Exception as e:
            st.error(f" Error making prediction: {e}")
            st.error("Please check that all inputs are valid and try again.")
            import traceback
            st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()