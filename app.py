import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide" 
)

# Function to load and encode the logo image
def get_logo_base64():
    # You would replace this with your actual logo loading code
    # For demo purposes, we'll assume the image is saved as 'kidney_logo.png'
    try:
        with open('logo.jpeg', 'rb') as f:
            image_data = f.read()
            b64_encoded = base64.b64encode(image_data).decode()
            return b64_encoded
    except FileNotFoundError:
        st.warning("Logo file not found. Please upload your logo.")
        uploaded_logo = st.file_uploader("Upload your logo", type=['png', 'jpg', 'jpeg'])
        if uploaded_logo:
            image_data = uploaded_logo.read()
            # Save the uploaded logo
            with open('logo.jpeg', 'wb') as f:
                f.write(image_data)
            b64_encoded = base64.b64encode(image_data).decode()
            return b64_encoded
        return None

# Add custom CSS for styling with logo
def get_custom_css(logo_base64=None):
    logo_css = ""
    if logo_base64:
        logo_css = f"""
        .logo-img {{
            display: block;
            margin: 0 auto 20px auto;
            max-width: 250px;
            max-height: 250px;
        }}
        """
    
    return f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            color: #0B4F6C;
            text-align: center;
            margin-bottom: 10px;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: #FAF9F6;
            margin-bottom: 30px;
            text-align: center;
        }}
        .prediction-text {{
            font-size: 1.8rem;
            font-weight: bold;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .stButton>button {{
            background-color: #0B4F6C;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #1976D2;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .css-1d391kg {{
            padding-top: 3rem;
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 3rem;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            font-size: 0.9rem;
            color: #666;
            border-top: 1px solid #eee;
        }}
        .stExpander {{
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .info-box {{
            background-color: #f8f9fa;
            border-left: 4px solid #0B4F6C;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        .risk-factor {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin: 5px 0;
            border-left: 3px solid #ff9800;
        }}
        {logo_css}
    </style>
    """

# Function to load models
@st.cache_resource
def load_models():
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory as the app.")
        return None, None

# Function to predict kidney disease
def predict_kidney_disease(input_data, model, scaler):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])
    
    # Order columns
    selected_features = ['BMI', 'SystolicBP', 'FastingBloodSugar', 'SerumCreatinine', 'BUNLevels', 'GFR',
                        'ProteinInUrine', 'SerumElectrolytesSodium', 'SerumElectrolytesPotassium',
                        'SerumElectrolytesPhosphorus', 'CholesterolHDL', 'CholesterolTriglycerides',
                        'MuscleCramps', 'Itching', 'HealthLiteracy']
    
    df = df[selected_features]
    
    # Normalize the data
    df_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    return prediction, probability

# Function to display metrics in a nicer format
def display_metric_card(title, value, unit, icon, description=None, color="#0B4F6C"):
    st.markdown(f"""
    <div style="padding: 15px; background-color: white; border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px; 
                border-left: 5px solid {color};">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 24px; margin-right: 10px; color: {color};">{icon}</div>
            <div>
                <div style="font-size: 0.9rem; color: #666;">{title}</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #333;">{value} {unit}</div>
                {f'<div style="font-size: 0.8rem; color: #888; margin-top: 5px;">{description}</div>' if description else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main app layout
def main():
    # Load models
    model, scaler = load_models()
    
    # Get logo in base64
    logo_base64 = get_logo_base64()
    
    # Apply custom CSS
    st.markdown(get_custom_css(logo_base64), unsafe_allow_html=True)
    
    # Display logo if available
    if logo_base64:
        st.markdown(f'<img src="data:image/png;base64,{logo_base64}" class="logo-img">', unsafe_allow_html=True)
    
    # App header
    st.markdown('<h1 class="main-header">Chronic Kidney Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter patient information to predict chronic kidney disease risk</p>', unsafe_allow_html=True)
    
    # If models couldn't be loaded, don't show the form
    if model is None or scaler is None:
        st.warning("Please upload the model files or place them in the app directory to proceed.")
        
        # Add file uploaders for model files
        uploaded_model = st.file_uploader("Upload model.pkl file", type=["pkl"])
        uploaded_scaler = st.file_uploader("Upload scaler.pkl file", type=["pkl"])
        
        if uploaded_model and uploaded_scaler:
            with open("model.pkl", "wb") as f:
                f.write(uploaded_model.getbuffer())
            with open("scaler.pkl", "wb") as f:
                f.write(uploaded_scaler.getbuffer())
            st.success("Files uploaded successfully! Please refresh the page.")
        return
    
    # Create tabs for data entry and information
    tab1 = st.tabs(["Patient Assessment"])[0]

    
    with tab1:
        # Create a form for input to ensure all data is collected before prediction
        with st.form("patient_data_form"):
            st.subheader("Patient Information")
            
            # Create three columns for input parameters
            col1, col2, col3 = st.columns(3)
            
            # Column 1: Health metrics
            with col1:
                st.markdown('<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: bold; color: #0B4F6C; margin-bottom: 10px;">Basic Health Metrics</div>', unsafe_allow_html=True)
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, 
                                    help="Body Mass Index (kg/m²)")
                systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=120, 
                                            help="Upper value of blood pressure reading")
                fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70, max_value=300, value=100, 
                                                    help="Blood glucose level after fasting")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Column 2: Kidney function tests
            with col2:
                st.markdown('<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: bold; color: #0B4F6C; margin-bottom: 10px;">Kidney Function Tests</div>', unsafe_allow_html=True)
                serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.4, max_value=15.0, value=1.0, step=0.1, 
                                                help="Waste product filtered by kidneys")
                bun_levels = st.number_input("BUN Levels (mg/dL)", min_value=5, max_value=150, value=15, 
                                            help="Blood Urea Nitrogen levels")
                gfr = st.number_input("GFR (mL/min/1.73m²)", min_value=1, max_value=120, value=90, 
                                    help="Glomerular Filtration Rate - kidney function measure")
                protein_in_urine = st.slider("Protein in Urine", min_value=0, max_value=3, value=0, 
                                            help="0=None, 1=Trace, 2=Moderate, 3=Heavy")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Column 3: Additional tests
            with col3:
                st.markdown('<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: bold; color: #0B4F6C; margin-bottom: 10px;">Electrolytes & Cholesterol</div>', unsafe_allow_html=True)
                sodium = st.number_input("Serum Sodium (mEq/L)", min_value=125, max_value=150, value=140, 
                                        help="Sodium electrolyte level")
                potassium = st.number_input("Serum Potassium (mEq/L)", min_value=3.0, max_value=7.0, value=4.5, step=0.1, 
                                        help="Potassium electrolyte level")
                phosphorus = st.number_input("Serum Phosphorus (mg/dL)", min_value=2.0, max_value=10.0, value=3.5, step=0.1, 
                                            help="Phosphorus level")
                hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50, 
                                    help="'Good' cholesterol level")
                triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=40, max_value=600, value=150, 
                                            help="Type of fat in the blood")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Symptoms and health literacy
            st.markdown('<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 15px;">', unsafe_allow_html=True)
            st.markdown('<div style="font-weight: bold; color: #0B4F6C; margin-bottom: 10px;">Symptoms & Additional Information</div>', unsafe_allow_html=True)
            col4, col5 = st.columns(2)
            
            with col4:
                muscle_cramps = st.checkbox("Patient experiences muscle cramps")
                itching = st.checkbox("Patient experiences itching")
            
            with col5:
                health_literacy = st.radio("Patient's Health Literacy", 
                                        ["Good understanding of health conditions", "Limited understanding of health conditions"],
                                        index=0)
                health_literacy_value = 0 if health_literacy == "Good understanding of health conditions" else 1
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button("Predict Kidney Disease", use_container_width=True)
        
        # Prepare input data
        input_data = {
            'BMI': bmi,
            'SystolicBP': systolic_bp,
            'FastingBloodSugar': fasting_blood_sugar,
            'SerumCreatinine': serum_creatinine,
            'BUNLevels': bun_levels,
            'GFR': gfr,
            'ProteinInUrine': protein_in_urine,
            'SerumElectrolytesSodium': sodium,
            'SerumElectrolytesPotassium': potassium,
            'SerumElectrolytesPhosphorus': phosphorus,
            'CholesterolHDL': hdl,
            'CholesterolTriglycerides': triglycerides,
            'MuscleCramps': 1 if muscle_cramps else 0,
            'Itching': 1 if itching else 0,
            'HealthLiteracy': health_literacy_value
        }
        
        # Process when form is submitted
        if submitted:
            prediction, probability = predict_kidney_disease(input_data, model, scaler)
            
            st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            # Display prediction with better styling
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-text" style="background-color: #ffcdd2;">'
                    f'<div style="font-size: 2rem; margin-bottom: 10px; color: #e30922;">⚠️ CKD Risk Detected</div>'
                    f'<div style="font-size: 1.2rem; color: #e30922;">Confidence: {probability:.2%}</div>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Add recommendation for positive prediction with better styling
                st.markdown(
                    '<div class="info-box" style="color: #1a1919;">'
                    '<div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 10px; color: #191561">📋 Recommendation</div>'
                    'Based on the analysis, there\'s a significant risk of Chronic Kidney Disease. '
                    'Please consult with a nephrologist for a thorough evaluation and proper diagnosis.'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Risk factors identified with better visualization
                st.subheader("Key Risk Factors Identified")
                
                # Create columns for risk factors
                risk_factors = []
                
                if gfr < 60:
                    risk_factors.append(("Low GFR", f"{gfr} mL/min/1.73m²", "Filter function of kidneys is reduced"))
                if serum_creatinine > 1.2:
                    risk_factors.append(("Elevated Serum Creatinine", f"{serum_creatinine} mg/dL", "Indicates decreased kidney function"))
                if bun_levels > 20:
                    risk_factors.append(("Elevated BUN Levels", f"{bun_levels} mg/dL", "Suggests kidneys aren't removing urea efficiently"))
                if protein_in_urine > 0:
                    severity = ["Trace", "Moderate", "Heavy"][protein_in_urine-1] if protein_in_urine > 0 else "None"
                    risk_factors.append(("Protein in Urine", f"Level: {severity}", "Indicates kidney damage"))
                if systolic_bp > 140:
                    risk_factors.append(("High Blood Pressure", f"{systolic_bp} mmHg", "Can damage blood vessels in kidneys"))
                
                # Display risk factors in a grid
                if risk_factors:
                    cols = st.columns(min(3, len(risk_factors)))
                    for i, (factor, value, desc) in enumerate(risk_factors):
                        with cols[i % len(cols)]:
                            st.markdown(
                                f'<div class="risk-factor">'
                                f'<div style="font-weight: bold; color: #d32f2f;">{factor}</div>'
                                f'<div style="font-size: 1.1rem; color: #666;">{value}</div>'
                                f'<div style="font-size: 0.9rem; color: #666;">{desc}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No specific high-risk factors identified, but overall metrics suggest CKD risk.")
                
            else:
                st.markdown(
                    f'<div class="prediction-text" style="background-color: #c8e6c9;">'
                    f'<div style="font-size: 2rem; margin-bottom: 10px; color: #22c727;">✅ No CKD Detected</div>'
                    f'<div style="font-size: 1.2rem; color: #22c727;">Confidence: {1-probability:.2%}</div>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Add recommendation for negative prediction
                st.markdown(
                    '<div class="info-box" style="color: #1a1919;">'
                    '<div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 10px; color: #191561">📋 Recommendation</div>'
                    'The analysis indicates low risk of Chronic Kidney Disease at this time. '
                    'Continue with regular health check-ups and maintain healthy lifestyle habits.'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Display healthy ranges
                st.subheader("Key Metrics Overview")
                cols = st.columns(3)
                
                # Display metrics with nice formatting
                with cols[0]:
                    gfr_status = "Normal" if gfr >= 90 else "Mildly Reduced" if gfr >= 60 else "Reduced"
                    gfr_color = "#4caf50" if gfr >= 90 else "#ff9800" if gfr >= 60 else "#f44336"
                    display_metric_card("GFR", gfr, "mL/min", "⚡", 
                                      f"Status: {gfr_status}", gfr_color)
                
                with cols[1]:
                    creat_status = "Normal" if serum_creatinine <= 1.2 else "Elevated"
                    creat_color = "#4caf50" if serum_creatinine <= 1.2 else "#f44336"
                    display_metric_card("Creatinine", serum_creatinine, "mg/dL", "🔬", 
                                       f"Status: {creat_status}", creat_color)
                
                with cols[2]:
                    bp_status = "Normal" if systolic_bp < 120 else "Elevated" if systolic_bp < 130 else "High"
                    bp_color = "#4caf50" if systolic_bp < 120 else "#ff9800" if systolic_bp < 130 else "#f44336"
                    display_metric_card("Blood Pressure", systolic_bp, "mmHg", "❤️", 
                                       f"Status: {bp_status}", bp_color)
    

  
    # Add explanatory section
    with st.expander("About the Prediction Model"):
        st.write("""
        This prediction system uses a machine learning model to assess the risk of Chronic Kidney Disease (CKD) 
        based on clinical parameters and patient information.
        
        **The key indicators in this model include:**
        - **GFR (Glomerular Filtration Rate):** A test that measures kidney function
        - **Serum Creatinine:** A waste product that healthy kidneys filter out
        - **Blood Urea Nitrogen (BUN):** Another measure of kidney function
        - **Protein in Urine:** Indicates kidney damage when present
        - **Other factors:** Including electrolytes, blood pressure, and symptoms
        
        The model analyzes these parameters collectively to determine the likelihood of chronic kidney disease. The prediction 
        is based on statistical patterns identified from training data of known CKD and non-CKD patients.
        
        This tool is designed to assist healthcare professionals and should not replace clinical judgment or proper medical evaluation.
        """)
    
    # Footer
    st.markdown(
        '<div class="footer">'
        '© 2025 Chronic Kidney Disease Prediction System | For clinical decision support only<br>'
        '<span style="font-size: 0.8rem;">This tool is intended for use by healthcare professionals. Results should be interpreted within the clinical context.</span>'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()