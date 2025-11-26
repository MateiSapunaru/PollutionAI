import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image

st.set_page_config(
    page_title="Soil Pollution & Disease AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1B5E20;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F1F8E9 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        margin: 2rem 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

DATASET_VALUES = {
    'regions': ['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'South America'],
    'countries': ['Argentina', 'Australia', 'Bangladesh', 'Brazil', 'Canada', 'Chile', 'China', 
                  'Colombia', 'Czech Republic', 'Egypt', 'Ethiopia', 'France', 'Germany', 'Ghana', 
                  'India', 'Indonesia', 'Italy', 'Japan', 'Kenya', 'Mexico', 'New Zealand', 'Nigeria', 
                  'Pakistan', 'Peru', 'Poland', 'Romania', 'South Africa', 'South Korea', 'Spain', 
                  'Tanzania', 'Thailand', 'UK', 'USA', 'Venezuela', 'Vietnam'],
    'pollutants': ['Arsenic', 'Cadmium', 'Chromium', 'Lead', 'Mercury'],
    'soil_textures': ['Clay', 'Loamy', 'Sandy', 'Silty'],
    'crop_types': ['Barley', 'Corn', 'Cotton', 'Potato', 'Rice', 'Soybean', 'Vegetables', 'Wheat'],
    'farming_practices': ['Conventional', 'Integrated', 'Organic', 'Permaculture'],
    'industries': ['Agriculture', 'Battery', 'Chemical', 'E-waste', 'Mining', 'Smelting', 'Tannery'],
    'water_sources': ['Borehole', 'Irrigation Canal', 'Lake', 'Rainwater', 'River', 'Well'],
    'disease_types': ['Anemia', 'Bone Disease', 'Cancer', 'Cardiovascular Disease', 'Developmental Disorder', 
                      'Diabetes', 'Kidney Disease', 'Neurological Disorder', 'None Detected', 
                      'Respiratory Issues', 'Skin Disease'],
    'health_symptoms': ['Abdominal Pain', 'Bone Pain', 'Breathing Difficulty', 'Cognitive Impairment', 
                        'Coordination Problems', 'Dermatitis', 'Fatigue', 'Headache', 'Hyperpigmentation', 
                        'Joint Pain', 'Memory Loss', 'Mood Changes', 'Nasal Irritation', 'Nausea', 
                        'Numbness', 'Protein in Urine', 'Rash', 'Skin Lesions', 'Skin Ulcers', 'Tremors', 
                        'Vision Problems', 'Weakness'],
    'age_groups': ['Adults', 'Children', 'Elderly'],
    'genders': ['Both', 'Female', 'Male']
}

@st.cache_resource
def load_concentration_model():
    """Load concentration prediction model"""
    try:
        paths = ['models/concentration_model.pkl', 'concentration_model.pkl']
        for path in paths:
            if os.path.exists(path):
                return joblib.load(path)
        return None
    except Exception as e:
        st.error(f"Error loading concentration model: {e}")
        return None

@st.cache_resource
def load_disease_model():
    """Load disease type prediction model"""
    try:
        paths = ['models/disease_type_model.pkl', 'disease_type_model.pkl']
        for path in paths:
            if os.path.exists(path):
                return joblib.load(path)
        return None
    except Exception as e:
        st.error(f"Error loading disease model: {e}")
        return None


def main():
    """Main application"""
    
    st.markdown('<h1 class="main-header">Soil Pollution & Disease AI System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; color: #555; margin-bottom: 2rem;'>
    Advanced AI-powered system for pollutant concentration and disease prediction
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select a tool:",
        ["Home", "Concentration Predictor", "Disease Predictor", 
         "Dataset Visualizations", "Model Performance", "About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Train models using concentration_ai.py and disease_ai.py first!")
    
    if page == "Home":
        show_home()
    elif page == "Concentration Predictor":
        show_concentration_predictor()
    elif page == "Disease Predictor":
        show_disease_predictor()
    elif page == "Dataset Visualizations":
        show_dataset_visualizations()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "About":
        show_about()


def show_home():
    """Home page"""
    st.markdown('<h2 class="sub-header">Welcome to Soil Pollution & Disease AI</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>What We Offer</h3>
        <ul>
            <li><b>Concentration Prediction:</b> Predict pollutant levels using 17 features</li>
            <li><b>Disease Detection:</b> Identify disease types using 21 features</li>
            <li><b>Scientific Analysis:</b> pH & bioavailability modeling</li>
            <li><b>Risk Assessment:</b> Comprehensive evaluation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>Key Features</h3>
        <ul>
            <li>ML models (Random Forest, XGBoost, LightGBM)</li>
            <li>pH-based bioavailability analysis</li>
            <li>Soil texture effects</li>
            <li>Distance decay modeling</li>
            <li>Age-specific vulnerability factors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        conc_model = load_concentration_model()
        if conc_model:
            st.success("Concentration Model: Loaded")
            st.caption(f"Model: {conc_model.get('model_name', 'Unknown')}")
            st.caption(f"Features: {len(conc_model.get('feature_names', []))}")
        else:
            st.error("Concentration Model: Not Found")
            st.info("Run: python concentration_ai.py")
    
    with col2:
        disease_model = load_disease_model()
        if disease_model:
            st.success("Disease Model: Loaded")
            st.caption(f"Model: {disease_model.get('model_name', 'Unknown')}")
            st.caption(f"Features: {len(disease_model.get('feature_names', []))}")
        else:
            st.error("Disease Model: Not Found")
            st.info("Run: python disease_ai.py")

    st.info("Select a tool from the sidebar!")


def show_concentration_predictor():
    """Concentration prediction with all 17 features using CORRECT values"""
    st.markdown('<h2 class="sub-header">Pollutant Concentration Predictor</h2>', unsafe_allow_html=True)
    
    model_package = load_concentration_model()
    
    if model_package is None:
        st.error("Concentration model not found. Please train the model first.")
        st.code("python concentration_ai.py", language="python")
        return
    st.success(f"Model: {model_package['model_name']}")
    
    required_features = model_package['feature_names']
    st.info(f"Uses {len(required_features)} features")
    
    with st.expander("Required Features"):
        for i, feature in enumerate(required_features, 1):
            st.write(f"{i}. {feature}")
    
    st.markdown("---")
    st.markdown("### Input Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pollutant_type = st.selectbox("Pollutant Type", DATASET_VALUES['pollutants'])
        bioavailable = st.number_input("Bioavailable Concentration (mg/kg)", 
                                     min_value=0.0, max_value=1000.0, value=20.0)
        
    with col2:
        soil_ph = st.slider("Soil pH", min_value=4.0, max_value=8.5, value=6.5, step=0.1)
        soil_texture = st.selectbox("Soil Texture", DATASET_VALUES['soil_textures'])
        
    with col3:
        soil_om = st.slider("Soil Organic Matter (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
        cec = st.slider("CEC (meq/100g)", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
    
    st.markdown("### Environmental Conditions")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        temperature = st.slider("Temperature (°C)", min_value=-5.0, max_value=45.0, value=20.0, step=0.5)
        humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
        
    with col5:
        rainfall = st.slider("Rainfall (mm)", min_value=100.0, max_value=3000.0, value=500.0, step=10.0)
        region = st.selectbox("Region", DATASET_VALUES['regions'])
        
    with col6:
        country = st.selectbox("Country", DATASET_VALUES['countries'])
        
    st.markdown("### Agricultural Factors")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        crop_type = st.selectbox("Crop Type", DATASET_VALUES['crop_types'])
        farming = st.selectbox("Farming Practice", DATASET_VALUES['farming_practices'])
        
    with col8:
        nearby_industry = st.selectbox("Nearby Industry", DATASET_VALUES['industries'])
        
    with col9:
        distance_km = st.slider("Distance from Source (km)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        years_contaminated = st.slider("Years Since Contamination", min_value=0, max_value=40, value=5)
        water_source = st.selectbox("Water Source Type", DATASET_VALUES['water_sources'])
    
    st.markdown("---")

    if st.button("Predict Concentration", type="primary", use_container_width=True):
        
        # Create input data with all 17 features
        input_data = pd.DataFrame([{
            'Pollutant_Type': pollutant_type,
            'Bioavailable_Concentration_mg_kg': bioavailable,
            'Soil_pH': soil_ph,
            'Soil_Texture': soil_texture,
            'Soil_Organic_Matter_%': soil_om,
            'CEC_meq_100g': cec,
            'Temperature_C': temperature,
            'Humidity_%': humidity,
            'Rainfall_mm': rainfall,
            'Region': region,
            'Country': country,
            'Crop_Type': crop_type,
            'Farming_Practice': farming,
            'Nearby_Industry': nearby_industry,
            'Distance_from_Source_km': distance_km,
            'Years_Since_Contamination': years_contaminated,
            'Water_Source_Type': water_source
        }])
        
        try:
            # Encode categorical variables
            for col, encoder in model_package['encoders'].items():
                if col in input_data.columns:
                    input_data[col] = encoder.transform(input_data[col].astype(str))
            
            # Ensure correct feature order
            input_data = input_data[model_package['feature_names']]
            
            # Scale features
            input_scaled = model_package['scaler'].transform(input_data)
            
            # Predict
            prediction = model_package['model'].predict(input_scaled)[0]
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Concentration", f"{prediction:.2f} mg/kg")
            with col2:
                st.metric("Bioavailable Input", f"{bioavailable:.2f} mg/kg")
            with col3:
                ratio = bioavailable / prediction if prediction > 0 else 0
                st.metric("Bioavailability %", f"{ratio*100:.1f}%")
            with col4:
                risk = "Low" if prediction < 50 else ("Moderate" if prediction < 150 else "High")
                st.metric("Risk Level", f"{risk}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis
            st.markdown("### Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ph_category = "Acidic" if soil_ph < 6.0 else ("Neutral" if soil_ph < 7.5 else "Alkaline")
                st.markdown(f"""
                **Soil Conditions:**
                - pH: {soil_ph:.1f} ({ph_category})
                - Texture: {soil_texture}
                - Organic Matter: {soil_om:.1f}%
                - CEC: {cec:.1f} meq/100g
                """)
            
            with col2:
                st.markdown(f"""
                **Environmental Factors:**
                - Pollutant: {pollutant_type}
                - Region: {region}, {country}
                - Temperature: {temperature}°C
                - Industry: {nearby_industry}
                """)
            
            # Recommendations
            st.markdown("### Recommendations")

            if prediction < 50:
                st.success("Low contamination - continue regular monitoring")
            elif prediction < 150:
                st.warning("Moderate contamination - consider pH adjustment and crop rotation")
            else:
                st.error("High contamination - immediate soil remediation required!")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            with st.expander("Debug Information"):
                st.write("Input data shape:", input_data.shape)
                st.write("Required features:", len(model_package['feature_names']))
                st.write("Model features:", model_package['feature_names'])
                st.write("Available encoders:", list(model_package['encoders'].keys()))


def show_disease_predictor():
    """Disease prediction with all 21 features using CORRECT values"""
    st.markdown('<h2 class="sub-header">Disease Type Predictor</h2>', unsafe_allow_html=True)
    
    model_package = load_disease_model()
    
    if model_package is None:
        st.error("Disease model not found. Please train the model first.")
        st.code("python disease_ai.py", language="python")
        return
    st.success(f"Model: {model_package['model_name']}")
    
    required_features = model_package['feature_names']
    st.info(f"Uses {len(required_features)} features")
    
    with st.expander("Required Features"):
        for i, feature in enumerate(required_features, 1):
            st.write(f"{i}. {feature}")
    
    st.markdown("---")
    st.markdown("### Input Parameters")
    
    # Pollutant information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pollutant_type = st.selectbox("Pollutant Type", DATASET_VALUES['pollutants'], key='d_poll')
        total_conc = st.number_input("Total Concentration (mg/kg)", 
                                   min_value=0.0, max_value=5000.0, value=100.0)
        bioavailable = st.number_input("Bioavailable Concentration (mg/kg)", 
                                     min_value=0.0, max_value=1000.0, value=20.0, key='d_bio')
        
    with col2:
        soil_ph = st.slider("Soil pH", min_value=4.0, max_value=8.5, value=6.5, step=0.1, key='d_ph')
        soil_texture = st.selectbox("Soil Texture", DATASET_VALUES['soil_textures'], key='d_texture')
        soil_om = st.slider("Soil Organic Matter (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.1, key='d_om')
        
    with col3:
        cec = st.slider("CEC (meq/100g)", min_value=1.0, max_value=50.0, value=12.0, step=0.5, key='d_cec')
        temperature = st.slider("Temperature (°C)", min_value=-5.0, max_value=45.0, value=20.0, step=0.5, key='d_temp')
        humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0, key='d_hum')
    
    st.markdown("### Environmental & Geographic")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        rainfall = st.slider("Rainfall (mm)", min_value=100.0, max_value=3000.0, value=500.0, step=10.0, key='d_rain')
        region = st.selectbox("Region", DATASET_VALUES['regions'], key='d_region')
        
    with col5:
        country = st.selectbox("Country", DATASET_VALUES['countries'], key='d_country')
        crop_type = st.selectbox("Crop Type", DATASET_VALUES['crop_types'], key='d_crop')
        
    with col6:
        farming = st.selectbox("Farming Practice", DATASET_VALUES['farming_practices'], key='d_farm')
        nearby_industry = st.selectbox("Nearby Industry", DATASET_VALUES['industries'], key='d_ind')
    
    st.markdown("### Health & Exposure Factors")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        distance = st.slider("Distance from Source (km)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key='d_dist')
        years_contaminated = st.slider("Years Since Contamination", min_value=0, max_value=40, value=5, key='d_years')
        
    with col8:
        water_source = st.selectbox("Water Source Type", DATASET_VALUES['water_sources'], key='d_water')
        health_symptoms = st.selectbox("Health Symptoms", DATASET_VALUES['health_symptoms'], key='d_symptoms')
        
    with col9:
        age_group = st.selectbox("Age Group Affected", DATASET_VALUES['age_groups'], key='d_age')
        gender = st.selectbox("Gender Most Affected", DATASET_VALUES['genders'], key='d_gender')
    
    st.markdown("---")
    
    if st.button("Predict Disease Type", type="primary", use_container_width=True):
        
        # Create input data with all 21 features
        input_data = pd.DataFrame([{
            'Pollutant_Type': pollutant_type,
            'Total_Concentration_mg_kg': total_conc,
            'Bioavailable_Concentration_mg_kg': bioavailable,
            'Soil_pH': soil_ph,
            'Soil_Texture': soil_texture,
            'Soil_Organic_Matter_%': soil_om,
            'CEC_meq_100g': cec,
            'Temperature_C': temperature,
            'Humidity_%': humidity,
            'Rainfall_mm': rainfall,
            'Region': region,
            'Country': country,
            'Crop_Type': crop_type,
            'Farming_Practice': farming,
            'Nearby_Industry': nearby_industry,
            'Distance_from_Source_km': distance,
            'Years_Since_Contamination': years_contaminated,
            'Water_Source_Type': water_source,
            'Health_Symptoms': health_symptoms,
            'Age_Group_Affected': age_group,
            'Gender_Most_Affected': gender
        }])
        
        try:
            # Encode categorical variables
            for col, encoder in model_package['encoders'].items():
                if col in input_data.columns:
                    input_data[col] = encoder.transform(input_data[col].astype(str))
            
            # Ensure correct feature order
            input_data = input_data[model_package['feature_names']]
            
            # Scale features
            input_scaled = model_package['scaler'].transform(input_data)
            
            # Predict
            disease_pred = model_package['model'].predict(input_scaled)[0]
            disease_proba = model_package['model'].predict_proba(input_scaled)[0]
            
            # Get disease type
            disease_type = model_package['target_encoder'].inverse_transform([disease_pred])[0]
            confidence = max(disease_proba) * 100
            
            vuln_mult = {'Children': 3.0, 'Elderly': 1.5, 'Adults': 1.0}[age_group]
            time_mult = min(2.0, 0.5 + (years_contaminated / 10))
            effective_conc = bioavailable * vuln_mult * time_mult
            
            severity = "Mild" if effective_conc < 10 else ("Moderate" if effective_conc < 50 else "Severe")
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Disease Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Disease Type", disease_type)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col3:
                st.metric("Severity", severity)
            with col4:
                st.metric("Vulnerability Factor", f"{vuln_mult:.1f}x")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Exposure Profile:**
                - Pollutant: {pollutant_type}
                - Total Concentration: {total_conc:.1f} mg/kg
                - Bioavailable: {bioavailable:.1f} mg/kg
                - Effective Concentration: {effective_conc:.1f} mg/kg
                """)
            
            with col2:
                st.markdown(f"""
                **Risk Factors:**
                - Age Group: {age_group} ({vuln_mult}x vulnerability)
                - Exposure Duration: {years_contaminated} years
                - Symptoms: {health_symptoms}
                - Distance from Source: {distance} km
                """)
            
            st.markdown("### Health Recommendations")

            if severity == "Mild":
                st.success("Mild risk - Regular health monitoring recommended")
            elif severity == "Moderate":
                st.warning("Moderate risk - Medical screening and lifestyle changes advised")
            else:
                st.error("Severe risk - Immediate medical consultation required!")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            with st.expander("Debug Information"):
                st.write("Input data shape:", input_data.shape)
                st.write("Required features:", len(model_package['feature_names']))
                st.write("Model features:", model_package['feature_names'])
                st.write("Available encoders:", list(model_package['encoders'].keys()))


def show_dataset_visualizations():
    st.markdown('<h2 class="sub-header">📊 Dataset Analysis Visualizations</h2>', unsafe_allow_html=True)
    
    viz_dir = 'visualizations'
    if not os.path.exists(viz_dir):
        st.error("⚠️ 'visualizations' folder not found. Please generate the analysis plots first.")
        st.code("python visualization.py", language="bash")
        return
    
    dataset_viz_files = [
        'ph_bioavailability_analysis.png',
        'soil_texture_analysis.png',
        'distance_decay_analysis.png',
        'pollutant_analysis.png',
        'age_vulnerability_analysis.png',
        'industry_patterns.png'
    ]
    
    available_viz = []
    for viz_file in dataset_viz_files:
        viz_path = os.path.join(viz_dir, viz_file)
        if os.path.exists(viz_path):
            available_viz.append((viz_file, viz_path))
    
    if not available_viz:
        st.warning("No dataset visualization images found in visualizations/ folder.")
        st.info("Generate them by running:")
        st.code("python visualization.py", language="bash")
        return
    
    st.success(f"Found {len(available_viz)} dataset visualization(s)")
    
    tab_names = []
    tab_contents = []
    
    for viz_name, viz_path in available_viz:
        tab_name = viz_name.replace('_', ' ').replace('.png', '').title()
        tab_names.append(tab_name)
        tab_contents.append((viz_name, viz_path))
    
    tabs = st.tabs(tab_names)
    
    for i, tab in enumerate(tabs):
        with tab:
            viz_name, viz_path = tab_contents[i]
            try:
                image = Image.open(viz_path)
                st.image(image, caption=tab_names[i], use_container_width=True)
            except Exception as e:
                st.error(f"Error loading {viz_name}: {str(e)}")


def show_model_performance():
    st.markdown('<h2 class="sub-header">🔍 Model Performance Visualizations</h2>', unsafe_allow_html=True)
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        st.error("⚠️ Models directory not found. Train the models first.")
        st.code("python concentration_ai.py\npython disease_ai.py", language="bash")
        return
    
    model_viz_files = [
        'concentration_model_comparison.png',
        'concentration_feature_importance.png',
        'concentration_residual_analysis.png',
        'disease_type_model_comparison.png',
        'disease_type_confusion_matrices.png'
    ]
    
    available_viz = []
    for viz_file in model_viz_files:
        viz_path = os.path.join(models_dir, viz_file)
        if os.path.exists(viz_path):
            available_viz.append((viz_file, viz_path))
    
    if not available_viz:
        st.warning("⚠️ No model performance visualizations found in models/ folder.")
        st.info("Train the models to generate performance visualizations:")
        st.code("python concentration_ai.py\npython disease_ai.py", language="bash")
        return
    
    st.success(f"✅ Found {len(available_viz)} model performance visualization(s)")
    
    concentration_viz = []
    disease_viz = []
    
    for viz_name, viz_path in available_viz:
        if 'concentration' in viz_name:
            concentration_viz.append((viz_name, viz_path))
        elif 'disease' in viz_name:
            disease_viz.append((viz_name, viz_path))
    
    if concentration_viz or disease_viz:
        tabs = st.tabs(["📊 Concentration Model", "🏥 Disease Model"])
        
        with tabs[0]:
            if concentration_viz:
                st.markdown("### Concentration Model Performance")
                for viz_name, viz_path in concentration_viz:
                    display_name = viz_name.replace('concentration_', '').replace('_', ' ').replace('.png', '').title()
                    try:
                        image = Image.open(viz_path)
                        st.image(image, caption=display_name, use_container_width=True)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error loading {viz_name}: {str(e)}")
            else:
                st.info("No concentration model visualizations found.")
        
        with tabs[1]:
            if disease_viz:
                st.markdown("### Disease Model Performance")
                for viz_name, viz_path in disease_viz:
                    display_name = viz_name.replace('disease_type_', '').replace('_', ' ').replace('.png', '').title()
                    try:
                        image = Image.open(viz_path)
                        st.image(image, caption=display_name, use_container_width=True)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error loading {viz_name}: {str(e)}")
            else:
                st.info("No disease model visualizations found.")
    
    st.markdown("### Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        conc_model = load_concentration_model()
        if conc_model:
            st.success("✅ Concentration Model Loaded")
            st.caption(f"**Model:** {conc_model.get('model_name', 'Unknown')}")
            st.caption(f"**Features:** {len(conc_model.get('feature_names', []))}")
            
            with st.expander("🔍 Feature List"):
                for i, feature in enumerate(conc_model.get('feature_names', []), 1):
                    st.write(f"{i}. {feature}")
        else:
            st.error("❌ Concentration Model Not Found")
    
    with col2:
        disease_model = load_disease_model()
        if disease_model:
            st.success("✅ Disease Model Loaded")
            st.caption(f"**Model:** {disease_model.get('model_name', 'Unknown')}")
            st.caption(f"**Features:** {len(disease_model.get('feature_names', []))}")
            
            with st.expander("🔍 Feature List"):
                for i, feature in enumerate(disease_model.get('feature_names', []), 1):
                    st.write(f"{i}. {feature}")
        else:
            st.error("❌ Disease Model Not Found")


def show_about():
    st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🌱 Soil Pollution & Disease AI System
    
    This is an advanced machine learning system for predicting soil pollutant concentrations 
    and associated disease risks based on environmental and exposure factors.
    
    **🔬 Scientific Foundation:**
    - pH-metal mobility relationships (2-5x increase in acidic soils)
    - Soil texture effects on bioavailability (up to 8x difference)
    - Age-specific vulnerability factors (children 3-5x more susceptible)
    - Distance decay modeling from contamination sources
    - Bioavailability modeling based on soil chemistry
    
    **🤖 AI Models:**
    - **Concentration Predictor**: Uses 17 features with Random Forest/XGBoost/LightGBM
    - **Disease Predictor**: Uses 21 features with advanced classifiers
    
    **📊 Dataset Information:**
    - 2,000+ soil contamination cases from global locations
    - Based on WHO/EPA contamination standards
    - Real categorical values from actual dataset
    
    **⚠️ IMPORTANT - Fixed Categorical Values:**
    This version uses the EXACT categorical values from the dataset that the models were trained on.
    
    **🚀 Usage Instructions:**
    1. Train models: `python concentration_ai.py` and `python disease_ai.py`
    2. Generate dataset visualizations: `python visualization.py`
    3. Run app: `streamlit run app.py`
    
    **⚠️ Disclaimer:**
    For educational and research purposes only. Consult professionals for real-world applications.
    """)


if __name__ == "__main__":
    main()
