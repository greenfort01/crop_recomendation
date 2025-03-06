import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.inspection import permutation_importance
import joblib

pd.options.display.max_colwidth = 2000
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide"
)

# Load dataset files with error handling
try:
    df_desc = pd.read_csv('Dataset/Crop_Desc.csv', sep=';', encoding='utf-8')
    df = pd.read_csv('Dataset/Crop_recommendation.csv')
except FileNotFoundError:
    st.error("Dataset files not found! Ensure 'Dataset/Crop_Desc.csv' and 'Dataset/Crop_recommendation.csv' exist.")
    st.stop()

# Load the correct model (check if 'knn.pkl' or 'RDF_model.pkl' exists)
try:
    rdf_clf = joblib.load('Model/randomforest.pkl')  # Assuming you want the RandomForest model
except FileNotFoundError:
    st.error("Model file not found! Ensure 'Model/RDF_model.pkl' exists.")
    st.stop()

X = df.drop('label', axis=1)
y = df['label']

st.markdown("<h3 style='text-align: center;'>Enter values to predict the best crop.</h3>", unsafe_allow_html=True)

# User Inputs
col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,4,1,4,1,1])

with col3:
    n_input = st.number_input('Nitrogen (N) (kg/ha)', min_value=0, max_value=140)
    p_input = st.number_input('Phosphorus (P) (kg/ha)', min_value=5, max_value=145)
    k_input = st.number_input('Potassium (K) (kg/ha)', min_value=5, max_value=205)
    temp_input = st.number_input('Temperature (°C)', min_value=9., max_value=43., step=1.)

with col5:
    hum_input = st.number_input('Humidity (%)', min_value=15., max_value=99., step=1.)
    ph_input = st.number_input('pH Level', min_value=3.6, max_value=9.9, step=0.1)
    rain_input = st.number_input('Rainfall (mm)', min_value=21.0, max_value=298.0, step=0.1)
    location = st.selectbox('Select Location', ['Central India', 'Eastern India', 'North Eastern India', 'Northern India', 'Western India', 'Other'])

location_mapping = {
    'Central India': [1, 0, 0, 0, 0, 0],
    'Eastern India': [0, 1, 0, 0, 0, 0],
    'North Eastern India': [0, 0, 1, 0, 0, 0],
    'Northern India': [0, 0, 0, 1, 0, 0],
    'Western India': [0, 0, 0, 0, 1, 0],
    'Other': [0, 0, 0, 0, 0, 1]
}
predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input] + location_mapping[location]]

# Predict Button
if st.button('Recommend Crop'):
    rdf_predicted_value = rdf_clf.predict(predict_inputs)[0]
    st.markdown(f"""<h3 style='text-align: center;'>Best Crop to Plant: <b>{rdf_predicted_value}</b></h3>""", unsafe_allow_html=True)
    
    # Fix image loading issue
    df_desc['label'] = df_desc['label'].str.strip()
    df_desc['image'] = df_desc['image'].str.strip()
    crop_data = df_desc[df_desc['label'] == rdf_predicted_value]
    
    if not crop_data.empty:
        img_path = crop_data['image'].values[0]
        try:
            image = Image.open(img_path)
            st.image(image, caption=f"{rdf_predicted_value}", use_column_width=True)
        except FileNotFoundError:
            st.warning("Image not found for this crop.")
    else:
        st.warning("No image available for this crop.")
    
    st.write("### Crop Statistics")
    st.dataframe(df[df['label'] == rdf_predicted_value].describe())

# Hide Streamlit's default footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
