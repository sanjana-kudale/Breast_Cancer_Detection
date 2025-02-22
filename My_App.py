import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

# Background Image Setup
def set_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    page_bg_img = f'''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg_image("D:\techathon_jupyter_notebook\breast-cancer-powerpoint-template.jpg")

# Load Trained CNN Model
@st.cache_resource
def load_cnn_model():
    with open("breast_cancer_model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("breast_cancer_model.h5")  # Ensure these files exist
    return model

model = load_cnn_model()

# UI Layout
st.title("Breast Cancer Detection System")
st.write("Upload medical reports to analyze for breast cancer.")

# Patient Information Form
with st.form("patient_form"):
    name = st.text_input("Patient Name")
    gender = st.radio("Gender", ("Male", "Female"))
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    uploaded_file = st.file_uploader("Upload Breast Cancer Scan (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    submit_button = st.form_submit_button("Submit & Analyze")

if submit_button:
    if name and uploaded_file:
        # Read and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))  # Adjust based on your CNN model input
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Model expects batch input
        
        # Model Prediction
        prediction = model.predict(image_array)
        diagnosis = "Cancerous" if prediction[0][0] > 0.5 else "Non-Cancerous"
        
        st.success(f"Analysis Complete: {diagnosis}")
    else:
        st.warning("Please fill all the required fields.")