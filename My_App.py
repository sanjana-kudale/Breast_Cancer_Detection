import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import os
import numpy as np
from PIL import Image

### Load Model ###
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")  # Ensure correct weight file name
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

### Streamlit UI ###
st.title("Breast Cancer Detection using CNN")
st.write("Upload an image to check for cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((25, 25))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = loaded_model.predict(image)
    has_cancer = f'The percentage of cancer: {round(prediction[0][0] * 100, 2)}%'
    has_no_cancer = f'The percentage of no cancer: {round(prediction[0][1] * 100, 2)}%'

    st.write("### Results:")
    st.write(has_cancer)
    st.write(has_no_cancer)
