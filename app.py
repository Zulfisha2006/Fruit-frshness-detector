import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("fruit_classifier_model.h5")

# Config
IMAGE_SIZE = (150, 150)

# Title
st.title("🍎🥭🍌 Fruit Freshness Detection App ")
st.write("Upload a fruit image (apple, banana, or orange), and I will tell you if it's *fresh* or *rotten*!")

# Upload image
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Fruit Image', use_column_width=True)

    # Preprocess
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Fresh  fruit" if prediction < 0.5 else "Rotten fruit"
    confidence = 1 - prediction if prediction < 0.5 else prediction

    # Result
    st.markdown(f"### Prediction: *{label}*")
    st.write(f"Confidence: {confidence:.2f}")