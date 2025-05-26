import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
@st.cache_resource
def load_model():
   model = tf.keras.models.load_model('cifar10_cnn_model.keras')

model = load_model()

# CIFAR-10 classes
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.title("CIFAR-10 Image Classification with CNN")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Load and preprocess image
    image = Image.open(uploaded_file)         # uploaded_file from st.file_uploader
    image = image.resize((32, 32))            # Resize to match CIFAR-10 input
    image = np.array(image)                   # Convert to array
    image = image / 255.0                     # Normalize to [0, 1]
    img_array = np.expand_dims(image, axis=0) # Shape: (1, 32, 32, 3)
  
    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")
