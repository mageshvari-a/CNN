import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cifar10_cnn_model.keras')

model = load_model()

# CIFAR-10 class labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.title("CIFAR-10 Image Classification with CNN")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and convert to RGB
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((32, 32))
        st.image(image_resized, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
        st.markdown(f"### ✅ Confidence: **{confidence:.2f}**")

        # 📊 Class Probability Bar Chart
        st.markdown("### 🔍 Class Probabilities:")
        fig, ax = plt.subplots()
        bars = ax.bar(class_names, predictions[0], color='skyblue')
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        ax.set_xticklabels(class_names, rotation=45)
        for bar, prob in zip(bars, predictions[0]):
            height = bar.get_height()
            ax.annotate(f'{prob:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        st.pyplot(fig)

        # 🧮 Show Raw Prediction Array
        if st.checkbox("Show raw prediction array"):
            st.write(predictions)

    except Exception as e:
        st.error("❌ An error occurred while processing the image. Please try a valid PNG/JPG image.")
        st.exception(e)
