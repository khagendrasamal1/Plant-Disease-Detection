import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model
model = load_model('potatoes.keras')

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


# Prediction function
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


# Streamlit app
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #4CAF50, #2196F3); /* Gradient colors */
        color: white; /* Optional: Change text color to make it readable */
        height: 100vh; /* Ensure body covers the full height of the viewport */
        margin: 0; /* Remove default margin */
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .animated-header {
        text-align: center;
        font-size: 3em;
        color: #4CAF50;
        animation: fadeIn 2s;
    }
    .stImage {
        border-radius: 10px; /* Optional: rounded corners for images */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='animated-header'>Image Classification Web App</div>", unsafe_allow_html=True)
st.write("Upload an image to classify")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    st.write("Classifying...")
    img = img.resize((224, 224))
    predicted_class, confidence = predict(model, img)

    # Display results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence}%")
