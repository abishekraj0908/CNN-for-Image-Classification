import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------
# 1️⃣ Page Config
# -------------------
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")

st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish, and the model will predict its species.")

# -------------------
# 2️⃣ Load Model
# -------------------
@st.cache_resource
def load_vgg16_model():
    model = load_model("fish_classifier_vgg16.h5")
    return model

model = load_vgg16_model()

# -------------------
# 3️⃣ Define Class Names
# -------------------
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# -------------------
# 4️⃣ Image Preprocessing Function
# -------------------
def preprocess_image(img):
    # Resize to model's expected input size
    img = img.resize((224, 224))
    # Convert to array
    img_array = image.img_to_array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values
    img_array = img_array / 255.0
    return img_array

# -------------------
# 5️⃣ File Upload
# -------------------
uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        # Preprocess image
        processed_img = preprocess_image(img)

        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Display results
        st.subheader(f"Prediction: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence*100:.2f}%")
