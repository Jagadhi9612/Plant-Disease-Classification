# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import feature
import os

# --- Constants ---
MODEL_PATH = "plantvillage_results/full_model.h5"
ENCODER_PATH = "plantvillage_results/label_encoder.pkl"
IMAGE_SIZE = (128, 128)
ULBP_POINTS = 24
ULBP_RADIUS = 8

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --- Load Label Encoder ---
@st.cache_resource
def load_label_encoder():
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return le

# --- ULBP Feature Extraction ---
def extract_ulbp_features(image_array, num_points=ULBP_POINTS, radius=ULBP_RADIUS, eps=1e-7):
    gray = np.mean(image_array, axis=2)
    gray = (gray * 255).astype("uint8")
    ulbp = feature.local_binary_pattern(gray, num_points, radius, method="default")
    hist, _ = np.histogram(ulbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

# --- Prediction Function ---
def predict(image, model, label_encoder):
    image = image.resize(IMAGE_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    ulbp_features = extract_ulbp_features(image_array[0])
    ulbp_features = np.expand_dims(ulbp_features, axis=0)

    prediction = model.predict([image_array, ulbp_features])
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = np.max(prediction)
    
    return predicted_class_label, confidence

# --- Streamlit GUI ---
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Detection (CNN + ULBP)")
st.markdown("Upload a plant leaf image to identify the disease class.")

# Load model and encoder
model = load_model()
label_encoder = load_label_encoder()

# Upload image
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width =True)
    
    # Predict
    predicted_label, confidence = predict(image, model, label_encoder)
    
    st.success(f"**Predicted Disease:** `{predicted_label}`")
    st.info(f"**Confidence:** `{confidence * 100:.2f}%`")
