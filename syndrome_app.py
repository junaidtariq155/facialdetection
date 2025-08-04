import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

st.set_page_config(page_title="Syndrome Detection", layout="wide")
st.title("👁️ Syndrome Detection via Facial Recognition")

model = load_model("syndrome_model.h5")  # Replace with your trained model
classes = ['Down Syndrome', 'Normal', 'Williams Syndrome']  # Adjust based on your dataset

uploaded = st.file_uploader("Upload a facial image", type=['jpg', 'png'])

if uploaded:
    img = image.load_img(uploaded, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Input Image", use_column_width=False)

    pred = model.predict(img_array)[0]
    top_index = np.argmax(pred)
    st.success(f"Predicted Syndrome: **{classes[top_index]}**")
    st.text("Class probabilities:")
    for i, p in enumerate(pred):
        st.write(f"{classes[i]}: {p:.2f}")
