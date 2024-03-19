import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Path to the Keras model
model_path = 'keras_model.h5'

# Try to load the model and handle exceptions
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r")]

def predict(image, model):
    if model is None:
        return "Model not loaded", 0

    # Prepare the image
    size = (224, 224)  # assuming the model expects this size, adjust if necessary
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Make prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def main():
    st.title("Image Classification App")
    st.write("Upload an image and the model will predict the class.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if model:
            st.write("Classifying...")
            class_name, confidence = predict(image, model)
            st.write("Class:", class_name)
            st.write("Confidence Score:", confidence)
        else:
            st.error("Model is not loaded, cannot classify the image.")

if __name__ == "__main__":
    main()
