import streamlit as st
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

model_path = 'keras_model.h5'

# Try to load the model and handle exceptions
try:
    model = load_model("keras_model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the labels
class_names = open("labels.txt", "r").readlines()


def load_and_predict(image, model_path):
    # Load the Keras model
    model = load_model(model_path, compile=False)

    # Prepare the image
    size = (224, 224)  # assuming the model expects this size, adjust if necessary
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Make prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)

    return prediction

def main():
    st.title("Image Classification App")
    st.write("Upload an image and the model will predict the class.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        st.write("Classifying...")
        class_name, confidence = predict(image)
        st.write("Class:", class_name)
        st.write("Confidence Score:", confidence)

if __name__ == "__main__":
    main()
