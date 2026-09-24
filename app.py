import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

# Resolve the model relative to this file so the app works regardless of the
# directory from which Streamlit is started.
MODEL_PATH = Path(__file__).resolve().parent / "671010165_mnist_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.is_file():
    st.error(
        f"Model file '{MODEL_PATH.name}' was not found. "
        "Make sure 671010165_mnist_model.keras is included in the repository."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the MNIST model: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")

        # Convert to the same grayscale, 28x28 format used by MNIST.
        image = image.convert("L").resize((28, 28))
        image_array = np.asarray(image, dtype="float32")

        # MNIST digits are white on a black background. Invert photographs or
        # drawings that use the opposite color scheme.
        if image_array.mean() > 127:
            image_array = 255 - image_array

        image_array = image_array / 255.0

        # Support both common Keras MNIST input shapes: (28, 28) and
        # (28, 28, 1).
        if len(model.input_shape) == 4:
            image_array = image_array.reshape(1, 28, 28, 1)
        else:
            image_array = image_array.reshape(1, 28, 28)

        prediction = model.predict(image_array, verbose=0)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        st.success(
            f"The model predicts the digit is: **{predicted_digit}** "
            f"(confidence: {confidence:.1%})"
        )

    except Exception as exc:
        st.error(f"An error occurred during prediction: {exc}")
