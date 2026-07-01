"""
app.py
Streamlit web app for Handwritten Digit Recognition.

Workflow:
User Draws -> Canvas -> Convert Image -> Resize 28x28 -> Grayscale ->
Normalize -> Flatten -> Load mnist_model.keras -> Predict -> Show Result
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

from predict import preprocess_image

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

MODEL_PATH = "model/mnist_model.keras"


@st.cache_resource
def get_model():
    """Load model once and cache across reruns."""
    return load_model(MODEL_PATH)


model = get_model()

with st.sidebar:
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        This app uses an **Artificial Neural Network (ANN)**
        trained on the **MNIST dataset** to recognize
        handwritten digits (0–9).

        **Model Architecture**
        - Input: 784 (28×28 flattened)
        - Dense 128 → ReLU
        - Dense 64 → ReLU
        - Dense 10 → Softmax
        """
    )

    st.divider()

    st.subheader("📝 Instructions")
    st.markdown(
        """
        1. Draw a single digit (0–9) in the canvas
        2. Click **Predict**
        3. View the predicted digit and confidence
        4. Click **Clear** to try again
        """
    )

    st.divider()

    st.subheader("🛠️ Built With")
    st.markdown(
        """
        - TensorFlow / Keras
        - Streamlit
        - OpenCV
        - NumPy
        """
    )

    st.divider()
    st.caption("Made by Md Salek Miah")

st.title("✍️ Handwritten Digit Recognition")
st.markdown(
    "Draw a digit (0–9) below and let the neural network predict it in real time."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎨 Draw Here")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    col_a, col_b = st.columns(2)
    predict_clicked = col_a.button("🔮 Predict", use_container_width=True)
    clear_clicked = col_b.button("🧹 Clear", use_container_width=True)

    if clear_clicked:
        st.rerun()

with col2:
    st.subheader("📊 Prediction Result")

    if predict_clicked:
        if canvas_result.image_data is not None:
            image_array = canvas_result.image_data.astype("uint8")

            if image_array[:, :, :3].sum() == 0:
                st.warning("⚠️ Please draw a digit before predicting.")
            else:
                processed = preprocess_image(image_array)
                probabilities = model.predict(processed, verbose=0)[0]
                predicted_digit = int(np.argmax(probabilities))
                confidence = float(np.max(probabilities) * 100)

                st.markdown(
                    f"""
                    <div style="
                        background-color:#0d1117;
                        padding:20px;
                        border-radius:12px;
                        text-align:center;
                        border: 2px solid #00d4ff;
                    ">
                        <h1 style="color:#00d4ff; font-size:60px; margin:0;">{predicted_digit}</h1>
                        <p style="color:#e5e7eb; font-size:18px;">Confidence: {confidence:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("")

                st.markdown("**Probability Distribution**")
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.bar(range(10), probabilities * 100, color="#1A5276")
                bars[predicted_digit].set_color("#00d4ff")
                ax.set_xticks(range(10))
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability (%)")
                ax.set_title("Prediction Confidence per Digit")
                st.pyplot(fig)
        else:
            st.warning("⚠️ Please draw a digit before predicting.")
    else:
        st.info("👈 Draw a digit and click **Predict** to see results here.")

st.divider()
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:13px;">
        Built with TensorFlow, Keras & Streamlit · MNIST Handwritten Digit Recognition<br>
        © 2026 Md Salek Miah
    </div>
    """,
    unsafe_allow_html=True
)
