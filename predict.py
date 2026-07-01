"""
predict.py
Loads the saved MNIST model and predicts a single image.

Workflow:
Image -> Resize (28x28) -> Grayscale -> Normalize -> Flatten -> Load Model -> Predict -> Return Digit
"""

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "model/mnist_model.keras"


def load_trained_model(model_path: str = MODEL_PATH):
    """Load the trained Keras model from disk."""
    model = load_model(model_path)
    return model


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw image array for prediction.

    Steps:
    1. Convert to grayscale (handles RGB and RGBA canvas input)
    2. Resize to 28x28
    3. Normalize pixel values to 0-1
    4. Flatten to shape (1, 784)
    """
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    image_flat = image.reshape(1, 28 * 28)

    return image_flat


def predict_digit(image_path: str, model=None):
    """
    Predict the digit in a given image file.

    Args:
        image_path: path to the image file (png/jpg)
        model: pre-loaded Keras model (optional, loads fresh if None)

    Returns:
        predicted_digit (int), confidence (float, 0-100), probabilities (np.ndarray)
    """
    if model is None:
        model = load_trained_model()

    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)

    processed = preprocess_image(image_array)

    probabilities = model.predict(processed, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities) * 100)

    return predicted_digit, confidence, probabilities


def predict_from_array(image_array: np.ndarray, model=None):
    """
    Predict the digit directly from a numpy array (used by app.py
    for canvas-drawn images, avoiding a round-trip through disk).

    Args:
        image_array: numpy array of the image (grayscale, RGB, or RGBA)
        model: pre-loaded Keras model (optional)

    Returns:
        predicted_digit (int), confidence (float, 0-100), probabilities (np.ndarray)
    """
    if model is None:
        model = load_trained_model()

    processed = preprocess_image(image_array)

    probabilities = model.predict(processed, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities) * 100)

    return predicted_digit, confidence, probabilities


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    digit, conf, probs = predict_digit(img_path)

    print(f"Predicted Digit : {digit}")
    print(f"Confidence      : {conf:.2f}%")
