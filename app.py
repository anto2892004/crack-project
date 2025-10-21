from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import cv2
from collections import OrderedDict

# ----------------------------
# Flask app initialization
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Model path and lazy loading
# ----------------------------
MODEL_PATH = os.path.join(os.getcwd(), "artifacts", "crack_detector_model.h5")
model = None

PIXEL_TO_MM = 0.1  # adjust according to your camera calibration

def load_model_once():
    """Load the model only once (lazy loading)."""
    global model
    if model is None:
        print(f"Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image_bytes):
    """Convert uploaded image to model-ready array."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((160, 160))  # match your model input
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Crack analysis
# ----------------------------
def analyze_cracks(image_bytes):
    """Analyze crack metrics: length, width, severity, and solution."""
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        length_mm = width_mm = 0.0
        severity_metric = 0.0
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        length_mm = h * PIXEL_TO_MM
        width_mm = w * PIXEL_TO_MM
        severity_metric = length_mm * width_mm

    if severity_metric < 50:
        severity = "Light"
        solution = "Monitor regularly; minor cosmetic repair if needed."
    elif severity_metric < 200:
        severity = "Medium"
        solution = "Repair crack with filler or sealant; monitor structure."
    else:
        severity = "Heavy"
        solution = "Structural assessment required; immediate repair needed."

    return {
        "length": f"{round(length_mm, 2)}mm",
        "width": f"{round(width_mm, 2)}mm",
        "severity": severity,
        "solution": solution
    }

# ----------------------------
# API endpoints
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()  # Ensure model is loaded

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)

        # Model prediction
        prediction_val = model.predict(img_array)[0][0]
        result = "Crack Detected" if prediction_val > 0.5 else "No Crack"

        # Crack analysis
        metrics = analyze_cracks(image_bytes)

        # Create ordered response
        response = OrderedDict()
        response["prediction"] = result
        response["confidence"] = round(float(prediction_val), 3)
        response["length"] = metrics["length"]
        response["width"] = metrics["width"]
        response["severity"] = metrics["severity"]
        response["solution"] = metrics["solution"]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 Surface Crack Detection API is running!"

# ----------------------------
# Main entry point
# ----------------------------
if __name__ == "__main__":
    load_model_once()  # Optional: preload on startup
    app.run(host="0.0.0.0", port=5001, debug=False)
