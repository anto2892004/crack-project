from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Build the correct model path
MODEL_PATH = os.path.join(os.getcwd(), "artifacts", "crack_detector_model.h5")

# Load your trained model
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Function to preprocess image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((160, 160))  # match your model input
    img = img.convert("RGB")  # ensure 3 color channels
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    img_array = preprocess_image(image_bytes)

    prediction = model.predict(img_array)[0][0]
    result = "Crack Detected" if prediction > 0.5 else "No Crack"

    return jsonify({
        "prediction": result,
        "confidence": float(prediction)
    })

# Default route
@app.route("/", methods=["GET"])
def home():
    return "🚀 Surface Crack Detection API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)


