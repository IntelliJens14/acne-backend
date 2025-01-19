from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)

# Load your .keras model
model = tf.keras.models.load_model(r"C:\Users\Engr. Jens\Desktop\best_acne_severity_model.keras")
print("Model loaded successfully!")

# Prediction endpoint
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Get the image from the request
        image_file = request.files["image"]
        image = Image.open(image_file).resize((224, 224))  # Resize to model input size
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        severity_level = np.argmax(predictions)  # Example: Classification

        # Map severity levels to labels
        severity_map = {
            0: "Extremely Mild (Level 0)",
            1: "Mild (Level 1)",
            2: "Moderate (Level 2)",
            3: "Severe (Level 3)"
        }

        # Respond with the prediction result
        return jsonify({"severityLevel": severity_map.get(severity_level, "Unknown")})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
