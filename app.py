import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App ---
app = Flask(__name__)
app.secret_key = "secret123"

# --- Directories ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Hugging Face model source ---
HF_MODEL_URL = "https://minhtriizkooooo-emr-analysis-cancer-detection.hf.space/file/models/best_weights_model.keras"
MODEL_PATH = os.path.join("models", "best_weights_model.keras")

# --- Ensure model exists locally ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model from Hugging Face Space...")
        response = requests.get(HF_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info("Model downloaded successfully.")
        else:
            logging.error(f"Failed to download model. HTTP {response.status_code}")
            raise RuntimeError("Cannot download model from Hugging Face.")

# --- Load Keras model ---
def load_trained_model():
    download_model()
    logging.info("Loading model...")
    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
    return model

# --- Lazy load to speed up startup ---
model = None

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# --- EMR Profiling ---
@app.route("/profile", methods=["POST"])
def profile():
    try:
        if "csv_file" not in request.files:
            flash("No file uploaded.", "error")
            return redirect(url_for("dashboard"))

        file = request.files["csv_file"]
        if file.filename == "":
            flash("Please select a CSV file.", "error")
            return redirect(url_for("dashboard"))

        df = pd.read_csv(file)
        summary = df.describe(include="all").T
        summary_html = summary.to_html(classes="table table-striped", border=0)

        return render_template("EMR_Profile.html", tables=[summary_html], titles=["EMR Profiling Summary"])
    except Exception as e:
        logging.error(f"Profile error: {e}")
        flash(f"Error during profiling: {str(e)}", "error")
        return redirect(url_for("dashboard"))

# --- Prediction ---
@app.route("/predict", methods=["POST"])
def predict():
    global model
    try:
        if "image_file" not in request.files:
            flash("No image uploaded.", "error")
            return redirect(url_for("dashboard"))

        file = request.files["image_file"]
        if file.filename == "":
            flash("Please select an image.", "error")
            return redirect(url_for("dashboard"))

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        # Load and preprocess image
        img = load_img(filepath, target_size=(240, 240))  # Resize theo Colab
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Load model lazily
        if model is None:
            model = load_trained_model()

        # Predict
        preds = model.predict(img_array)
        pred_class = "Nodule" if preds[0][0] >= 0.5 else "Non-Nodule"
        confidence = float(preds[0][0]) * 100 if preds[0][0] >= 0.5 else (100 - float(preds[0][0]) * 100)

        # Encode image for HTML
        with open(filepath, "rb") as img_f:
            img_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        return render_template(
            "EMR_Prediction.html",
            image_data=img_base64,
            prediction=pred_class,
            confidence=f"{confidence:.2f}%"
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        flash(f"Error during prediction: {str(e)}", "error")
        return redirect(url_for("dashboard"))

# --- Run (Render-compatible) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port)
