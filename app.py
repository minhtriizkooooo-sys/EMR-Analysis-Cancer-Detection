import os
import io
import base64
import logging
import gc
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
            logging.info("✅ Model downloaded successfully.")
        else:
            logging.error(f"❌ Failed to download model. HTTP {response.status_code}")
            raise RuntimeError("Cannot download model from Hugging Face.")

# --- Load Keras model ---
def load_trained_model():
    download_model()
    logging.info("Loading model...")
    model = load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully.")
    return model

# --- Lazy loading for performance ---
model = None

# ================= ROUTES =================

# --- Home (Login Page) ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --- Login Route ---
@app.route("/login", methods=["POST"])
def login():
    userID = request.form.get("userID")
    password = request.form.get("password")

    # Demo credentials
    if userID == "user_demo" and password == "Test@123456":
        session["username"] = userID
        return redirect(url_for("dashboard"))
    else:
        return redirect(url_for("index"))

# --- Dashboard ---
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")

# --- EMR Profile Page (GET) ---
@app.route("/profile_page")
def emr_profile():
    if "username" not in session:
        return redirect(url_for("index"))
    # Render empty profile page first
    return render_template("EMR_Profile.html", tables=[], titles=[])

# --- EMR Profiling (POST) ---
@app.route("/profile", methods=["POST"])
def profile():
    if "username" not in session:
        return redirect(url_for("index"))

    try:
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            return redirect(url_for("emr_profile"))

        df = pd.read_csv(file)
        summary = df.describe(include="all").T
        summary_html = summary.to_html(
            classes="table table-striped text-sm text-center border-collapse",
            border=0
        )

        return render_template("EMR_Profile.html", tables=[summary_html], titles=["EMR Profiling Summary"])
    except Exception as e:
        logging.error(f"Profile error: {e}")
        return redirect(url_for("emr_profile"))

# --- Image Prediction Page (GET) ---
@app.route("/prediction_page")
def emr_prediction():
    if "username" not in session:
        return redirect(url_for("index"))
    return render_template("EMR_Prediction.html", image_data=None, prediction=None, confidence=None)

# --- Prediction (POST) ---
@app.route("/predict", methods=["POST"])
def predict():
    global model
    if "username" not in session:
        return redirect(url_for("index"))

    try:
        file = request.files.get("image_file")
        if not file or file.filename == "":
            return redirect(url_for("emr_prediction"))

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        # --- Preprocess image ---
        img = load_img(filepath, target_size=(240, 240))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # --- Lazy load model ---
        if model is None:
            model = load_trained_model()

        # --- Prediction ---
        preds = model.predict(img_array)
        prob = float(preds[0][0])
        pred_class = "Nodule" if prob >= 0.5 else "Non-Nodule"
        confidence = prob * 100 if prob >= 0.5 else (100 - prob * 100)

        # --- Encode image for display ---
        with open(filepath, "rb") as img_f:
            img_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        # --- Clean memory ---
        gc.collect()

        return render_template(
            "EMR_Prediction.html",
            image_data=img_base64,
            prediction=pred_class,
            confidence=f"{confidence:.2f}%"
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return redirect(url_for("emr_prediction"))

# --- Logout ---
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# --- Run (Render-compatible) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
