import os
import io
import base64
import logging
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Logger ---
logging.basicConfig(level=logging.INFO)

# --- Flask app ---
app = Flask(__name__)
app.secret_key = "supersecretkey"

# --- Upload folder ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model settings ---
MODEL_DIR = "models"
MODEL_FILE = "best_weights_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
os.makedirs(MODEL_DIR, exist_ok=True)

# HF Space raw URL for the model
HF_MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"

model = None

# --- Download model from HF Space ---
def download_model_from_hf_space():
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model from Hugging Face Space...")
        r = requests.get(HF_MODEL_URL, stream=True)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Model downloaded to {MODEL_PATH}")
        else:
            logging.error(f"Failed to download model, status code: {r.status_code}")
            raise Exception("Cannot download model from HF Space")

# --- Load Keras model ---
def load_trained_model():
    global model
    if model is None:
        download_model_from_hf_space()
        logging.info("Loading Keras model...")
        model = load_model(MODEL_PATH)
        logging.info("Model loaded successfully")
    return model

# --- Preprocess image for prediction ---
def preprocess_image(x):
    x = x / 255.0
    return x

# --- Routes ---

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get(" user_demo)
        password = request.form.get("Test@123456")
        if username and password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        flash("Sai thông tin đăng nhập!", "danger")
    return render_template("index.html")  # form action trong index.html: url_for('login')

# --- Add login route to match index.html form action ---
@app.route("/login", methods=["POST"])
def login():
    return index()

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")

@app.route("/profile_page", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("index"))

    summary_html = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                # Read CSV/Excel
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)

                # Generate summary table
                summary_html = df.describe(include="all").T.to_html(
                    classes="table-auto w-full border-collapse border border-gray-300",
                    border=0, escape=False
                )

            except Exception as e:
                logging.error(f"EMR Profile error: {e}")
                flash("File không hợp lệ hoặc lỗi đọc dữ liệu.", "danger")

    return render_template("EMR_Profile.html", summary=summary_html, filename=filename)

@app.route("/prediction_page", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("index"))

    filename = None
    image_b64 = None
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                # Load model
                model = load_trained_model()

                # Preprocess image
                img = load_img(filepath, target_size=(240, 240))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_image(x)

                # Predict
                prob = float(model.predict(x)[0][0])
                pred_class = "Nodule" if prob >= 0.5 else "Non-nodule"

                # Encode image preview
                with open(filepath, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")

                prediction = {"result": pred_class, "probability": prob}

            except Exception as e:
                logging.error(f"Prediction error: {e}")
                flash("Lỗi dự đoán hình ảnh. Kiểm tra định dạng file.", "danger")

    return render_template("EMR_Prediction.html",
                           prediction=prediction,
                           filename=filename,
                           image_b64=image_b64)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

