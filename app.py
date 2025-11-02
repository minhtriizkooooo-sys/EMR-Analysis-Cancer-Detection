import os
import io
import base64
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from huggingface_hub import hf_hub_download

# --- Logger ---
logging.basicConfig(level=logging.INFO)

# --- Flask app ---
app = Flask(__name__)
app.secret_key = "supersecretkey"

# --- Upload folder ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model settings ---
MODEL_REPO = "your_hf_username/your_model_repo"  # Thay bằng HF repo của bạn
MODEL_FILE = "best_weights_model.keras"
MODEL_PATH = os.path.join("models", MODEL_FILE)
os.makedirs("models", exist_ok=True)

model = None

def download_model_from_hf():
    """Tải model từ Hugging Face nếu chưa có"""
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model from Hugging Face...")
        path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        os.rename(path, MODEL_PATH)
        logging.info(f"Model downloaded to {MODEL_PATH}")

def load_trained_model():
    global model
    if model is None:
        download_model_from_hf()
        logging.info("Loading Keras model...")
        model = load_model(MODEL_PATH)
        logging.info("Model loaded successfully")
    return model

def preprocess_image(x):
    """Giống logic Colab: resize, normalize"""
    x = x / 255.0
    return x

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username and password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        flash("Sai thông tin đăng nhập!", "danger")
    return render_template("index.html")

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

                # Đọc CSV/Excel
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)

                # Summary đẹp
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
    app.run(debug=True)
