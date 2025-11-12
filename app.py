import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import requests
from flask import (
    Flask, flash, redirect, render_template, request, session, url_for, jsonify
)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from functools import wraps
from pandas.errors import ParserError

# ==============================================================
# --- LOGGER CONFIGURATION ---
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================
# --- FLASK CONFIGURATION ---
# ==============================================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_strong_secret_key_12345")

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"csv", "xlsx", "xls", "png", "jpg", "jpeg", "gif", "bmp"}

# ==============================================================
# --- MODEL CONFIGURATION ---
# ==============================================================
MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"
MODEL_PATH = "models/best_weights_model.keras"
TARGET_SIZE = (240, 240)
MODEL = None

# --- Ensure model exists ---
os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    try:
        logger.info("⬇️ Downloading Keras model from Hugging Face...")
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        logger.info("✅ Model downloaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to download model from Hugging Face: {e}")

# --- Load model ---
def load_keras_model():
    global MODEL
    if MODEL is None:
        try:
            logger.info(f"🔥 Loading Keras model from {MODEL_PATH} ...")
            MODEL = load_model(MODEL_PATH, compile=False)
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            MODEL = None
    return MODEL

with app.app_context():
    load_keras_model()

# ==============================================================
# --- HELPER FUNCTIONS ---
# ==============================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Vui lòng đăng nhập để truy cập trang này.", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def preprocess_image(image_file):
    """Chuẩn hóa ảnh đầu vào giống với huấn luyện: RGB, 240x240"""
    if not MODEL:
        raise RuntimeError("Model is not loaded.")
    img = load_img(image_file, target_size=TARGET_SIZE, color_mode="rgb")
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0  # Chuẩn hóa nếu model yêu cầu
    return arr

# ==============================================================
# --- ROUTES ---
# ==============================================================

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("userID") == "user_demo" and request.form.get("password") == "Test@123456":
            session["user"] = "user_demo"
            return redirect(url_for("dashboard"))
        flash("Sai ID hoặc mật khẩu.", "danger")
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    status = "✅ Model đã tải" if MODEL else "⚠️ Model chưa tải"
    return render_template("dashboard.html", model_status=status)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

# ==============================================================
# --- EMR PROFILE ANALYSIS (CSV/XLSX) ---
# ==============================================================
@app.route("/emr_profile", methods=["GET", "POST"])
@login_required
def emr_profile():
    summary_html = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Vui lòng chọn file dữ liệu.", "danger")
            return render_template("emr_profile.html")

        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in {"csv", "xls", "xlsx"}:
            flash("Chỉ hỗ trợ CSV, XLS, XLSX.", "danger")
            return render_template("emr_profile.html")

        try:
            data = io.BytesIO(file.read())
            df = pd.read_csv(data) if ext == "csv" else pd.read_excel(data)
            n_rows, n_cols = df.shape
            summary_html = df.describe(include="all").to_html(classes="table-auto")
            flash(f"Phân tích {n_rows} hàng × {n_cols} cột hoàn tất.", "success")
        except Exception as e:
            logger.exception(f"Lỗi đọc file: {e}")
            flash(f"Lỗi đọc dữ liệu: {e}", "danger")

    return render_template("emr_profile.html", summary=summary_html)

# ==============================================================
# --- MEDICAL IMAGE PREDICTION (KERAS MODEL) ---
# ==============================================================
@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required
def emr_prediction():
    prediction_result, filename, image_b64 = None, None, None

    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            flash("Vui lòng chọn file hình ảnh.", "danger")
            return redirect(request.url)
        if not allowed_file(uploaded.filename):
            flash("Định dạng file không được hỗ trợ.", "danger")
            return redirect(request.url)

        filename = secure_filename(uploaded.filename)
        data = uploaded.read()
        image_b64 = base64.b64encode(data).decode("utf-8")
        image_stream = io.BytesIO(data)

        if MODEL is None:
            flash("❌ Model chưa được tải lên. Vui lòng chờ hoặc kiểm tra kết nối.", "danger")
            logger.error("MODEL is None when predicting.")
            return redirect(request.url)

        try:
            processed = preprocess_image(image_stream)
            logger.info(f"🧠 Predicting image: shape={processed.shape}")
            preds = MODEL.predict(processed)
            logger.info(f"Raw model output: {preds}")

            # Đảm bảo đầu ra hợp lệ
            p_nodule = float(preds[0][0]) if preds.ndim == 2 else float(preds[0])
            label = "Nodule" if p_nodule >= 0.5 else "Non-nodule"
            prob = p_nodule if p_nodule >= 0.5 else 1 - p_nodule

            prediction_result = {
                "result": label,
                "probability": round(prob, 6),
                "raw_output": round(p_nodule, 6)
            }
            flash("✅ Dự đoán AI hoàn tất.", "success")

        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            flash(f"Lỗi khi xử lý hình ảnh hoặc dự đoán: {e}", "danger")
            return redirect(request.url)

    return render_template("emr_prediction.html", prediction=prediction_result, filename=filename, image_b64=image_b64)

# ==============================================================
# --- RUN APP ---
# ==============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 EMR AI is running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
