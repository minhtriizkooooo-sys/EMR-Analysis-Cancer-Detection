# -*- coding: utf-8 -*-
"""
app.py — EMR AI LITE
→ Phân tích dữ liệu EMR (CSV) bằng pandas
→ Dự đoán hình ảnh y tế bằng mô hình Keras lưu trên HuggingFace
→ Lazy loading model để tránh lỗi 502 / timeout
"""
import os
import io
import base64
import logging
import tempfile
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

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emr-ai")

# === FLASK SETUP ===
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "emr-secure-2025")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

MAX_FILE_SIZE_MB = 10
ALLOWED_IMG_EXT = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_DATA_EXT = {'csv', 'xls', 'xlsx'}

# === MODEL CONFIGURATION ===
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")
HF_MODEL_URL = (
    "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/"
    "resolve/main/models/best_weights_model.keras"
)

# === GLOBAL STATE ===
model = None  # Lazy loaded later

# --------------------------------------------------------
# UTILITIES
# --------------------------------------------------------
def get_model():
    """Lazy load model from HuggingFace just in time."""
    global model
    if model is None:
        logger.info("Loading AI model (lazy mode)...")
        try:
            if not os.path.exists(MODEL_PATH):
                logger.info("Downloading model from HuggingFace...")
                r = requests.get(HF_MODEL_URL, stream=True, timeout=300)
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("✅ Model downloaded successfully.")

            model = load_model(MODEL_PATH)
            logger.info("✅ Keras model loaded into memory.")
        except Exception as e:
            logger.error(f"❌ Failed to load AI model: {e}")
            raise RuntimeError(f"Cannot load model: {e}")
    return model


def allowed_file(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts


def safe_thumbnail(img_bytes, size=200):
    """Generate a small preview image as base64."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.error(f"Thumbnail generation error: {e}")
        return None


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

# --------------------------------------------------------
# ROUTES
# --------------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (
            request.form.get("userID") == "user_demo"
            and request.form.get("password") == "Test@123456"
        ):
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
    status = "Model đã tải" if model else "Model chưa tải (Lazy)"
    return render_template("dashboard.html", model_status=status)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# --------------------------------------------------------
# 1️⃣ EMR FILE ANALYSIS (PANDAS)
# --------------------------------------------------------
@app.route("/emr_profile", methods=["GET", "POST"])
@login_required
def emr_profile():
    filename = None
    summary_html = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Vui lòng chọn file dữ liệu.", "danger")
            return render_template("emr_profile.html")

        filename = secure_filename(file.filename)
        if not allowed_file(filename, ALLOWED_DATA_EXT):
            flash("Chỉ hỗ trợ CSV, XLS, XLSX.", "danger")
            return render_template("emr_profile.html")

        file_bytes = file.read()
        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            flash(f"File quá lớn (> {MAX_FILE_SIZE_MB}MB).", "danger")
            return render_template("emr_profile.html")

        try:
            stream = io.BytesIO(file_bytes)
            if filename.lower().endswith(".csv"):
                df = pd.read_csv(stream, low_memory=False)
            else:
                df = pd.read_excel(stream, engine="openpyxl")

            # Giới hạn dữ liệu lớn
            if len(df) > 5000:
                df_size = len(df)
                df = df.sample(2000, random_state=42)
                flash(
                    f"File có {df_size} dòng. Phân tích mẫu 2000 dòng để tránh timeout.",
                    "warning",
                )

            # Tạo phân tích cơ bản bằng pandas
            summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "describe": df.describe(include="all").to_html(classes="table-auto", border=0),
                "missing": df.isnull().sum().to_frame("Số ô trống").to_html(classes="table-auto", border=0)
            }

            summary_html = f"""
            <div class="space-y-8">
                <h3 class='text-2xl font-semibold text-primary-green'>Tổng quan dữ liệu</h3>
                <p><strong>Kích thước:</strong> {summary['shape'][0]} hàng × {summary['shape'][1]} cột</p>
                <p><strong>Các cột:</strong> {', '.join(summary['columns'])}</p>
                <h4 class='text-xl font-bold mt-4'>Thống kê mô tả</h4>
                {summary['describe']}
                <h4 class='text-xl font-bold mt-4'>Số ô trống theo cột</h4>
                {summary['missing']}
            </div>
            """
            flash("✅ Phân tích dữ liệu hoàn thành!", "success")

        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            flash(f"Lỗi xử lý dữ liệu: {e}", "danger")

    return render_template("emr_profile.html", summary=summary_html, filename=filename)


# --------------------------------------------------------
# 2️⃣ MEDICAL IMAGE PREDICTION (KERAS)
# --------------------------------------------------------
@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required
def emr_prediction():
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        try:
            current_model = get_model()
        except Exception as e:
            flash(f"❌ Không thể tải mô hình: {e}", "danger")
            return render_template("emr_prediction.html")

        file = request.files.get("file")
        if not file or not file.filename:
            flash("Vui lòng chọn hình ảnh.", "danger")
            return render_template("emr_prediction.html")

        filename = secure_filename(file.filename)
        if not allowed_file(filename, ALLOWED_IMG_EXT):
            flash("Chỉ hỗ trợ ảnh JPG, PNG, BMP.", "danger")
            return render_template("emr_prediction.html")

        img_bytes = file.read()
        if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            flash(f"Ảnh quá lớn (> {MAX_FILE_SIZE_MB}MB).", "danger")
            return render_template("emr_prediction.html")

        image_b64 = safe_thumbnail(img_bytes)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            img = load_img(tmp_path, target_size=(240, 240))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            prob = float(current_model.predict(arr, verbose=0)[0][0])
            label = "Nodule" if prob > 0.5 else "Non-nodule"
            prediction = {"result": label, "probability": prob}

            flash(f"AI Dự đoán: {label} ({prob*100:.1f}%)", "success")

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            flash(f"Lỗi xử lý AI: {e}", "danger")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return render_template("emr_prediction.html", prediction=prediction, filename=filename, image_b64=image_b64)


# --------------------------------------------------------
# RUN APP
# --------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 EMR AI is running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
