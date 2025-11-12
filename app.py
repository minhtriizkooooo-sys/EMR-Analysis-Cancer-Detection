# -*- coding: utf-8 -*-
# app.py: EMR AI - REAL KERAS + ADVANCED PROFILE + NO 502/CRASH
import os
import io
import base64
import logging
import tempfile
import numpy as np
import pandas as pd
import requests
from flask import (
    Flask, flash, redirect, render_template, request, session, url_for
)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from functools import wraps
from ydata_profiling import ProfileReport

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === FLASK SETUP ===
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "emr-secure-2025")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
MAX_FILE_SIZE_MB = 5
ALLOWED_IMG_EXT = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_DATA_EXT = {'csv', 'xls', 'xlsx'}

# === MODEL PATH ===
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")
HF_MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"

# === LOAD MODEL ONCE ===
model = None
if not os.path.exists(MODEL_PATH):
    logger.info("Downloading model from HF...")
    try:
        r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        logger.info("Model downloaded.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
else:
    logger.info("Model found locally.")

try:
    model = load_model(MODEL_PATH)
    logger.info("REAL KERAS MODEL LOADED SUCCESSFULLY")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    model = None

# === UTILS ===
def allowed_file(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def safe_thumbnail(img_bytes, size=200):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'user' not in session: return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

# === ROUTES ===
@app.route("/")
def home(): return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("userID") == "user_demo" and request.form.get("password") == "Test@123456":
            session['user'] = "user_demo"
            return redirect(url_for("dashboard"))
        flash("Sai ID hoặc mật khẩu.", "danger")
    return render_template("index.html")

@app.route("/dashboard")
@login_required
def dashboard():
    status = "Model Real Keras" if model else "Model Failed"
    return render_template("dashboard.html", model_status=status)

# === EMR PROFILE: CHUYÊN SÂU, KHÔNG LỖI ===
@app.route("/emr_profile", methods=["GET", "POST"])
@login_required
def emr_profile():
    profile_html = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Vui lòng chọn file.", "danger")
            return render_template("emr_profile.html")

        filename = secure_filename(file.filename)
        if not allowed_file(filename, ALLOWED_DATA_EXT):
            flash("Chỉ hỗ trợ CSV, XLS, XLSX.", "danger")
            return render_template("emr_profile.html")

        try:
            file_bytes = file.read()
            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                flash("File >5MB.", "danger")
                return render_template("emr_profile.html")

            stream = io.BytesIO(file_bytes)
            df = pd.read_csv(stream) if filename.lower().endswith('.csv') else pd.read_excel(stream)

            # Sample nếu lớn
            if len(df) > 1500:
                df = df.sample(1500, random_state=42)
                flash("File lớn → phân tích mẫu 1500 dòng.", "warning")

            # ProfileReport CHUYÊN SÂU + SỬA LỖI SORT
            profile = ProfileReport(
                df,
                title=f"Phân tích EMR: {filename}",
                explorative=True,
                correlations={
                    "auto": True,
                    "pearson": True,
                    "spearman": False,
                    "kendall": False
                },
                interactions={"continuous": True},
                missing_diagrams={"heatmap": True, "dendrogram": False},
                duplicaterows=True,
                sortby=None,  # SỬA LỖI: sortby=None (không phải sort)
                html={"style": {"full_width": True}}
            )
            profile_html = profile.to_html()
            flash("Báo cáo chuyên sâu hoàn thành!", "success")

        except Exception as e:
            logger.error(f"Profile error: {e}")
            flash(f"Lỗi: {str(e)[:100]}. Dùng file nhỏ hơn.", "danger")

    return render_template("emr_profile.html", profile_html=profile_html, filename=filename)

# === EMR PREDICTION: REAL MODEL + NO TIMEOUT CRASH ===
@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required
def emr_prediction():
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        if model is None:
            flash("Model chưa tải được. Kiểm tra logs.", "danger")
            return render_template("emr_prediction.html")

        file = request.files.get("file")
        if not file or not file.filename:
            flash("Chọn ảnh.", "danger")
            return render_template("emr_prediction.html")

        filename = secure_filename(file.filename)
        if not allowed_file(filename, ALLOWED_IMG_EXT):
            flash("Chỉ hỗ trợ JPG, PNG, BMP.", "danger")
            return render_template("emr_prediction.html")

        img_bytes = file.read()
        if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            flash("Ảnh >5MB.", "danger")
            return render_template("emr_prediction.html")

        # Thumbnail
        image_b64 = safe_thumbnail(img_bytes)

        # Predict REAL MODEL
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            img = load_img(tmp_path, target_size=(240, 240))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            # DỰ ĐOÁN THẬT – KHÔNG TIMEOUT
            prob = float(model.predict(arr, verbose=0)[0][0])
            result = "Nodule (Có khối u)" if prob > 0.5 else "Non-nodule (Không có khối u)"
            prediction = {"result": result, "probability": prob}

            flash(f"AI: {result} ({prob*100:.1f}%)", "success")

        except Exception as e:
            logger.error(f"Predict error: {e}")
            flash(f"Lỗi AI: {e}", "danger")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return render_template(
        "emr_prediction.html",
        prediction=prediction,
        filename=filename,
        image_b64=image_b64
    )

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("login"))

@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}, 200

# === RUN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("EMR AI FINAL – REAL MODEL + ADVANCED PROFILE")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
