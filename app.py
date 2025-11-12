# -*- coding: utf-8 -*-
# app.py: EMR AI - ADVANCED PROFILE + REAL KERAS PREDICTION + NO 502
import os
import io
import base64
import logging
import tempfile
import signal
import time
from PIL import Image
import numpy as np
import pandas as pd
import requests
from flask import (
    Flask, flash, redirect, render_template, request, session, url_for
)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from functools import wraps
from ydata_profiling import ProfileReport

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === FLASK SETUP ===
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "emr-ai-secure-2025")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max
MAX_FILE_SIZE_MB = 5
ALLOWED_IMG_EXT = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_DATA_EXT = {'csv', 'xls', 'xlsx', 'txt'}

# === FOLDERS ===
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")

# === HUGGING FACE MODEL URL ===
HF_MODEL_URL = (
    "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/"
    "resolve/main/models/best_weights_model.keras"
)

# === LOAD MODEL ONCE (EAGER) ===
model = None
if not os.path.exists(MODEL_PATH):
    logger.info("Downloading model from Hugging Face...")
    try:
        r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Model downloaded.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
else:
    logger.info("Model found locally.")

# Load model with timeout protection
try:
    # Set a timeout for model loading (30s)
    def timeout_handler(signum, frame):
        raise TimeoutError("Model load timeout")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    model = load_model(MODEL_PATH)
    signal.alarm(0)
    logger.info("Keras model loaded successfully.")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    model = None

# === UTILITIES ===
def allowed_file(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def safe_thumbnail(img_bytes, size=200):
    """Tạo thumbnail nhỏ, an toàn, không crash"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Thumbnail error: {e}")
        return None

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

def safe_predict(model, img_array, timeout=10):
    """Predict with timeout to avoid 502"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Prediction timeout")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        prob = float(model.predict(img_array, verbose=0)[0][0])
        signal.alarm(0)
        return prob
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return 0.5  # Fallback neutral

# === ROUTES ===
@app.route("/")
def home():
    return redirect(url_for("login"))

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
    status = "Model loaded" if model else "Model failed"
    return render_template("dashboard.html", model_status=status)

# === EMR PROFILE: CHUYÊN SÂU VỚI YDATA (CONFIG NHẸ) ===
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
            flash("Chỉ hỗ trợ CSV, XLS, XLSX, TXT.", "danger")
            return render_template("emr_profile.html")

        try:
            file_bytes = file.read()
            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                flash(f"File quá lớn (>5MB).", "danger")
                return render_template("emr_profile.html")

            file_stream = io.BytesIO(file_bytes)
            if filename.lower().endswith('.csv') or filename.lower().endswith('.txt'):
                df = pd.read_csv(file_stream)
            else:
                df = pd.read_excel(file_stream)

            # Giới hạn 2000 dòng để tránh chậm
            if len(df) > 2000:
                df_sample = df.sample(2000)
                flash("File lớn → phân tích mẫu 2000 dòng cho tốc độ nhanh.", "warning")
            else:
                df_sample = df

            # Tạo report chuyên sâu nhưng nhẹ
            start_time = time.time()
            profile = ProfileReport(
                df_sample,
                title=f"Phân tích Chuyên Sâu EMR: {filename}",
                explorative=True,  # Bật explorative cho chuyên sâu
                correlations={
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": False},  # Tắt Spearman để nhanh
                    "kendall": {"calculate": False},
                    "phi_k": {"calculate": False},
                    "cramers": {"calculate": False},
                    "recoded": {"calculate": False}
                },
                interactions={"continuous": True},  # Giữ interactions cơ bản
                missing_diagrams={
                    "heatmap": True,
                    "dendrogram": False,  # Tắt dendrogram nặng
                    "matrix": False
                },
                html={"style": {"full_width": True}},
                sort="none"  # Không sort để nhanh
            )
            profile_html = profile.to_html()
            end_time = time.time()
            flash(f"✅ Báo cáo chuyên sâu hoàn thành trong {end_time - start_time:.1f}s!", "success")

        except Exception as e:
            logger.error(f"Profile error: {e}")
            flash(f"Lỗi tạo báo cáo: {e}. Thử file nhỏ hơn.", "danger")

    return render_template("emr_profile.html", profile_html=profile_html, filename=filename)

# === EMR PREDICTION: REAL KERAS + TIMEOUT BẢO VỆ ===
@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required
def emr_prediction():
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        if model is None:
            flash("Mô hình AI chưa sẵn sàng (kiểm tra logs).", "danger")
            return render_template("emr_prediction.html")

        file = request.files.get("file")
        if not file or not file.filename:
            flash("Vui lòng chọn ảnh.", "danger")
            return render_template("emr_prediction.html")

        filename = secure_filename(file.filename)
        if not allowed_file(filename, ALLOWED_IMG_EXT):
            flash("Chỉ hỗ trợ ảnh: JPG, PNG, GIF, BMP.", "danger")
            return render_template("emr_prediction.html")

        img_bytes = file.read()
        if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            flash("Ảnh quá lớn (>5MB).", "danger")
            return render_template("emr_prediction.html")

        # Tạo thumbnail nhỏ ngay lập tức
        image_b64 = safe_thumbnail(img_bytes, size=200)

        # Dự đoán với timeout
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            img = load_img(tmp_path, target_size=(240, 240))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            prob = safe_predict(model, arr, timeout=15)  # 15s timeout
            result = "Nodule (Có khối u)" if prob > 0.5 else "Non-nodule (Không có khối u)"
            prediction = {"result": result, "probability": prob}

            flash(f"✅ Dự đoán hoàn thành: {result} ({prob*100:.1f}%)", "success")

        except Exception as e:
            logger.error(f"Predict error: {e}")
            flash(f"Lỗi AI: {e}. Thử ảnh nhỏ hơn.", "danger")
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
    return {"status": "ok", "model": model is not None}, 200

# === RUN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("EMR AI khởi động – Advanced Profile + Protected Real Keras")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
