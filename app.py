# -*- coding: utf-8 -*-
# app.py: EMR AI - REAL KERAS MODEL + LIGHT PROFILE + NO 502/CRASH
import os
import io
import base64
import logging
import tempfile
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
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB max
MAX_FILE_SIZE_MB = 4
ALLOWED_IMG_EXT = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_DATA_EXT = {'csv', 'xls', 'xlsx'}

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
        r = requests.get(HF_MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Model downloaded.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
else:
    logger.info("Model found locally.")

# Load model
try:
    model = load_model(MODEL_PATH)
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

# === EMR PROFILE: NHẸ, NHANH, KHÔNG DÙNG YDATA ===
@app.route("/emr_profile", methods=["GET", "POST"])
@login_required
def emr_profile():
    summary = None
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

        # Đọc file vào bộ nhớ
        try:
            file_bytes = file.read()
            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                flash(f"File quá lớn (>4MB).", "danger")
                return render_template("emr_profile.html")

            file_stream = io.BytesIO(file_bytes)
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_stream)
            else:
                df = pd.read_excel(file_stream)

            # Giới hạn 1000 dòng để tránh chậm
            if len(df) > 1000:
                df = df.sample(1000)
                flash("File lớn → chỉ phân tích mẫu 1000 dòng.", "warning")

            rows, cols = df.shape
            info_html = f"""
            <div class="bg-green-50 p-6 rounded-lg">
                <h3 class="text-2xl font-bold text-green-700 mb-4">Tổng quan</h3>
                <p><strong>Dòng:</strong> {rows} | <strong>Cột:</strong> {cols}</p>
            </div>
            """

            col_details = []
            for col in df.columns:
                miss = df[col].isnull().sum()
                uniq = df[col].nunique()
                dtype = str(df[col].dtype)
                stats = ""
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe()
                    stats = f"Min: {desc['min']:.2f}, Max: {desc['max']:.2f}, Mean: {desc['mean']:.2f}"
                col_details.append(f"""
                <div class="bg-gray-50 p-3 rounded border-l-4 border-green-600">
                    <strong>{col}</strong><br>
                    <small>Kiểu: {dtype} | Thiếu: {miss} | Duy nhất: {uniq}</small>
                    {f'<br><small class="text-gray-600">{stats}</small>'.strip() if stats else ''}
                </div>
                """)

            head_html = df.head(5).to_html(classes="table-auto w-full text-sm", index=False)

            summary = info_html + "<h4 class='mt-6 font-bold'>Cột dữ liệu:</h4>" + \
                      "<div class='grid grid-cols-1 md:grid-cols-2 gap-3'>" + "".join(col_details) + "</div>" + \
                      "<h4 class='mt-6 font-bold'>5 dòng đầu:</h4><div class='overflow-x-auto'>" + head_html + "</div>"

        except Exception as e:
            flash(f"Lỗi xử lý file: {e}", "danger")

    return render_template("emr_profile.html", summary=summary, filename=filename)

# === EMR PREDICTION: REAL KERAS + THUMBNAIL NHỎ ===
@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required
def emr_prediction():
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        if model is None:
            flash("Mô hình AI chưa sẵn sàng.", "danger")
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
            flash("Ảnh quá lớn (>4MB).", "danger")
            return render_template("emr_prediction.html")

        # Tạo thumbnail nhỏ
        image_b64 = safe_thumbnail(img_bytes, size=200)

        # Dự đoán bằng mô hình thật
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            img = load_img(tmp_path, target_size=(240, 240))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            prob = float(model.predict(arr, verbose=0)[0][0])
            result = "Nodule (Có khối u)" if prob > 0.5 else "Non-nodule (Không có khối u)"
            prediction = {"result": result, "probability": prob}

        except Exception as e:
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
    return {"status": "ok", "model": model is not None}, 200

# === RUN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("EMR AI khởi động – Profile nhẹ + Real Keras Model")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
