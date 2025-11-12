# -*- coding: utf-8 -*-
# app.py: EMR AI - FINAL LAZY LOADING AND OPTIMIZATION FOR RENDER STABILITY
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === FLASK SETUP ===
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "emr-secure-2025")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
MAX_FILE_SIZE_MB = 10
ALLOWED_IMG_EXT = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_DATA_EXT = {'csv', 'xls', 'xlsx'}

# === MODEL CONFIGURATION ===
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")
HF_MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"

# Global variable for LAZY LOADING
model = None

def get_model():
    """
    Load the Keras model just-in-time (JIT) and cache it.
    This prevents the application from crashing during startup (Error 502/500).
    """
    global model
    if model is None:
        logger.info("JIT Model loading started...")
        try:
            if not os.path.exists(MODEL_PATH):
                logger.info("Model not found locally. Downloading from HF...")
                # Tăng timeout tải model lên 5 phút (300 giây)
                r = requests.get(HF_MODEL_URL, stream=True, timeout=300) 
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                logger.info("Model downloaded successfully.")
            
            # Tải model vào bộ nhớ
            model = load_model(MODEL_PATH)
            logger.info("✅ Keras Model loaded successfully into memory.")
        except Exception as e:
            logger.error(f"❌ JIT Model load failed: {e}")
            model = None
            raise RuntimeError("Failed to load AI model.") from e
            
    return model

# === UTILS (Không đổi) ===
def allowed_file(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def safe_thumbnail(img_bytes, size=200):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.error(f"Thumbnail generation error: {e}")
        return None

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'user' not in session: return redirect(url_for('login'))
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
    # Kiểm tra trạng thái model để hiển thị, không tải model ở đây.
    status = "Model Đã Sẵn Sàng (Lazy Loaded)" if model else "Model Chưa Tải"
    return render_template("dashboard.html", model_status=status)

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("login"))

@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}, 200

# === EMR PROFILE: TỐI ƯU PANDAS & YDATA (Vẫn cần Gunicorn Timeout cao) ===
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
                flash(f"File quá lớn (> {MAX_FILE_SIZE_MB}MB).", "danger")
                return render_template("emr_profile.html")

            stream = io.BytesIO(file_bytes)
            
            # Đọc dữ liệu với tối ưu hóa
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(stream, low_memory=False) 
            else:
                df = pd.read_excel(stream, engine='openpyxl') 

            # === TỐI ƯU HÓA: Lấy mẫu và minimal=True
            if len(df) > 2000:
                df_size = len(df)
                df = df.sample(2000, random_state=42)
                flash(f"File có {df_size} dòng. Đang phân tích mẫu 2000 dòng để tránh Timeout và Crash.", "warning")

            flash("🕒 Đang tạo báo cáo chuyên sâu. Quá trình này **YÊU CẦU Gunicorn Timeout lớn**.", "info")
            profile = ProfileReport(
                df,
                title=f"Phân tích Dữ liệu EMR: {filename}",
                minimal=True,  # CHẾ ĐỘ NHANH VÀ ỔN ĐỊNH NHẤT
                html={"style": {"full_width": True}}
            )
            profile_html = profile.to_html()
            flash("✅ Báo cáo chuyên sâu hoàn thành!", "success")

        except Exception as e:
            logger.error(f"Profile error: {e}")
            flash(f"❌ Lỗi xử lý dữ liệu: {str(e)}. Vui lòng kiểm tra định dạng file (header, encoding) hoặc dùng file nhỏ hơn.", "danger")

    return render_template("emr_profile.html", profile_html=profile_html, filename=filename)

# === EMR PREDICTION: SỬ DỤNG HÀM get_model() ===
@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required
def emr_prediction():
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        # Gọi hàm tải mô hình (Tải lần đầu nếu chưa có)
        try:
            current_model = get_model()
        except RuntimeError as e:
            flash(f"❌ Lỗi dự đoán: {e}", "danger")
            return render_template("emr_prediction.html")

        file = request.files.get("file")
        if not file or not file.filename:
            flash("Vui lòng chọn ảnh.", "danger")
            return render_template("emr_prediction.html")

        filename = secure_filename(file.filename)
        if not allowed_file(filename, ALLOWED_IMG_EXT):
            flash("Chỉ hỗ trợ JPG, PNG, BMP.", "danger")
            return render_template("emr_prediction.html")

        img_bytes = file.read()
        if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            flash(f"Ảnh >{MAX_FILE_SIZE_MB}MB.", "danger")
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

            # Dự đoán bằng current_model
            prob = float(current_model.predict(arr, verbose=0)[0][0])
            result = "Nodule (Có khối u)" if prob > 0.5 else "Non-nodule (Không có khối u)"
            prediction = {"result": result, "probability": prob}

            flash(f"AI: {result} ({prob*100:.1f}%)", "success")

        except Exception as e:
            logger.error(f"Predict error: {e}")
            flash(f"❌ Lỗi AI: {e}", "danger")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return render_template(
        "emr_prediction.html",
        prediction=prediction,
        filename=filename,
        image_b64=image_b64
    )

# === RUN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"EMR AI starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
