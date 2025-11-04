import os
import io
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adamax
from functools import wraps
import requests
import logging

# --- Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s INFO:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Flask config ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_strong_secret_key_12345')

# --- Cấu hình thư mục ---
UPLOAD_DIR = Path('/tmp/uploads')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# --- Giới hạn và cấu hình model ---
MAX_FILE_SIZE_MB = 10
MAX_ROWS_FOR_PROFILE = 5000
MIN_MODEL_SIZE_MB = 5

MODEL = None
TARGET_SIZE = (240, 240)
MODEL_FILENAME = "best_weights_model.keras"
MODEL_DIR = Path('/models')
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
HF_MODEL_URL = (
    "https://huggingface.co/spaces/minhtriizkooooo/"
    "EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"
)

# --- Tải model ---
def load_keras_model():
    """Tự động tải model từ Hugging Face nếu chưa có và load vào bộ nhớ."""
    global MODEL
    if MODEL is not None:
        return MODEL

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    min_bytes = MIN_MODEL_SIZE_MB * 1024 * 1024

    if not MODEL_PATH.exists():
        try:
            logger.info("📥 Model chưa tồn tại. Đang tải từ Hugging Face...")
            with requests.get(HF_MODEL_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("✅ Model đã tải xong: %s", MODEL_PATH)
        except Exception as e:
            logger.error("❌ Không thể tải model từ Hugging Face: %s", e)
            return None

    if MODEL_PATH.stat().st_size < min_bytes:
        logger.error("❌ Model file quá nhỏ (%s bytes).", MODEL_PATH.stat().st_size)
        return None

    try:
        logger.info("🔥 Đang load model từ: %s", MODEL_PATH)
        MODEL = load_model(str(MODEL_PATH), compile=False, custom_objects={'Adamax': Adamax})
        logger.info("✅ Model load thành công và sẵn sàng dự đoán.")
    except Exception as e:
        logger.error("❌ Lỗi khi load model: %s", e)
        MODEL = None

    return MODEL

# --- Tải model khi app khởi động ---
with app.app_context():
    load_keras_model()

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Vui lòng đăng nhập để truy cập trang này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def preprocess_image_match_training(image_file):
    """Preprocessing matched to Colab training (240x240 RGB)."""
    if not MODEL:
        raise RuntimeError("Model is not loaded.")
    img = load_img(image_file, target_size=TARGET_SIZE, color_mode='rgb')
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    userID = request.form.get("userID", "").strip()
    password = request.form.get("password", "").strip()
    if userID == "user_demo" and password == "Test@123456":
        session['user'] = userID
        return redirect(url_for("dashboard"))
    flash("Sai ID hoặc mật khẩu.", "danger")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    model_status = "✅ Model loaded" if MODEL is not None else "⚠️ Model not loaded"
    return render_template("dashboard.html", model_status=model_status)

@app.route("/emr_profile", methods=["GET", "POST"])
@login_required
def emr_profile():
    summary = None
    filename = None
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("Không có file nào được tải lên.", "danger")
            return render_template('emr_profile.html', summary=None, filename=None)

        filename = file.filename
        try:
            file_stream = io.BytesIO(file.read())
            if len(file_stream.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError("File quá lớn.")

            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_stream)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_stream)
            else:
                flash("Chỉ hỗ trợ file CSV hoặc Excel.", "danger")
                return render_template('emr_profile.html')

            rows, cols = df.shape
            summary = f"<p><strong>Số dòng:</strong> {rows}</p><p><strong>Số cột:</strong> {cols}</p>"
            summary += df.head().to_html(classes="table table-bordered", index=False)
        except Exception as e:
            summary = f"<p class='text-danger'>Lỗi xử lý file: {e}</p>"

    return render_template('emr_profile.html', summary=summary, filename=filename)

@app.route('/emr_prediction', methods=['GET','POST'])
@login_required
def emr_prediction():
    prediction_result = None
    filename = None
    image_b64 = None
    if MODEL is None:
        flash('Hệ thống AI chưa sẵn sàng. Vui lòng thử lại sau.', 'danger')
        return render_template('emr_prediction.html')
    if request.method == 'POST':
        uploaded = request.files.get('file')
        if not uploaded or uploaded.filename == '':
            flash('Vui lòng chọn file hình ảnh.', 'danger')
            return redirect(request.url)
        if not allowed_file(uploaded.filename):
            flash('Định dạng file không hợp lệ.', 'danger')
            return redirect(request.url)
        filename = secure_filename(uploaded.filename)
        data = uploaded.read()
        image_b64 = base64.b64encode(data).decode('utf-8')
        image_stream = io.BytesIO(data)
        image_stream.seek(0)
        try:
            processed = preprocess_image_match_training(image_stream)
            preds = MODEL.predict(processed)
            p_nodule = float(preds[0][0]) if preds.ndim == 2 and preds.shape[1] == 1 else float(np.max(preds[0]))
            label = '🩸 Nodule (Có dấu hiệu ung thư)' if p_nodule >= 0.5 else '🌿 Non-Nodule (Bình thường)'
            prediction_result = {'result': label, 'probability': round(float(p_nodule), 5)}
        except Exception as e:
            logger.error("Prediction error: %s", e)
            flash(f'Lỗi khi xử lý ảnh: {e}', 'danger')
            return redirect(request.url)
    return render_template('emr_prediction.html', prediction=prediction_result, filename=filename, image_b64=image_b64)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# --- Run app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render tự cấp PORT
    logger.info(f"🚀 Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
