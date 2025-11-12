import os
import io
import secrets
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
from pandas.errors import ParserError # Import cụ thể lỗi ParserError


 --- Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s INFO:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Flask config ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_strong_secret_key_12345')

# Use container-safe temp folder
UPLOAD_FOLDER = '/tmp/uploads'
# Đảm bảo thư mục được tạo với quyền tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# --- Model config ---
MODEL_PATH = 'models/best_weights_model.keras'
MODEL = None
TARGET_SIZE = (240, 240)

def load_keras_model():
    """Load model once at startup."""
    global MODEL
    if MODEL is None:
        try:
            logger.info("🔥 Loading Keras model from %s ...", MODEL_PATH)
            MODEL = load_model(MODEL_PATH, compile=False)
            logger.info("✅ Model loaded.")
        except Exception as e:
            logger.error("❌ Error loading model: %s", e)
            MODEL = None
    return MODEL

with app.app_context():
    load_keras_model()

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Vui lòng đăng nhập để truy cập trang này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def preprocess_image(image_file):
    """Preprocessing matched to Colab training (240x240 RGB, no rescale)."""
    if not MODEL:
        raise RuntimeError("Model is not loaded.")
    img = load_img(image_file, target_size=TARGET_SIZE, color_mode='rgb')
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # Tùy chỉnh: Nếu mô hình yêu cầu chuẩn hóa 0-1, thêm dòng này:
    # arr = arr / 255.0
    return arr



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
# 1️⃣ EMR FILE ANALYSIS (PANDAS - CHUYÊN SÂU)
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

            # Giới hạn dữ liệu quá lớn
            if len(df) > 5000:
                df_size = len(df)
                df = df.sample(2000, random_state=42)
                flash(f"File có {df_size} dòng, phân tích mẫu 2000 dòng.", "warning")

            # --- Phân tích chuyên sâu ---
            n_rows, n_cols = df.shape
            missing_ratio = df.isnull().mean().mean()

            # Kiểu dữ liệu
            dtype_counts = df.dtypes.value_counts().to_frame("Số lượng").to_html(classes="table-auto")

            # Cột số - Thống kê
            numeric_df = df.select_dtypes(include=[np.number])
            numeric_summary = numeric_df.describe().T
            numeric_summary["missing_%"] = df[numeric_df.columns].isnull().mean() * 100
            numeric_html = numeric_summary.to_html(classes="table-auto", float_format="%.2f")

            # Cột phân loại - Top giá trị
            categorical_df = df.select_dtypes(exclude=[np.number])
            cat_summary = []
            for col in categorical_df.columns:
                top_vals = categorical_df[col].value_counts().head(5)
                cat_summary.append(
                    f"<b>{col}</b>: {len(categorical_df[col].unique())} giá trị duy nhất<br>{top_vals.to_frame().to_html(classes='table-auto', border=0)}"
                )
            cat_html = "<hr>".join(cat_summary) if cat_summary else "<p>Không có cột phân loại.</p>"

            # Cột có nhiều giá trị thiếu
            missing_table = df.isnull().sum()
            missing_table = missing_table[missing_table > 0].sort_values(ascending=False)
            missing_html = (
                missing_table.to_frame("Số ô trống").to_html(classes="table-auto")
                if not missing_table.empty
                else "<p>Không có dữ liệu bị thiếu.</p>"
            )

            # --- Tổng hợp ra HTML ---
            summary_html = f"""
            <div class="space-y-6">
                <h3 class='text-2xl font-semibold text-primary-green'>Tổng quan dữ liệu</h3>
                <p><strong>Kích thước:</strong> {n_rows} hàng × {n_cols} cột</p>
                <p><strong>Tỷ lệ ô trống trung bình:</strong> {missing_ratio*100:.2f}%</p>
                <h4 class='text-xl font-bold mt-4'>Phân bố kiểu dữ liệu</h4>
                {dtype_counts}
                <h4 class='text-xl font-bold mt-4'>Thống kê dữ liệu số</h4>
                {numeric_html}
                <h4 class='text-xl font-bold mt-4'>Cột có nhiều ô trống</h4>
                {missing_html}
                <h4 class='text-xl font-bold mt-4'>Phân tích dữ liệu phân loại</h4>
                {cat_html}
            </div>
            """

        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            flash(f"Lỗi xử lý dữ liệu: {e}", "danger")

    return render_template("emr_profile.html", summary=summary_html, filename=filename)


# --------------------------------------------------------
# 2️⃣ MEDICAL IMAGE PREDICTION (KERAS)
# --------------------------------------------------------
app.route('/emr_prediction', methods=['GET','POST'])
@login_required
def emr_prediction():
    prediction_result, filename, image_b64 = None, None, None
    if request.method == 'POST':
        uploaded = request.files.get('file')
        if not uploaded or uploaded.filename == '':
            flash('Vui lòng chọn file hình ảnh.', 'danger')
            return redirect(request.url)
        if not allowed_file(uploaded.filename):
            flash('Định dạng file không được hỗ trợ.', 'danger')
            return redirect(request.url)
        filename = secure_filename(uploaded.filename)
        data = uploaded.read()
        image_b64 = base64.b64encode(data).decode('utf-8')
        image_stream = io.BytesIO(data)
        try:
            processed = preprocess_image(image_stream)
            preds = MODEL.predict(processed)
            logger.info("Raw model output: %s", preds.tolist())
            # FIX: Giả sử mô hình trả về [0] là Non-nodule và [1] là Nodule (hoặc chỉ trả về xác suất Nodule)
            # Dùng logic an toàn cho cả 1 và 2 chiều (giả sử chỉ trả về xác suất Nodule)
            p_nodule = float(preds[0][0]) if preds.ndim == 2 and preds.shape[1] >= 1 else float(preds[0])

            label = 'Nodule' if p_nodule >= 0.5 else 'Non-nodule'
            prob = p_nodule if p_nodule >= 0.5 else 1.0 - p_nodule
            prediction_result = {'result': label, 'probability': float(np.round(prob,6)), 'raw_output': float(np.round(p_nodule,6))}
            flash('Dự đoán AI hoàn tất.', 'success')
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            flash(f'Lỗi khi xử lý hình ảnh hoặc dự đoán: {e}', 'danger')
            return redirect(request.url)
    return render_template('emr_prediction.html', prediction=prediction_result, filename=filename, image_b64=image_b64)



# --------------------------------------------------------
# RUN APP
# --------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 EMR AI is running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)


