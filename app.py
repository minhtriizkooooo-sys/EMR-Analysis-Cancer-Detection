# -*- coding: utf-8 -*-
"""
app.py — EMR AI LITE
→ Phân tích dữ liệu EMR (CSV) bằng pandas (nâng cao)
→ Dự đoán hình ảnh y tế bằng mô hình Keras lưu trên HuggingFace
→ Lazy loading model để tránh lỗi 502 / timeout
"""
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

# ==========================================================
# 🧠 SAFE TENSORFLOW CONFIG
# ==========================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    # Disable GPU visibility for CPU usage (good practice on resource-limited environment)
    tf.config.set_visible_devices([], 'GPU')
    K.clear_session()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# 🔧 FLASK CONFIG
# ==========================================================
app = Flask(__name__)
# Thay secrets.token_hex(16) bằng biến môi trường hoặc giá trị cố định an toàn
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['ALLOWED_EMR_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
app.config['ALLOWED_EXTENSIONS'] = app.config['ALLOWED_IMAGE_EXTENSIONS'] | app.config['ALLOWED_EMR_EXTENSIONS']
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==========================================================
# 🔬 GLOBAL MODEL CONFIG AND VARIABLE
# ==========================================================
MODEL_REPO = 'minhtriizkooooo/EMR-Analysis-Cancer_Detection'
MODEL_FILENAME = 'best_weights_model.keras'
IMG_SIZE = (224, 224)

# KHỞI TẠO model = None ở mức toàn cục
model = None 

# ==========================================================
# ⚙️ LOAD MODEL SAFELY
# ==========================================================
def load_keras_model():
    """Load Keras model safely from Hugging Face"""
    global model
    
    try:
        logger.info("⏳ Downloading model from Hugging Face...")
        # Note: hf_hub_download is blocking
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        
        # Note: load_model is blocking
        model = load_model(model_path, compile=False)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")

# ==========================================================
# 🧩 HELPER FUNCTIONS
# ==========================================================
def allowed_file(filename, allowed_extensions=app.config['ALLOWED_EXTENSIONS']):
    """Check allowed file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions




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
@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    """Handle EMR image prediction"""
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang dự đoán.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Chưa chọn file ảnh.', 'danger')
            return redirect(url_for('emr_prediction'))

        # Check for image file extensions
        if not allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
            flash('Chỉ chấp nhận file ảnh (PNG, JPG, JPEG, GIF, BMP).', 'danger')
            return redirect(url_for('emr_prediction'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        if img_array is None:
            flash('Lỗi khi xử lý hình ảnh.', 'danger')
            return redirect(url_for('emr_prediction'))

        try:
            # Check if model is loaded globally by the Master process
            global model
            if model is None:
                flash('Mô hình AI chưa được tải. Vui lòng kiểm tra logs để biết lỗi tải mô hình.', 'danger')
                return redirect(url_for('emr_prediction'))

            # BƯỚC QUAN TRỌNG: Gọi predict
            input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            pred = model.predict(input_tensor, verbose=0)
            
            # 👇 THÊM DỌN DẸP BỘ NHỚ SAU DỰ ĐOÁN (Chống OOM)
            K.clear_session() 
            gc.collect()
            logger.info("✅ Keras/TF session and Python garbage collected.")
            # 👆 KẾT THÚC DỌN DẸP BỘ NHỚ

            # Assuming binary classification where pred[0][0] is the probability of the positive class
            probability = float(pred[0][0])
            result = 'Nodule' if probability > 0.5 else 'Non-nodule'
            
            # Store prediction data in session
            session['prediction_result'] = {
                'result': result,
                'probability': round(probability * 100, 2),
                'filename': filename,
                'image_b64': image_to_base64(file_path),
                'mime_type': mimetypes.guess_type(file_path)[0] or 'image/jpeg'
            }
            
            flash(f'Dự đoán hoàn tất: {result} với xác suất {round(probability * 100, 2)}%.', 'success')
            return redirect(url_for('emr_prediction'))

        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            flash('Lỗi khi dự đoán hình ảnh. Có thể do timeout.', 'danger')
            
            # DỌN DẸP BỘ NHỚ KỂ CẢ KHI CÓ LỖI
            try:
                K.clear_session()
                gc.collect()
            except:
                pass
            
            return redirect(url_for('emr_prediction'))

    # Retrieve and clear prediction data for GET request (display results)
    prediction_data = session.pop('prediction_result', None)

    return render_template(
        'emr_prediction.html',
        prediction=prediction_data,
        uploaded_image=None, 
        image_b64=None if not prediction_data else prediction_data['image_b64'],
        filename=None if not prediction_data else prediction_data['filename']
    )


# --------------------------------------------------------
# RUN APP
# --------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 EMR AI is running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)

