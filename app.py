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

# Cấu hình thư mục và giới hạn
UPLOAD_DIR = Path('/tmp/uploads')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)

app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# --- CONFIG CƠ BẢN VÀ GIỚI HẠN TỐI ƯU ---
MAX_FILE_SIZE_MB = 10 
# GIẢM GIỚI HẠN DÒNG XỬ LÝ ĐỂ TRÁNH TIMEOUT CỦA GUNICORN (TỪ 20K -> 5K)
MAX_ROWS_FOR_PROFILE = 5000 
MIN_MODEL_SIZE_MB = 5 

# --- Model config (Local Load) ---
MODEL = None
TARGET_SIZE = (240, 240)
MODEL_FILENAME = "best_weights_model.keras"

MODEL_DIR = Path('/app/models') 
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

def load_keras_model():
    """Tải model trực tiếp từ thư mục cục bộ (/app/models) và kiểm tra kích thước."""
    global MODEL
    
    if MODEL is not None:
        return MODEL
    
    min_bytes = MIN_MODEL_SIZE_MB * 1024 * 1024
    
    # 1. KIỂM TRA SỰ TỒN TẠI VÀ KÍCH THƯỚC FILE
    if not MODEL_PATH.exists():
        logger.error("❌ CRITICAL: Model file NOT FOUND at %s.", MODEL_PATH)
        return None
    
    if MODEL_PATH.stat().st_size < min_bytes:
        logger.error("❌ CRITICAL: Model file is too small (%s bytes). Likely an error file.", MODEL_PATH.stat().st_size)
        return None
        
    # 2. TẢI MODEL VÀO BỘ NHỚ
    try:
        logger.info("🔥 Loading Keras model from local path: %s", MODEL_PATH)
        MODEL = load_model(str(MODEL_PATH), compile=False, custom_objects={'Adamax': Adamax}) 
        logger.info("✅ Model loaded successfully from local file.")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Error loading model from local file: {e}")
        MODEL = None
    
    return MODEL

# Tải model ngay khi ứng dụng bắt đầu
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





@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("userID", "").strip()
    password = request.form.get("password", "").strip()
    
    if username == "user_demo" and password == "Test@123456":
        session['user'] = username
        return redirect(url_for("dashboard"))
    flash("Sai ID hoặc mật khẩu.", "danger")
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for("index"))
    # FIXED MODE vì đã loại bỏ model TensorFlow/Keras
    return render_template("dashboard.html", model_status="✅ FIXED MODE")

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
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
            
            # Check file size early (if not already done by Nginx/MAX_CONTENT_LENGTH)
            if len(file_stream.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File quá lớn ({len(file_stream.getvalue())//(1024*1024)}MB > 4MB)")

            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_stream)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_stream)
            else:
                summary = f"<p class='text-red-500 font-semibold'>Chỉ hỗ trợ file CSV hoặc Excel. File: {filename}</p>"
                return render_template('emr_profile.html', summary=summary, filename=filename)

            rows, cols = df.shape
            col_info = []
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                unique_count = df[col].nunique()
                desc_stats = ""
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe().to_dict()
                    desc_stats = (
                        f"Min: {desc.get('min', 'N/A'):.2f}, "
                        f"Max: {desc.get('max', 'N/A'):.2f}, "
                        f"Mean: {desc.get('mean', 'N/A'):.2f}, "
                        f"Std: {desc.get('std', 'N/A'):.2f}"
                    )
                
                col_info.append(f"""
                    <li class="bg-gray-50 p-3 rounded-lg border-l-4 border-primary-green">
                        <strong class="text-gray-800">{col}</strong>
                        <ul class="ml-4 text-sm space-y-1 mt-1 text-gray-600">
                            <li><i class="fas fa-code text-indigo-500 w-4"></i> Kiểu dữ liệu: {dtype}</li>
                            <li><i class="fas fa-exclamation-triangle text-yellow-500 w-4"></i> Thiếu: {missing} ({missing/rows*100:.2f}%)</li>
                            <li><i class="fas fa-hashtag text-teal-500 w-4"></i> Giá trị duy nhất: {unique_count}</li>
                            {'<li class="text-xs text-gray-500"><i class="fas fa-chart-bar text-green-500 w-4"></i> Thống kê mô tả: ' + desc_stats + '</li>' if desc_stats else ''}
                        </ul>
                    </li>
                """)
            
            info = f"""
            <div class='bg-green-50 p-6 rounded-lg shadow-inner'>
                <h3 class='text-2xl font-bold text-product-green mb-4'><i class='fas fa-info-circle mr-2'></i> Thông tin Tổng quan</h3>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-left'>
                    <p class='font-medium text-gray-700'><i class='fas fa-th-list text-indigo-500 mr-2'></i> Số dòng dữ liệu: <strong>{rows}</strong></p>
                    <p class='font-medium text-gray-700'><i class='fas fa-columns text-indigo-500 mr-2'></i> Số cột dữ liệu: <strong>{cols}</strong></p>
                </div>
            </div>
            """
            
            table_html = df.head().to_html(classes="table-auto min-w-full divide-y divide-gray-200", index=False)
            summary = info
            summary += f"<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-cogs mr-2 text-primary-green'></i> Phân tích Cấu trúc Cột ({cols} Cột):</h4>"
            summary += f"<ul class='space-y-3 grid grid-cols-1 md:grid-cols-2 gap-3'>{''.join(col_info)}</ul>"
            summary += "<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-table mr-2 text-primary-green'></i> 5 Dòng Dữ liệu Đầu tiên:</h4>"
            summary += "<div class='overflow-x-auto shadow-md rounded-lg'>" + table_html + "</div>"
            
        except Exception as e:
            summary = f"<p class='text-red-500 font-semibold text-xl'>Lỗi xử lý file EMR: <code class='text-gray-700 bg-gray-100 p-1 rounded'>{e}</code></p>"
            
    return render_template('emr_profile.html', summary=summary, filename=filename)


# --- EMR Prediction ---
@app.route('/emr_prediction', methods=['GET','POST'])
@login_required
def emr_prediction():
    prediction_result = None
    filename = None
    image_b64 = None
    
    if MODEL is None:
        flash('Hệ thống AI chưa sẵn sàng. Vui lòng kiểm tra log lỗi tải model.', 'danger')
        return render_template('emr_prediction.html')
    
    if request.method == 'POST':
        uploaded = request.files.get('file')
        if not uploaded or uploaded.filename == '':
            flash('Vui lòng chọn file hình ảnh.', 'danger')
            return redirect(request.url)
        if not allowed_file(uploaded.filename):
            flash('Định dạng file không được hỗ trợ.', 'danger')
            return redirect(request.url)
        filename = secure_filename(uploaded.filename)
        # read bytes and keep base64 for UI
        data = uploaded.read()
        image_b64 = base64.b64encode(data).decode('utf-8')
        # prepare file-like for preprocess
        image_stream = io.BytesIO(data)
        image_stream.seek(0)
        try:
            processed = preprocess_image_match_training(image_stream)
            preds = MODEL.predict(processed)
            logger.info("Raw model output: %s", preds.tolist())
            
            # Xử lý output model (Giả định Sigmoid, shape (1,1))
            if preds.ndim == 2 and preds.shape[1] == 1:
                p_nodule = float(preds[0][0])
            else:
                p_nodule = float(np.max(preds[0])) # Fallback
            
            # decide label
            if p_nodule >= 0.5:
                label = 'Nodule'
                prob = p_nodule
            else:
                label = 'Non-nodule'
                prob = 1.0 - p_nodule
            
            prediction_result = {'result': label, 'probability': float(np.round(prob, 6)), 'raw_output': float(np.round(p_nodule, 6))}
            flash('Dự đoán AI hoàn tất.', 'success')
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            flash(f'Lỗi khi xử lý hình ảnh hoặc dự đoán: {e}', 'danger')
            return redirect(request.url)
            
    return render_template('emr_prediction.html', 
                           prediction=prediction_result, 
                           filename=filename, 
                           image_b64=image_b64)




@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    # KHÔNG DÙNG 10000. DÙNG BIẾN MÔI TRƯỜNG $PORT DO Render CUNG CẤP
    port = int(os.environ.get("PORT", 10000)) # Dùng 5000 làm mặc định cho local
    logger.info("🚀 EMR AI - FIXED BASE64 CRASH")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


