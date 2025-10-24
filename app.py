# -*- coding: utf-8 -*-
# app.py: EMR AI - FIXED GOOGLE DRIVE DOWNLOAD + REAL KERAS PREDICTION
# CHỈ HIỂN THỊ THUMBNAIL 200x200 thay vì full image

import base64
import os
import io
import logging
import time
import requests
from PIL import Image
from flask import (
    Flask, flash, redirect, render_template, request, session, url_for
)
import pandas as pd
import gdown
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten

# ==========================================================
# KHỞI TẠO BIẾN TOÀN CỤC
# ==========================================================
MODEL = None
MODEL_LOADED = False
IS_DUMMY_MODE = False

# ==========================================================
# CẤU HÌNH GOOGLE DRIVE VÀ MODEL
# ==========================================================
DRIVE_FILE_ID = "1ORV8tDkT03fxjRyaWU5liZ2bHQz3YQC"
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILE_NAME)
MODEL_INPUT_SIZE = (224, 224)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================================
# HÀM TẢI FILE TỪ GOOGLE DRIVE
# ==========================================================
def download_file_from_gdrive(file_id, destination, max_retries=3):
    """
    Tải file từ Google Drive với cơ chế thử lại và kiểm tra quyền truy cập.
    Trả về: (success: bool, is_dummy: bool).
    """
    if os.path.exists(destination):
        logger.info(f"File đã tồn tại: {destination}. Đang kiểm tra tính hợp lệ...")
        try:
            # Kiểm tra xem file có phải là mô hình Keras hợp lệ
            model = load_model(destination)
            model.predict(np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)))
            logger.info("✅ File tồn tại là mô hình Keras hợp lệ.")
            return True, False
        except Exception as e:
            logger.warning(f"❌ File tồn tại nhưng không phải mô hình Keras hợp lệ: {e}. Tải lại...")
            os.remove(destination)  # Xóa file không hợp lệ

    logger.info(f"Đang cố gắng tải model từ GDrive ID: {file_id} về {destination}...")
    url = f"https://drive.google.com/uc?id={file_id}&export=download"

    for attempt in range(max_retries):
        try:
            # Kiểm tra quyền truy cập trước
            response = requests.head(url, allow_redirects=True)
            if response.status_code != 200:
                logger.error(f"❌ Không thể truy cập file. Status code: {response.status_code}")
                continue

            # Tải file bằng gdown
            gdown.download(id=file_id, output=destination, quiet=False, fuzzy=True)
            if os.path.exists(destination):
                # Kiểm tra tính hợp lệ của mô hình
                model = load_model(destination)
                model.predict(np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)))
                logger.info("✅ Tải xuống và xác minh mô hình thực tế hoàn tất.")
                return True, False
            else:
                logger.error("❌ Tải xuống thất bại: File không tìm thấy sau khi chạy gdown.")
                continue

        except Exception as e:
            logger.error(f"❌ Lỗi khi tải file GDrive (lần {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

            # Tạo mô hình dummy nếu tất cả các lần thử thất bại
            logger.warning("⚠️ Chuyển sang tạo mô hình dummy...")
            try:
                dummy_model = Sequential([
                    Input(shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)),
                    Conv2D(8, (3, 3), activation='relu'),
                    Flatten(),
                    Dense(1, activation='sigmoid')
                ])
                dummy_model.compile(optimizer='adam', loss='binary_crossentropy')
                dummy_model.save(destination)
                logger.info(f"✅ Đã tạo mô hình dummy tại {destination} để mô phỏng.")
                return True, True
            except Exception as dummy_e:
                logger.error(f"❌ Lỗi nghiêm trọng: Không thể tạo mô hình dummy: {dummy_e}")
                return False, False

    return False, False

# ==========================================================
# TẢI VÀ LOAD MODEL
# ==========================================================
try:
    success, is_dummy = download_file_from_gdrive(DRIVE_FILE_ID, MODEL_PATH)
    if success:
        logger.info(f"⏳ Đang tải model Keras từ: {MODEL_PATH}")
        MODEL = load_model(MODEL_PATH)
        logger.info("🚀 Tải model Keras thành công! Model đã sẵn sàng.")
        MODEL_LOADED = True
        IS_DUMMY_MODE = is_dummy
    else:
        logger.error("❌ Không thể tải hoặc tạo model. Chuyển sang FIXED MODE.")
        MODEL_LOADED = False
        IS_DUMMY_MODE = True

except Exception as e:
    logger.error(f"❌ LỖI KHI LOAD MODEL KERAS: {e}. Chuyển sang FIXED MODE.")
    MODEL = None
    MODEL_LOADED = False
    IS_DUMMY_MODE = True

# ==========================================================
# HÀM XỬ LÝ ẢNH VÀ DỰ ĐOÁN
# ==========================================================
def img_to_array(img):
    """Chuyển ảnh PIL thành mảng numpy."""
    return np.asarray(img)

def predict_image(img_bytes):
    if IS_DUMMY_MODE and MODEL_LOADED:
        prob_val = 0.5 + (np.random.rand() * 0.1 - 0.05)
        result = "Nodule (U)" if prob_val > 0.5 else "Non-nodule (Không U)"
        prob = prob_val if prob_val > 0.5 else 1.0 - prob_val
        return {
            "result": result,
            "probability": float(prob),
            "message": "Model đang chạy chế độ mô phỏng (DUMMY MODE). Kết quả không đáng tin cậy."
        }

    if not MODEL_LOADED or MODEL is None:
        return {"result": "LỖI HỆ THỐNG", "probability": 0.0, "message": "Model AI chưa được tải thành công."}

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(MODEL_INPUT_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = MODEL.predict(img_array)[0][0]
        threshold = 0.5
        result = "Nodule (U)" if prediction >= threshold else "Non-nodule (Không U)"
        prob = float(prediction) if prediction >= threshold else float(1.0 - prediction)

        return {"result": result, "probability": prob, "message": "Dự đoán thành công bằng REAL KERAS MODEL."}

    except Exception as e:
        logger.error(f"Error during REAL Keras prediction: {e}")
        return {"result": "LỖI KỸ THUẬT", "probability": 0.0, "message": f"Lỗi: {e}"}

# ==========================================================
# FLASK APP CONFIG
# ==========================================================
app = Flask(__name__)
app.secret_key = "emr-fixed-2025-no-crash"
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB MAX
MAX_FILE_SIZE_MB = 4
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_image_to_b64(img_bytes, max_size=200):
    """Tạo thumbnail 200x200."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")
        return None

# ==========================================================
# ROUTES
# ==========================================================
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
    
    model_status = (
        "❌ LỖI TOÀN BỘ (KHÔNG CÓ MODEL AI)" if not MODEL_LOADED and IS_DUMMY_MODE else
        "⚠️ FIXED MODE (ĐANG DÙNG DUMMY MODEL)" if MODEL_LOADED and IS_DUMMY_MODE else
        "✅ REAL KERAS MODEL LOADED" if MODEL_LOADED and not IS_DUMMY_MODE else
        "❓ TRẠNG THÁI KHÔNG XÁC ĐỊNH"
    )
    return render_template("dashboard.html", model_status=model_status)

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
            file_stream_bytes = file.read()
            file_stream = io.BytesIO(file_stream_bytes)
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

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if 'user' not in session:
        return redirect(url_for("index"))
        
    prediction_result = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        try:
            file = request.files.get('file')
            if not file or not file.filename:
                flash("❌ Chưa chọn file.", "danger")
                return render_template('emr_prediction.html')
                
            filename = file.filename
            if not allowed_file(filename):
                flash("❌ Chỉ chấp nhận JPG, PNG, GIF, BMP", "danger")
                return render_template('emr_prediction.html')

            img_bytes = file.read()
            file_size = len(img_bytes)
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                flash(f"❌ File quá lớn ({file_size//(1024*1024)}MB > 4MB)", "danger")
                return render_template('emr_prediction.html')
            
            if file_size == 0:
                flash("❌ File rỗng.", "danger")
                return render_template('emr_prediction.html')

            # Limit cache size to prevent session cookie overflow
            if 'prediction_cache' not in session:
                session['prediction_cache'] = {}
            if len(session['prediction_cache']) > 5:  # Limit to 5 cached predictions
                session['prediction_cache'].pop(next(iter(session['prediction_cache'])))
            
            if filename in session['prediction_cache']:
                cached = session['prediction_cache'][filename]
                prediction_result = cached['prediction']
                image_b64 = cached['image_b64']
                flash(f"✅ Từ cache: {filename}", "info")
            else:
                start_time = time.time()
                prediction_result = predict_image(img_bytes)
                thumb_b64 = safe_image_to_b64(img_bytes, max_size=200)
                image_b64 = thumb_b64 if thumb_b64 else None
                end_time = time.time()
                logger.info(f"AI Prediction took {end_time - start_time:.2f} seconds.")

                session['prediction_cache'][filename] = {
                    'prediction': prediction_result,
                    'image_b64': image_b64
                }
                session.modified = True
            
        except Exception as e:
            logger.error(f"PREDICTION CRASH: {e}")
            flash("❌ Lỗi xử lý. Thử file nhỏ hơn 4MB.", "danger")
            return render_template('emr_prediction.html')

    return render_template('emr_prediction.html', 
                           prediction=prediction_result, 
                           filename=filename, 
                           image_b64=image_b64,
                           model_loaded=MODEL_LOADED)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/health")
def health():
    return {"status": "healthy"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    model_status_log = (
        "FATAL ERROR/FIXED MODE (MODEL NOT LOADED)" if not MODEL_LOADED else
        "DUMMY MODE (FAKE MODEL LOADED)" if IS_DUMMY_MODE else
        "REAL KERAS LOADED"
    )
    logger.info(f"🚀 EMR AI - MODE: {model_status_log}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
