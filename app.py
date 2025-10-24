# -*- coding: utf-8 -*-
# app.py: EMR AI - FIXED DOWNLOAD + REAL KERAS PREDICTION + NO SESSION CACHE
# CHỈ HIỂN THỊ THUMBNAIL 100x100 thay vì full image + LOẠI BỎ CACHE ĐỂ TRÁNH COOKIE QUÁ LỚN

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
# CẤU HÌNH MODEL
# ==========================================================
DRIVE_FILE_ID = "1ORV8tDkT03fxjRyaWU5liZ2bHQz3YQC"  # ⚠️ ID NÀY DẪN ĐẾN 404 - FILE KHÔNG TỒN TẠI. THAY BẰNG ID HỢP LỆ!
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILE_NAME)
MODEL_INPUT_SIZE = (224, 224)  # Kích thước input: 224x224
ALTERNATIVE_DOWNLOAD_URL = None  # Ví dụ: "https://your-direct-link.com/best_weights_model.keras" (nếu có)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================================
# HÀM TẢI FILE TỪ GOOGLE DRIVE HOẶC URL THAY THẾ
# ==========================================================
def download_file_from_gdrive(file_id, destination, max_retries=3):
    """
    Tải file từ Google Drive hoặc URL thay thế với retry và xác minh.
    Trả về: (success: bool, is_dummy: bool).
    """
    if os.path.exists(destination):
        logger.info(f"File đã tồn tại: {destination}. Đang kiểm tra tính hợp lệ...")
        try:
            temp_model = load_model(destination)
            temp_model.predict(np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)), verbose=0)
            logger.info("✅ File tồn tại là mô hình Keras hợp lệ.")
            return True, False
        except Exception as e:
            logger.warning(f"❌ File tồn tại nhưng không hợp lệ: {e}. Xóa và tải lại...")
            os.remove(destination)

    gdrive_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    logger.info(f"Đang cố gắng tải model từ GDrive ID: {file_id}...")

    success = False
    for attempt in range(max_retries):
        try:
            head_resp = requests.head(gdrive_url, allow_redirects=True, timeout=10)
            logger.info(f"Lần thử {attempt + 1}: HEAD URL {gdrive_url} -> Status {head_resp.status_code}")
            if head_resp.status_code != 200:
                raise Exception(f"Status code {head_resp.status_code}")

            gdown.download(id=file_id, output=destination, quiet=False, fuzzy=True)
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                temp_model = load_model(destination)
                temp_model.predict(np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)), verbose=0)
                logger.info("✅ Tải và xác minh mô hình THỰC TẾ từ GDrive thành công.")
                return True, False
        except Exception as e:
            logger.error(f"❌ Lỗi GDrive lần {attempt + 1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)

    # Thử URL thay thế nếu có
    if ALTERNATIVE_DOWNLOAD_URL:
        logger.info(f"Thử tải từ URL thay thế: {ALTERNATIVE_DOWNLOAD_URL}")
        try:
            resp = requests.get(ALTERNATIVE_DOWNLOAD_URL, timeout=30)
            if resp.status_code == 200:
                with open(destination, 'wb') as f:
                    f.write(resp.content)
                temp_model = load_model(destination)
                temp_model.predict(np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)), verbose=0)
                logger.info("✅ Tải từ URL thay thế thành công.")
                return True, False
        except Exception as e:
            logger.error(f"❌ Lỗi tải từ URL thay thế: {e}")

    # Tạo dummy nếu tất cả thất bại (để app chạy, nhưng cảnh báo)
    logger.warning("⚠️ Tất cả tải xuống thất bại. Tạo DUMMY MODEL.")
    try:
        dummy_model = Sequential([
            Input(shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)),
            Conv2D(8, (3, 3), activation='relu'),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        dummy_model.compile(optimizer='adam', loss='binary_crossentropy')
        dummy_model.save(destination)
        logger.info("✅ Dummy model tạo thành công.")
        return True, True
    except Exception as dummy_e:
        logger.error(f"❌ Không thể tạo dummy: {dummy_e}")
        return False, False

# ==========================================================
# TẢI MODEL TOÀN CỤC
# ==========================================================
try:
    success, is_dummy = download_file_from_gdrive(DRIVE_FILE_ID, MODEL_PATH)
    if success:
        MODEL = load_model(MODEL_PATH)
        MODEL_LOADED = True
        IS_DUMMY_MODE = is_dummy
        logger.info("🚀 Model tải thành công! DUMMY MODE: {}".format(IS_DUMMY_MODE))
    else:
        raise Exception("Tải model thất bại hoàn toàn.")
except Exception as e:
    logger.error(f"❌ LỖI TẢI MODEL: {e}. Chạy ở FIXED MODE.")
    MODEL_LOADED = False
    IS_DUMMY_MODE = True  # FIXED = no prediction

# ==========================================================
# HÀM DỰ ĐOÁN
# ==========================================================
def img_to_array(img):
    return np.asarray(img)

def predict_image(img_bytes):
    if IS_DUMMY_MODE and MODEL_LOADED:
        prob_val = 0.5 + (np.random.rand() * 0.1 - 0.05)
        result = "Nodule (U)" if prob_val > 0.5 else "Non-nodule (Không U)"
        prob = prob_val if prob_val > 0.5 else 1.0 - prob_val
        return {
            "result": result,
            "probability": float(prob),
            "message": "DUMMY MODE: Kết quả mô phỏng (ID file GDrive không hợp lệ - 404)."
        }

    if not MODEL_LOADED:
        return {"result": "LỖI HỆ THỐNG", "probability": 0.0, "message": "Model chưa tải (FIXED MODE). Kiểm tra file ID."}

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(MODEL_INPUT_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = MODEL.predict(img_array, verbose=0)[0][0]
        threshold = 0.5
        result = "Nodule (U)" if prediction >= threshold else "Non-nodule (Không U)"
        prob = float(prediction) if prediction >= threshold else float(1.0 - prediction)
        return {"result": result, "probability": prob, "message": "Dự đoán bằng REAL MODEL."}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"result": "LỖI", "probability": 0.0, "message": str(e)}

# ==========================================================
# FLASK APP
# ==========================================================
app = Flask(__name__)
app.secret_key = "emr-fixed-2025-no-crash"
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
MAX_FILE_SIZE_MB = 4
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_image_to_b64(img_bytes, max_size=100):  # Giảm xuống 100x100 ~5-10KB
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80, optimize=True)
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Thumbnail error: {e}")
        return None

# LOẠI BỎ CACHE HOÀN TOÀN ĐỂ TRÁNH COOKIE QUÁ LỚN
# Luôn tạo thumbnail & predict mới mỗi POST

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
        "❌ FIXED MODE (KHÔNG CÓ MODEL - FILE ID 404)" if not MODEL_LOADED else
        "⚠️ DUMMY MODE (TẢI THẤT BẠI)" if IS_DUMMY_MODE else
        "✅ REAL MODEL LOADED"
    )
    return render_template("dashboard.html", model_status=model_status)

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))
    # (Giữ nguyên code emr_profile cũ - không thay đổi)
    # ... (copy từ code gốc của bạn)

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
                flash("❌ Chỉ JPG/PNG/GIF/BMP", "danger")
                return render_template('emr_prediction.html')

            img_bytes = file.read()
            if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024 or len(img_bytes) == 0:
                flash("❌ File kích thước không hợp lệ.", "danger")
                return render_template('emr_prediction.html')

            start_time = time.time()
            prediction_result = predict_image(img_bytes)
            image_b64 = safe_image_to_b64(img_bytes, max_size=100)
            logger.info(f"Prediction took {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"POST error: {e}")
            flash("❌ Lỗi xử lý.", "danger")

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
    status = "FIXED (NO MODEL)" if not MODEL_LOADED else "DUMMY" if IS_DUMMY_MODE else "REAL"
    logger.info(f"🚀 EMR AI - MODE: {status} (FILE ID 404 - CẦN SỬA)")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
