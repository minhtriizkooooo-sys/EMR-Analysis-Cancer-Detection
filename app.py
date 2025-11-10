import os
import io
import base64
import zipfile
import requests
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ydata_profiling import ProfileReport

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)
app.secret_key = "secret_key"

# Đường dẫn tạm
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ----------------------------
# Tải hoặc giải nén model
# ----------------------------
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")

def extract_model():
    """Tự động hợp nhất và giải nén các phần model nếu cần"""
    if not os.path.exists(MODEL_PATH):
        parts = [f"models/best_weights_model.keras.00{i}" for i in range(1, 5)]
        if all(os.path.exists(p) for p in parts):
            print("🔄 Extracting model parts...")
            with open(MODEL_PATH, "wb") as f_out:
                for p in parts:
                    with open(p, "rb") as f_in:
                        f_out.write(f_in.read())
            print("✅ Model reconstructed successfully.")
        else:
            print("⚠️ Model file not found. Please upload model parts.")

extract_model()

# ----------------------------
# Load model nếu có
# ----------------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("⚠️ Model file missing!")

# ----------------------------
# ROUTES
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ---------- EMR PROFILE ----------
@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Không tìm thấy file tải lên.")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("Vui lòng chọn file CSV hoặc Excel.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Tạo báo cáo
            profile = ProfileReport(df, title="Báo cáo Phân tích Dữ liệu EMR", explorative=True)
            profile_html = profile.to_html()

            return render_template("EMR_Profile.html", report_html=profile_html)
        except Exception as e:
            flash(f"Lỗi khi xử lý file: {str(e)}")
            return redirect(request.url)
    return render_template("EMR_Profile.html", report_html=None)

# ---------- EMR PREDICTION ----------
@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Không tìm thấy file ảnh.")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("Vui lòng chọn ảnh để dự đoán.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            if model is None:
                flash("Model chưa được tải.")
                return redirect(request.url)

            # Chuẩn bị ảnh
            image = load_img(file_path, target_size=(240, 240))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0

            # Dự đoán
            pred = model.predict(image)[0][0]
            result = "Nodule" if pred > 0.5 else "Non-Nodule"
            probability = round(float(pred) * 100, 2)

            # Hiển thị ảnh
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

            return render_template(
                "EMR_Prediction.html",
                prediction=result,
                probability=probability,
                image_data=img_b64
            )
        except Exception as e:
            flash(f"Lỗi dự đoán: {str(e)}")
            return redirect(request.url)

    return render_template("EMR_Prediction.html", prediction=None)

# ----------------------------
# MAIN
# ----------------------------
i



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info("🚀 EMR AI App Started")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

