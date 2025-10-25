from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import requests
from huggingface_hub import hf_hub_download

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)

# =============================
# Tải model từ Hugging Face
# =============================
MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
MODEL_FILENAME = "best_weights_model.keras"

print("🔽 Đang tải model từ Hugging Face...")
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = load_model(model_path)
    print("✅ Tải model thành công.")
except Exception as e:
    print(f"❌ Không thể load model: {e}")
    model = None

# =============================
# Trang chủ
# =============================
@app.route('/')
def index():
    return render_template('index.html')

# =============================
# Dashboard
# =============================
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# =============================
# Trang upload & dự đoán ảnh
# =============================
@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    prediction_result = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('emr_prediction.html', prediction_result="Không có tệp tải lên.")

        file = request.files['file']
        if file.filename == '':
            return render_template('emr_prediction.html', prediction_result="Chưa chọn ảnh.")

        try:
            # Đọc ảnh
            img = Image.open(file.stream)

            # Nếu ảnh là grayscale → chuyển sang RGB để khớp model
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize về kích thước đúng với model (thay đổi nếu cần)
            img = img.resize((224, 224))  # chỉnh theo input của model bạn

            # Chuẩn hóa
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Dự đoán
            if model is not None:
                preds = model.predict(img_array)
                label = "Nodule" if preds[0][0] > 0.5 else "Non-nodule"
                prediction_result = f"Kết quả dự đoán: {label} (xác suất: {preds[0][0]:.4f})"
            else:
                prediction_result = "Model chưa được tải thành công."

        except Exception as e:
            prediction_result = f"Lỗi xử lý ảnh: {str(e)}"

    return render_template('emr_prediction.html', prediction_result=prediction_result, image_url=image_url)

# =============================
# Hồ sơ EMR
# =============================
@app.route('/emr_profile')
def emr_profile():
    return render_template('emr_profile.html')

# =============================
# Cấu hình Render
# =============================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
