import os
import secrets
import numpy as np
import pandas as pd
# SỬ DỤNG TF.KERAS CHUẨN ĐỂ TƯƠNG THÍCH TỐT NHẤT
import tensorflow as tf 
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image

# THAY ĐỔI: Sử dụng tf.keras.models.load_model
from tensorflow.keras.models import load_model 

# ĐỊNH NGHĨA CUSTOM OBJECTS MỚI (FIX LỖI)
# EfficientNet B0 sử dụng hàm kích hoạt Swish (hay còn gọi là SiLU).
# Khi tải model, Keras đôi khi không nhận diện được hàm này một cách chính xác
# và dẫn đến việc xây dựng lại cấu trúc mạng sai (lỗi 1 kênh).
from tensorflow.keras.layers import Activation 
from tensorflow.keras.activations import swish as tf_swish

# CẬP NHẬT QUAN TRỌNG: Đăng ký cả hai tên gọi 'swish' và 'SiLU'
custom_objects = {
    'swish': tf_swish,
    'SiLU': tf_swish  
}

from huggingface_hub import hf_hub_download

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_CACHE = "model_cache"
os.makedirs(MODEL_CACHE, exist_ok=True)

# =============================
# Tải model từ Hugging Face
# =============================
HF_MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer-Detection"
HF_MODEL_FILE = "best_weights_model.keras"
LOCAL_MODEL_PATH = os.path.join(MODEL_CACHE, HF_MODEL_FILE)

# LẤY HF TOKEN TỪ BIẾN MÔI TRƯỜNG (Dành cho Private/Gated Repos)
# Nếu model là Public, biến token sẽ là None
HF_TOKEN = os.environ.get("HF_TOKEN") # <--- CODE ĐÃ ĐỌC BIẾN NÀY

model = None
try:
    print("⏳ Tải model từ Hugging Face...")
    
    # CẬP NHẬT: Thêm tham số token=HF_TOKEN vào hf_hub_download
    LOCAL_MODEL_PATH = hf_hub_download(
        repo_id=HF_MODEL_REPO, 
        filename=HF_MODEL_FILE, 
        cache_dir=MODEL_CACHE,
        token=HF_TOKEN # <--- TOKEN ĐƯỢC SỬ DỤNG TẠI ĐÂY
    )
    
    # CHIẾN LƯỢC MỚI NHẤT: Cung cấp custom_objects cho cả 'swish' và 'SiLU', và vẫn tắt compile
    model = load_model(LOCAL_MODEL_PATH, custom_objects=custom_objects, compile=False)
    print(f"✅ Model THẬT (EfficientNetB0) đã tải xong và lưu tại {LOCAL_MODEL_PATH}")

except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    # Cập nhật thông báo lỗi để nhắc nhở về token
    print("LƯU Ý QUAN TRỌNG: Model THẬT không tải được. Chức năng dự đoán ảnh sẽ bị vô hiệu hóa.")
    print("KIỂM TRA: Lỗi 401 có thể do repo Hugging Face là Private/Gated. Vui lòng thêm biến môi trường HF_TOKEN trên Render.")
    print("Kiểm tra: Nếu lỗi vẫn là (..., 1), hãy thử đảm bảo phiên bản TensorFlow/Keras trên Render khớp với Colab.")


# =============================
# Dummy user
# =============================
USERS = {"user_demo": "Test@123456"}

# =============================
# Routes
# =============================
@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        userID = request.form.get("userID")
        password = request.form.get("password")
        if userID in USERS and USERS[userID] == password:
            session["user"] = userID
            return redirect(url_for("dashboard"))
        else:
            flash("Sai ID hoặc mật khẩu", "danger")
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# =============================
# EMR CSV/Excel Analysis
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    filename = None
    summary = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Xử lý dữ liệu EMR
            try:
                # Kiểm tra định dạng file và đọc
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                else:
                    flash("Định dạng file không được hỗ trợ. Vui lòng tải lên CSV hoặc Excel.", "danger")
                    return render_template("emr_profile.html")
                    
                summary = df.describe().to_html(classes="table-auto w-full")
            except Exception as e:
                flash(f"Lỗi khi đọc file: {e}", "danger")
    return render_template("emr_profile.html", filename=filename, summary=summary)

# =============================
# EMR Image Prediction
# =============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        # Chỉ chạy dự đoán khi file có VÀ model đã được tải thành công (model is NOT None)
        if file and model: 
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                # TIỀN XỬ LÝ ẢNH
                # Đảm bảo output là (224, 224, 3) (RGB) để khớp với EfficientNetB0
                img = Image.open(file_path).convert("RGB")
                img = img.resize((224, 224)) 
                x = np.array(img)/255.0
                x = np.expand_dims(x, axis=0)

                # KIỂM TRA ĐẦU VÀO ĐỂ DEBUG
                print(f"Input shape cho model: {x.shape}")
                
                pred = model.predict(x)
                
                # Chuyển kết quả dự đoán thành chuỗi
                prediction_value = np.argmax(pred[0])
                prediction = f"Kết quả dự đoán: Lớp {prediction_value} | Probabilities: {pred[0]}" 
            except Exception as e:
                flash(f"Lỗi khi dự đoán: {e}", "danger")
        elif not model:
            # Thông báo lỗi rõ ràng nếu model không load được
            flash("Model chưa load được! (Lỗi khởi tạo - vui lòng kiểm tra log tải model)", "danger")
    return render_template("emr_prediction.html", filename=filename, prediction=prediction)

# =============================
# Chạy Flask
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
