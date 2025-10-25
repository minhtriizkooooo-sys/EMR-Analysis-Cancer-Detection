import os
import secrets
import numpy as np
import pandas as pd
# Import TensorFlow (cần thiết cho Keras)
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image
# Đảm bảo dùng keras.models.load_model hoặc từ tensorflow.keras.models
from keras.models import load_model 
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
HF_MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
HF_MODEL_FILE = "best_weights_model.keras"
LOCAL_MODEL_PATH = os.path.join(MODEL_CACHE, HF_MODEL_FILE)

model = None
try:
    print("⏳ Tải model từ Hugging Face...")
    LOCAL_MODEL_PATH = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILE, cache_dir=MODEL_CACHE)
    # NOTE QUAN TRỌNG: Lỗi Shape Mismatch (expected 3 channels, received 1) thường xảy ra tại đây.
    # Lỗi này cần được khắc phục bằng cách thay đổi cách lưu/định nghĩa model của bạn, 
    # nhưng ít nhất ứng dụng sẽ chạy được Flask routing.
    model = load_model(LOCAL_MODEL_PATH)
    print(f"✅ Model đã tải xong và lưu tại {LOCAL_MODEL_PATH}")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    print("LƯU Ý: Lỗi này thường liên quan đến sự không tương thích giữa cách model được lưu và môi trường TensorFlow hiện tại.")

# =============================
# Dummy user
# =============================
USERS = {"user_demo": "Test@123456"}

# =============================
# Routes
# =============================
# FIX: Đổi tên hàm từ 'index' thành 'login' để Flask tạo ra endpoint 'login'.
# Điều này khớp với url_for('login') mà template index.html đang tìm kiếm,
# khắc phục lỗi BuildError: Could not build url for endpoint 'login'.
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
    # Cập nhật url_for('index') thành url_for('login')
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    # Cập nhật url_for('index') thành url_for('login')
    return redirect(url_for("login"))

# =============================
# EMR CSV/Excel Analysis
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    # Cập nhật url_for('index') thành url_for('login')
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
                if filename.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                summary = df.describe().to_html(classes="table-auto w-full")
            except Exception as e:
                flash(f"Lỗi khi đọc file: {e}", "danger")
    return render_template("emr_profile.html", filename=filename, summary=summary)

# =============================
# EMR Image Prediction
# =============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    # Cập nhật url_for('index') thành url_for('login')
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and model:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                # Mở ảnh, convert RGB (3 kênh), resize, normalize
                img = Image.open(file_path).convert("RGB")
                img = img.resize((224, 224))  # tùy theo input model
                x = np.array(img)/255.0
                x = np.expand_dims(x, axis=0)

                pred = model.predict(x)
                prediction = str(pred[0])  # hiển thị dạng chuỗi
            except Exception as e:
                flash(f"Lỗi khi dự đoán: {e}", "danger")
        elif not model:
            flash("Model chưa load được! (Lỗi khởi tạo)", "danger")
    return render_template("emr_prediction.html", filename=filename, prediction=prediction)

# =============================
# Chạy Flask
# =============================
if __name__ == "__main__":
    # Đảm bảo dùng host 0.0.0.0 và PORT 
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
