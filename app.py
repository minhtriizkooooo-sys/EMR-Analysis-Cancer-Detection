import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
import pandas as pd
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import requests
from keras.models import load_model

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
# Load Keras model từ Hugging Face
# =============================
MODEL_URL = "https://huggingface.co/minhtriizkooooo/EMR-Analysis-Cancer_Detection/resolve/main/best_weights_model.keras"
MODEL_PATH = os.path.join(MODEL_CACHE, "best_weights_model.keras")

# Tải model nếu chưa tồn tại
if not os.path.exists(MODEL_PATH):
    print("⏳ Tải model từ Hugging Face...")
    r = requests.get(MODEL_URL, stream=True)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Model đã tải xong và lưu tại {MODEL_PATH}")
    else:
        raise Exception(f"❌ Không tải được model, status_code={r.status_code}")

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model Keras đã load thành công!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    model = None

# =============================
# Route: Trang đăng nhập
# =============================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        userID = request.form.get("userID")
        password = request.form.get("password")
        # Demo login
        if userID == "user_demo" and password == "Test@123456":
            session["user"] = userID
            return redirect(url_for("dashboard"))
        else:
            flash("ID hoặc mật khẩu không đúng!", "danger")
    return render_template("index.html")

# =============================
# Route: Dashboard
# =============================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

# =============================
# Route: Logout
# =============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =============================
# Route: EMR Profile (CSV/Excel Analysis)
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
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                summary = df.describe(include="all").to_html(classes="table-auto border-collapse border border-gray-300")
            except Exception as e:
                flash(f"Lỗi đọc file: {e}", "danger")
    return render_template("emr_profile.html", filename=filename, summary=summary)

# =============================
# Route: EMR Prediction (Image Prediction)
# =============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and model:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            try:
                # Mở ảnh, resize nếu cần, chuyển về array numpy
                img = Image.open(filepath).convert("RGB")
                img = img.resize((224, 224))  # sửa theo input model của bạn
                x = np.array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                pred = model.predict(x)[0]
                prediction = float(pred)  # giả sử output 1 giá trị
            except Exception as e:
                flash(f"Lỗi dự đoán: {e}", "danger")
    return render_template("emr_prediction.html", filename=filename, prediction=prediction)

# =============================
# Chạy Flask
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
