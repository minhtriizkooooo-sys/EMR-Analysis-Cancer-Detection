import os
import secrets
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image
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
    model = load_model(LOCAL_MODEL_PATH)
    print(f"✅ Model đã tải xong và lưu tại {LOCAL_MODEL_PATH}")
except Exception as e:
    print("❌ Lỗi load model:", e)

# =============================
# Dummy user
# =============================
USERS = {"user_demo": "Test@123456"}

# =============================
# Routes
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
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
        return redirect(url_for("index"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

# =============================
# EMR CSV/Excel Analysis
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("index"))
    
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
    if "user" not in session:
        return redirect(url_for("index"))

    filename = None
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and model:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                # Mở ảnh, convert RGB, resize, normalize
                img = Image.open(file_path).convert("RGB")
                img = img.resize((224, 224))  # tùy theo input model
                x = np.array(img)/255.0
                x = np.expand_dims(x, axis=0)

                pred = model.predict(x)
                prediction = str(pred[0])  # hiển thị dạng chuỗi
            except Exception as e:
                flash(f"Lỗi khi dự đoán: {e}", "danger")
        elif not model:
            flash("Model chưa load được!", "danger")
    return render_template("emr_prediction.html", filename=filename, prediction=prediction)

# =============================
# Chạy Flask
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
