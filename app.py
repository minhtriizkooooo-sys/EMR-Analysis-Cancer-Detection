import os
import io
import base64
import requests
import tempfile
import numpy as np
import pandas as pd
import cv2
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ydata_profiling import ProfileReport

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "emr_secret_key")

# --- Folder setup ---
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Model configuration ---
MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"
MODEL_PATH = os.path.join("models", "best_weights_model.keras")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL, stream=True, timeout=180)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded successfully.")
    else:
        print("✅ Model already cached.")

def load_ai_model():
    download_model()
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# --- Load model ---
model = load_ai_model()

# --- Demo login ---
USERS = {"user_demo": "Test@123456"}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    user = request.form.get("userID", "").strip()
    pwd = request.form.get("password", "").strip()

    if user in USERS and USERS[user] == pwd:
        session["user"] = user
        print(f"✅ Login success: {user}")
        return redirect(url_for("dashboard"))
    else:
        flash("Sai ID hoặc mật khẩu. Vui lòng thử lại.", "danger")
        return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", user=session["user"])

# --- EMR PROFILE PAGE (GET) ---
@app.route("/emr_profile", methods=["GET"])
def emr_profile_page():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("emr_profile.html")

# --- EMR PROFILE UPLOAD (POST) ---
@app.route("/emr_profile", methods=["POST"])
def emr_profile_upload():
    if "user" not in session:
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("Không tìm thấy file tải lên!", "danger")
        return redirect(url_for("emr_profile_page"))

    file = request.files["file"]
    if file.filename == "":
        flash("Chưa chọn file!", "danger")
        return redirect(url_for("emr_profile_page"))

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        df = pd.read_csv(file_path) if filename.endswith(".csv") else pd.read_excel(file_path)
        profile = ProfileReport(df, title="Báo cáo Phân tích Hồ sơ Bệnh án EMR", minimal=True)
        report_path = os.path.join(app.config["UPLOAD_FOLDER"], "EMR_Profile_Report.html")
        profile.to_file(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            report_html = f.read()

        return render_template("emr_profile.html", report_html=report_html)

    except Exception as e:
        print(f"❌ Error analyzing EMR: {e}")
        flash("Lỗi xử lý file EMR. Kiểm tra lại định dạng!", "danger")
        return redirect(url_for("emr_profile_page"))

# --- EMR PREDICTION PAGE (GET) ---
@app.route("/emr_prediction", methods=["GET"])
def emr_prediction_page():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("emr_prediction.html")

# --- EMR PREDICTION UPLOAD (POST) ---
@app.route("/emr_prediction", methods=["POST"])
def emr_prediction_upload():
    if "user" not in session:
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("Không tìm thấy file ảnh!", "danger")
        return redirect(url_for("emr_prediction_page"))

    file = request.files["image"]
    if file.filename == "":
        flash("Chưa chọn ảnh!", "danger")
        return redirect(url_for("emr_prediction_page"))

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Load & preprocess image
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        prob = float(prediction[0][0])
        result = "Nodule" if prob > 0.5 else "Non-Nodule"

        # Convert to base64 for HTML display
        with open(file_path, "rb") as img_f:
            img_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        return render_template("emr_prediction.html", result=result, image_data=img_base64, prob=prob)

    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        flash("Lỗi trong quá trình dự đoán ảnh.", "danger")
        return redirect(url_for("emr_prediction_page"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Đã đăng xuất thành công.", "info")
    return redirect(url_for("index"))

# --- Run app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
