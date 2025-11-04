import os
import io
import base64
import requests
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ydata_profiling import ProfileReport

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = "emr_secret_key"

UPLOAD_FOLDER = "uploads"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# LOGGING
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------
# MODEL DOWNLOAD
# ---------------------------------------------------
MODEL_URL = "https://huggingface.co/minhtriizkooooo/EMR-Analysis-Cancer_Detection/resolve/main/models/best_weights_model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("🔽 Đang tải model từ Hugging Face...")
        try:
            response = requests.get(MODEL_URL, stream=True, timeout=180)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info("✅ Model tải thành công.")
        except Exception as e:
            logging.error(f"❌ Lỗi tải model: {e}")

download_model()

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = None
try:
    model = load_model(MODEL_PATH)
    logging.info("✅ Model nạp thành công.")
except Exception as e:
    logging.error(f"❌ Lỗi nạp model: {e}")

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("login"))

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("userID")
        password = request.form.get("password")

        if user_id == "user_demo" and password == "Test@123456":
            session["logged_in"] = True
            session["user"] = user_id  # 🔥 FIX QUAN TRỌNG
            flash("Đăng nhập thành công.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Sai ID hoặc mật khẩu!", "danger")
            return redirect(url_for("login"))

    return render_template("index.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# ---------------- EMR PROFILE ----------------
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":
        if "file" not in request.files:
            flash("Không tìm thấy tệp tải lên!", "danger")
            return redirect(url_for("emr_profile"))

        file = request.files["file"]
        if file.filename == "":
            flash("Vui lòng chọn tệp!", "warning")
            return redirect(url_for("emr_profile"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
            profile = ProfileReport(df, title="EMR Data Analysis", explorative=True)
            report_path = os.path.join("templates", "EMR_Profile.html")
            profile.to_file(report_path)
            logging.info(f"✅ EMR profile generated: {report_path}")
            return render_template("EMR_Profile.html")
        except Exception as e:
            logging.error(f"Lỗi xử lý EMR: {e}")
            flash(f"Lỗi xử lý tệp: {e}", "danger")
            return redirect(url_for("emr_profile"))

    return render_template("EMR_Profile.html")

# ---------------- IMAGE PREDICTION ----------------
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":
        if "file" not in request.files:
            flash("Không tìm thấy tệp ảnh!", "danger")
            return redirect(url_for("emr_prediction"))

        file = request.files["file"]
        if file.filename == "":
            flash("Vui lòng chọn ảnh!", "warning")
            return redirect(url_for("emr_prediction"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            image = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = "Nodule" if prediction[0][0] > 0.5 else "Non-Nodule"

            with open(filepath, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

            return render_template("EMR_Prediction.html", result=result, image_data=encoded_img)

        except Exception as e:
            logging.error(f"Lỗi dự đoán: {e}")
            flash(f"Lỗi dự đoán: {e}", "danger")
            return redirect(url_for("emr_prediction"))

    return render_template("EMR_Prediction.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("login"))

# ---------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
