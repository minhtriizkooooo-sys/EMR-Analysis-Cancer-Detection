import os
import io
import cv2
import base64
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from ydata_profiling import ProfileReport

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "emr_secret_key")

# --- Folder setup ---
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Model URL & Path ---
MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"
MODEL_PATH = os.path.join("models", "best_weights_model.keras")

# --- Download model if missing ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Hugging Face...")
        try:
            r = requests.get(MODEL_URL, stream=True, timeout=180)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
    else:
        print("✅ Model already exists locally.")

# --- Load AI model once ---
def load_ai_model():
    download_model()
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_ai_model()

# --- Demo User ---
USERS = {"user_demo": "Test@123456"}


# --- Routes ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login():
    user = request.form.get("userID", "").strip()
    pwd = request.form.get("password", "").strip()

    if user in USERS and USERS[user] == pwd:
        session["user"] = user
        flash("Đăng nhập thành công!", "success")
        return redirect(url_for("dashboard"))
    else:
        flash("Sai ID hoặc mật khẩu. Vui lòng thử lại.", "danger")
        return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", user=session["user"])


# --- EMR Profile Page (GET/POST) ---
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("emr_profile.html")

    # --- Handle file upload ---
    if "file" not in request.files:
        flash("Không tìm thấy file tải lên!", "danger")
        return redirect(url_for("emr_profile"))

    file = request.files["file"]
    if file.filename == "":
        flash("Chưa chọn file!", "danger")
        return redirect(url_for("emr_profile"))

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        profile = ProfileReport(df, title="Báo cáo Phân tích Hồ sơ Bệnh án EMR", minimal=True)
        report_path = os.path.join(app.config["UPLOAD_FOLDER"], "EMR_Profile_Report.html")
        profile.to_file(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            report_html = f.read()

        return render_template("emr_profile.html", report_html=report_html)

    except Exception as e:
        print(f"❌ EMR analysis error: {e}")
        flash("Lỗi khi xử lý file EMR!", "danger")
        return redirect(url_for("emr_profile"))


# --- EMR Prediction Page (GET/POST) ---
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("emr_prediction.html")

    # --- Handle image upload ---
    if "image" not in request.files:
        flash("Không tìm thấy file ảnh!", "danger")
        return redirect(url_for("emr_prediction"))

    file = request.files["image"]
    if file.filename == "":
        flash("Chưa chọn ảnh!", "danger")
        return redirect(url_for("emr_prediction"))

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # --- Image preprocessing (Giống Colab) ---
        image = cv2.imread(file_path)
        image = cv2.resize(image, (240, 240))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # --- Prediction ---
        prediction = model.predict(image)
        binary_prediction = np.round(prediction)
        label = "Nodule" if binary_prediction[0][0] == 1 else "Non-Nodule"

        # --- Convert image to base64 for web display ---
        with open(file_path, "rb") as img_f:
            img_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        return render_template(
            "emr_prediction.html",
            result=label,
            probability=float(prediction[0][0]),
            image_data=img_base64,
            filename=filename
        )

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        flash("Lỗi trong quá trình dự đoán!", "danger")
        return redirect(url_for("emr_prediction"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("index"))


# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
