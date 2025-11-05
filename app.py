import os
import io
import base64
import requests
import tempfile
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ydata_profiling import ProfileReport

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = "emr_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model Configuration ---
MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"
MODEL_PATH = os.path.join("models", "best_weights_model.keras")
os.makedirs("models", exist_ok=True)

# --- Download model from HF if not cached ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded successfully.")
    else:
        print("✅ Model already cached.")

# --- Load model safely ---
def load_ai_model():
    download_model()
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

model = load_ai_model()

# --- Demo Login ---
USERS = {
    "user_demo": "Test@123456"
}

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    user = request.form.get("userID")
    pwd = request.form.get("password")
    if user in USERS and USERS[user] == pwd:
        session["user"] = user
        return redirect(url_for("dashboard"))
    flash("Sai ID hoặc mật khẩu. Vui lòng thử lại.", "danger")
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", user=session["user"])

# --- EMR Profile Analysis (CSV) ---
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "file" not in request.files:
        flash("Không tìm thấy file tải lên!", "danger")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("Chưa chọn file!", "danger")
        return redirect(url_for("dashboard"))

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
        return redirect(url_for("dashboard"))

# --- Image Prediction (Nodule/Non-Nodule) ---
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "image" not in request.files:
        flash("Không tìm thấy file ảnh!", "danger")
        return redirect(url_for("dashboard"))

    file = request.files["image"]
    if file.filename == "":
        flash("Chưa chọn ảnh!", "danger")
        return redirect(url_for("dashboard"))

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result = "Nodule" if prediction[0][0] > 0.5 else "Non-Nodule"

        # Convert image to base64 for HTML preview
        with open(file_path, "rb") as img_f:
            img_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        return render_template("emr_prediction.html", result=result, image_data=img_base64)

    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        flash("Lỗi trong quá trình dự đoán ảnh.", "danger")
        return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

# --- Run app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


