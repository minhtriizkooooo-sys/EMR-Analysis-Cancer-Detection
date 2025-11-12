import os
import io
import base64
import tempfile
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ydata_profiling import ProfileReport

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

# --- Upload Folder Setup ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")

# --- Hugging Face Space Model URL ---
HF_SPACE_MODEL_URL = (
    "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"
)

# --- Model Loading ---
if not os.path.exists(MODEL_PATH):
    try:
        print("⚠️ Model not found locally, downloading from Hugging Face Space...")
        response = requests.get(HF_SPACE_MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Model downloaded successfully: {MODEL_PATH}")
    except Exception as e:
        raise FileNotFoundError(f"❌ Failed to download model from Hugging Face Space: {e}")

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")


# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
            except Exception as e:
                flash(f"Lỗi đọc file: {e}", "danger")
                return redirect(url_for("emr_profile"))

            try:
                profile = ProfileReport(df, title="Báo cáo Phân tích Dữ liệu EMR", explorative=True)
                report_file = os.path.join(UPLOAD_FOLDER, f"{filename}_report.html")
                profile.to_file(report_file)
                return send_file(report_file)
            except Exception as e:
                flash(f"Lỗi khi tạo báo cáo: {e}", "danger")
        else:
            flash("Vui lòng chọn file CSV hoặc Excel", "warning")

    return render_template("emr_profile.html")

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))
    
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                image = load_img(filepath, target_size=(240, 240))
                image_array = img_to_array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                prob = model.predict(image_array)[0][0]
                result = "Nodule" if prob > 0.5 else "Non-nodule"
                prediction = {"result": result, "probability": float(prob)}

                with open(filepath, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as e:
                flash(f"Lỗi dự đoán hình ảnh: {e}", "danger")
        else:
            flash("Vui lòng chọn hình ảnh để dự đoán", "warning")

    return render_template("emr_prediction.html",
                           prediction=prediction,
                           filename=filename,
                           image_b64=image_b64)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        userID = request.form.get("userID")
        password = request.form.get("password")

        # Demo login
        if userID == "user_demo" and password == "Test@123456":
            session["user"] = userID
            return redirect(url_for("dashboard"))
        else:
            flash("Sai tên đăng nhập hoặc mật khẩu", "danger")

    return render_template("index.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Đã đăng xuất thành công.", "info")
    return redirect(url_for("login"))

# --- Render Compatible Entrypoint ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
