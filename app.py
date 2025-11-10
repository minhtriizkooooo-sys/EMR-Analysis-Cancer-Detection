import os
import io
import base64
import tempfile
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ydata_profiling import ProfileReport

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# --- Model Loading ---
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")

# Nếu chỉ có các phần .keras.001-.004, ghép lại
if not os.path.exists(MODEL_PATH):
    parts = sorted([f for f in os.listdir(MODEL_FOLDER) if f.startswith("best_weights_model.keras.")])
    if parts:
        with open(MODEL_PATH, "wb") as outfile:
            for part in parts:
                with open(os.path.join(MODEL_FOLDER, part), "rb") as infile:
                    outfile.write(infile.read())
        print(f"✅ Model ghép thành công: {MODEL_PATH}")
    else:
        print("⚠️ Model file missing! Vui lòng upload model parts hoặc file .keras")
        # Raise an exception để Render không khởi chạy khi thiếu model
        raise FileNotFoundError("Model file not found. Please upload model parts.")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    profile_report = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
            profile = ProfileReport(df, title="Báo cáo Phân tích Dữ liệu EMR", explorative=True)
            report_file = os.path.join(UPLOAD_FOLDER, f"{filename}_report.html")
            profile.to_file(report_file)
            return send_file(report_file)
        else:
            flash("Vui lòng chọn file CSV hoặc Excel", "warning")

    return render_template("EMR_Profile.html")

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

            # Đọc ảnh và resize về 240x240
            image = load_img(filepath, target_size=(240, 240))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Dự đoán
            prob = model.predict(image_array)[0][0]
            result = "Nodule" if prob > 0.5 else "Non-nodule"
            prediction = {"result": result, "probability": float(prob)}

            # Chuyển ảnh sang base64 để hiển thị trên web
            with open(filepath, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        else:
            flash("Vui lòng chọn hình ảnh để dự đoán", "warning")

    return render_template("EMR_Prediction.html",
                           prediction=prediction,
                           filename=filename,
                           image_b64=image_b64)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # TODO: Replace with real user check
        if username == "admin" and password == "admin":
            session["user"] = username
            return redirect(url_for("dashboard"))
        flash("Sai tên đăng nhập hoặc mật khẩu", "danger")
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
