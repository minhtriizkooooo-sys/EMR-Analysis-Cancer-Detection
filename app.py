import os
import io
import base64
import logging
import tempfile
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from ydata_profiling import ProfileReport
import cv2

# ================================
# --- Flask Setup ---
# ================================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "emr_secret_key")

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_CSV_EXTENSIONS = {"csv", "xlsx"}

# Folder setup (Render-safe)
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================================
# --- Model Loading ---
# ================================
MODEL_PATH = "models/best_weights_model.keras"
model = None

def download_and_load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print("🚀 Model file not found — please upload or fetch it.")
    else:
        try:
            model = load_model(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

download_and_load_model()

# ================================
# --- Utility ---
# ================================
def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# ================================
# --- Routes ---
# ================================
@app.route("/")
def index():
    return render_template("index.html")


# --- LOGIN / LOGOUT ---
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    if userID== "user_demo" and password == "Test@123456":
        session["user"] = userID
        return redirect(url_for("dashboard"))
    else:
        flash("Sai tài khoản hoặc mật khẩu!")
        return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    #flash("Đã đăng xuất thành công.")
    return redirect(url_for("index"))


# --- DASHBOARD ---
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")


# ================================
# --- EMR Profiling ---
# ================================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            flash("Vui lòng chọn file CSV hoặc Excel để tải lên.")
            return redirect(request.url)

        if not allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
            flash("Định dạng file không hợp lệ.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        try:
            # Load dataset
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            # Quick profiling to avoid Render timeout
            profile = ProfileReport(df, explorative=True, minimal=True)
            report_html = profile.to_html()

            # Save to file (optional)
            output_html = os.path.join("uploads", "EMR_Profile.html")
            with open(output_html, "w", encoding="utf-8") as f:
                f.write(report_html)

            return render_template("EMR_Profile.html", report_html=report_html)

        except Exception as e:
            flash(f"Lỗi khi xử lý dữ liệu: {e}")
            return redirect(request.url)

    return render_template("emr_profile.html", report_html=None)


# ================================
# --- EMR Image Prediction ---
# ================================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            flash("Không tìm thấy file ảnh!")
            return redirect(request.url)

        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash("Định dạng ảnh không hợp lệ!")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        try:
            # === Colab-style preprocessing ===
            image = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prediction
            if model is None:
                flash("Model chưa được tải. Kiểm tra lại file model.")
                return redirect(request.url)

            prediction = model.predict(img_array)[0][0]
            label = "Nodule" if prediction >= 0.5 else "Non-Nodule"
            confidence = float(prediction) if prediction >= 0.5 else 1 - float(prediction)
            confidence = round(confidence * 100, 2)

            # Visualization (optional OpenCV)
            img_cv = cv2.imread(filepath)
            img_cv = cv2.putText(
                img_cv, f"{label} ({confidence}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Nodule" else (255, 0, 0), 2
            )
            _, buffer = cv2.imencode(".png", img_cv)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            return render_template(
                "EMR_Prediction.html",
                filename=filename,
                label=label,
                confidence=confidence,
                img_data=img_base64
            )

        except Exception as e:
            flash(f"Lỗi khi dự đoán ảnh: {e}")
            return redirect(request.url)

    return render_template("EMR_Prediction.html", filename=None)


# ================================
# --- App Runner ---
# ================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
