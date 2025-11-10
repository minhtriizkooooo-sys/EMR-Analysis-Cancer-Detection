import os
import io
import base64
import logging
import tempfile
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ydata_profiling import ProfileReport
import cv2

# ================================
# --- Flask Setup ---
# ================================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", "emr_secret_key")

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_CSV_EXTENSIONS = {"csv", "xlsx"}

os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================================
# --- Model Handling ---
# ================================
MODEL_PATH = "models/best_weights_model.keras"
HF_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"
model = None


def download_and_load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print("🔽 Model not found — downloading from Hugging Face...")
        try:
            r = requests.get(HF_URL, stream=True, timeout=30)
            if r.status_code == 200:
                content = r.content
                if len(content) < 1_000_000:
                    print("⚠️ Model file nhỏ bất thường — có thể tải lỗi.")
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
                print("✅ Model downloaded successfully.")
            else:
                print(f"❌ Failed to download model: {r.status_code}")
        except Exception as e:
            print(f"⚠️ Error downloading model: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Model load failed: {e}")


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
    userID = request.form.get("userID")
    password = request.form.get("password")

    if userID == "user_demo" and password == "Test@123456":
        session["user"] = userID
        return redirect(url_for("dashboard"))
    flash("Sai tài khoản hoặc mật khẩu!", "danger")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
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
    if "user" not in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Vui lòng chọn file CSV hoặc Excel để tải lên.", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
            flash("Định dạng file không hợp lệ.", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            profile = ProfileReport(df, minimal=True, explorative=True)
            report_html = profile.to_html()

            return render_template(
                "emr_profile.html",
                summary=report_html,
                filename=filename,
                session=session,
            )
        except Exception as e:
            flash(f"Lỗi khi xử lý dữ liệu: {e}", "danger")
            return redirect(request.url)

    return render_template("emr_profile.html", summary=None, filename=None, session=session)


# ================================
# --- EMR Image Prediction ---
# ================================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Không tìm thấy file ảnh!", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash("Định dạng ảnh không hợp lệ!", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        try:
            img = cv2.imread(filepath)
            img = cv2.resize(img, (240, 240))
            img = np.expand_dims(img, axis=0)

            if model is None:
                flash("Model chưa được tải. Kiểm tra lại file model.", "danger")
                return redirect(request.url)

            preds = model.predict(img)
            prob = float(preds[0][0])
            binary_prediction = np.round(prob)
            result = "Nodule" if binary_prediction == 1 else "Non-Nodule"
            confidence = round(prob * 100, 2) if result == "Nodule" else round((1 - prob) * 100, 2)

            img_cv = cv2.imread(filepath)
            color = (0, 255, 0) if result == "Nodule" else (255, 0, 0)
            img_cv = cv2.putText(
                img_cv, f"{result} ({confidence}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )
            _, buffer = cv2.imencode(".png", img_cv)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            return render_template(
                "emr_prediction.html",
                filename=filename,
                result=result,
                confidence=confidence,
                img_data=img_base64,
                session=session,
            )

        except Exception as e:
            flash(f"Lỗi khi dự đoán ảnh: {e}", "danger")
            return redirect(request.url)

    return render_template("emr_prediction.html", session=session)


# ================================
# --- Runner ---
# ================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
