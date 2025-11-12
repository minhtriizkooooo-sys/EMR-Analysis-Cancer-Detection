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
from huggingface_hub import hf_hub_download

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

# --- Upload Folder Setup ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")

# --- Model Loading ---
if not os.path.exists(MODEL_PATH):
    try:
        print("⚠️ Model not found locally, downloading from Hugging Face...")
        MODEL_PATH = hf_hub_download(
            repo_id="minhtriizkooooo/EMR-Analysis-Cancer-Detection",
            filename="models/best_weights_model.keras"
        )
        print(f"✅ Model downloaded successfully: {MODEL_PATH}")
    except Exception as e:
        raise FileNotFoundError(f"❌ Failed to download model from Hugging Face: {e}")

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")


# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")


# ============================
# 🔹 EMR Data Profiling Route
# ============================
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
                # 🔹 Cấu hình profiling nhẹ để tránh OOM trên Render
                profile = ProfileReport(
                    df,
                    title="Báo cáo Phân tích Dữ liệu EMR",
                    explorative=True,
                    minimal=True,
                    progress_bar=False,
                    correlations={"pearson": {"calculate": False}},
                    interactions={"continuous": False},
                    missing_diagrams={"heatmap": False, "dendrogram": False},
                    duplicates={"head": 0}
                )

                # 🔹 Xuất file HTML nhẹ, tránh to_file() gây lỗi bộ nhớ
                report_file = os.path.join(UPLOAD_FOLDER, f"{filename}_report.html")
                html = profile.to_html()
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(html)

                return send_file(report_file)
            except Exception as e:
                flash(f"Lỗi khi tạo báo cáo: {e}", "danger")
        else:
            flash("Vui lòng chọn file CSV hoặc Excel", "warning")

    return render_template("emr_profile.html")


# ============================
# 🔹 EMR Prediction (AI Model)
# ============================
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

                # 🔹 Dự đoán an toàn (thêm timeout fallback)
                prob = float(model.predict(image_array, verbose=0)[0][0])
                result = "Nodule" if prob > 0.5 else "Non-nodule"
                prediction = {"result": result, "probability": prob}

                # Encode ảnh để hiển thị lại
                with open(filepath, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as e:
                flash(f"Lỗi dự đoán hình ảnh: {e}", "danger")
        else:
            flash("Vui lòng chọn hình ảnh để dự đoán", "warning")

    return render_template(
        "emr_prediction.html",
        prediction=prediction,
        filename=filename,
        image_b64=image_b64
    )


# ============================
# 🔹 Login / Logout
# ============================
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
