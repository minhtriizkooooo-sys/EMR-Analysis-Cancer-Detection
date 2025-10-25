from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import secrets
import requests
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# =============================
# Thư mục upload & model
# =============================
UPLOAD_FOLDER = "static/uploads"
MODEL_DIR = "model"
MODEL_FILE = "best_weights_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# Link model trên Hugging Face
# =============================
MODEL_URL = (
    "https://huggingface.co/minhtriizkooooo/EMR-Analysis-Cancer_Detection/"
    "resolve/main/best_weights_model.keras"
)

# =============================
# Tải model nếu chưa có
# =============================
if not os.path.exists(MODEL_PATH):
    print("🔽 Đang tải model từ Hugging Face...")
    try:
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Tải model thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")

# =============================
# Load model
# =============================
model = None
try:
    model = load_model(MODEL_PATH)
    print("✅ Model đã load thành công.")
except Exception as e:
    print(f"❌ Không thể load model: {e}")

# =============================
# Route: Đăng nhập
# =============================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("userID")
        password = request.form.get("password")

        if user == "user_demo" and password == "Test@123456":
            session["user"] = user
            flash("Đăng nhập thành công!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Sai ID người dùng hoặc mật khẩu!", "danger")
            return redirect(url_for("login"))
    return render_template("index.html")


# =============================
# Route: Dashboard
# =============================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Vui lòng đăng nhập trước.", "danger")
        return redirect(url_for("login"))
    return render_template("dashboard.html")


# =============================
# Route: Phân tích EMR (CSV/Excel)
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        flash("Vui lòng đăng nhập trước.", "danger")
        return redirect(url_for("login"))

    summary = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("Vui lòng chọn file CSV hoặc Excel!", "danger")
            return redirect(url_for("emr_profile"))

        filename = file.filename
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            summary_html = f"""
            <h3 class='text-xl font-semibold text-primary-green mb-3'>📊 Tổng quan dữ liệu</h3>
            <p><b>Số dòng:</b> {df.shape[0]} | <b>Số cột:</b> {df.shape[1]}</p>
            <h4 class='text-lg font-semibold mt-4 mb-2'>Mẫu dữ liệu:</h4>
            {df.head().to_html(classes='table-auto w-full border border-gray-300 rounded-lg shadow-sm', index=False)}
            <h4 class='text-lg font-semibold mt-4 mb-2'>Mô tả thống kê:</h4>
            {df.describe().to_html(classes='table-auto w-full border border-gray-300 rounded-lg shadow-sm')}
            """
            summary = summary_html
        except Exception as e:
            flash(f"Lỗi khi đọc file: {e}", "danger")

    return render_template("emr_profile.html", summary=summary, filename=filename)


# =============================
# Route: Dự đoán ảnh y tế (AI)
# =============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        flash("Vui lòng đăng nhập trước.", "danger")
        return redirect(url_for("login"))

    result = None
    image_path = None

    if request.method == "POST":
        if model is None:
            flash("Model chưa sẵn sàng!", "danger")
            return redirect(url_for("emr_prediction"))

        file = request.files.get("file")
        if not file:
            flash("Vui lòng chọn ảnh để dự đoán!", "danger")
            return redirect(url_for("emr_prediction"))

        try:
            filename = file.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            # Đọc và xử lý ảnh
            img = Image.open(image_path)

            # Nếu grayscale thì chuyển sang RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Dự đoán
            prediction = model.predict(img_array)
            prob = float(prediction[0][0])
            label = "Nodule" if prob >= 0.5 else "Non-Nodule"
            confidence = prob if label == "Nodule" else 1 - prob
            result = f"Kết quả: {label} ({confidence * 100:.2f}% tin cậy)"

            flash("✅ Phân tích hình ảnh thành công!", "success")

        except Exception as e:
            flash(f"Lỗi khi dự đoán ảnh: {e}", "danger")

    return render_template("emr_prediction.html", result=result, image_path=image_path)


# =============================
# Route: Logout
# =============================
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("login"))


# =============================
# Run app
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
