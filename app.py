from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import secrets
import numpy as np
import pandas as pd
from PIL import Image
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import base64

# =============================
# Flask config
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
ALLOWED_DATA = {'csv', 'xls', 'xlsx'}

# =============================
# Load model Keras từ Hugging Face
# =============================
MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
MODEL_FILENAME = "best_weights_model.keras"  # Đặt đúng tên file trong repo của bạn

model = None
model_input_shape = (240, 240, 3)  # default fallback

try:
    print("⏳ Đang tải model từ Hugging Face…")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = load_model(model_path)
    model_input_shape = model.input_shape[1:]  # bỏ batch dim
    print(f"✅ Model đã load với input_shape: {model_input_shape}")
except Exception as e:
    print(f"❌ Không thể tải hoặc load model từ HF: {e}")
    model = None

# =============================
# Helpers
# =============================
def allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext

# =============================
# Login
# =============================
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("userID")
        password = request.form.get("password")
        if user_id == "user_demo" and password == "Test@123456":
            session["user"] = user_id
            return redirect(url_for("dashboard"))
        else:
            flash("Sai thông tin đăng nhập!", "danger")
            return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Bạn đã đăng xuất.", "danger")
    return redirect(url_for("login"))

# =============================
# Dashboard
# =============================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Bạn cần đăng nhập để tiếp tục.", "danger")
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# =============================
# EMR Profile
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này.", "danger")
        return redirect(url_for("login"))

    summary_html = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Chưa chọn file để tải lên!", "danger")
            return redirect(request.url)
        if not allowed_file(file.filename, ALLOWED_DATA):
            flash("Chỉ hỗ trợ định dạng .csv, .xls, .xlsx!", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            df = df.loc[:, df.notna().any()]
            summary_html = df.describe(include="all").to_html(
                classes="table-auto w-full border-collapse border border-gray-300",
                border=0
            )
        except Exception as e:
            flash(f"Lỗi khi đọc file: {e}", "danger")

    return render_template("emr_profile.html", summary=summary_html, filename=filename)

# =============================
# EMR Prediction
# =============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này.", "danger")
        return redirect(url_for("login"))

    filename = None
    image_b64 = None
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Chưa chọn hình ảnh!", "danger")
            return redirect(request.url)
        if not allowed_file(file.filename, ALLOWED_IMG):
            flash("Định dạng ảnh không hợp lệ!", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Base64 để hiển thị ảnh trong HTML
        with open(filepath, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        if model is None:
            flash("Model chưa được tải thành công!", "danger")
            return render_template("emr_prediction.html", filename=filename, image_b64=image_b64)

        try:
            # Load ảnh và resize về input của model
            img = Image.open(filepath).convert("L")  # đọc grayscale
            w, h = model_input_shape[1], model_input_shape[0]
            img = img.resize((w, h))
            arr = np.array(img)

            # Convert grayscale -> 3 channel
            arr = np.stack((arr,) * 3, axis=-1)
            arr = arr / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr, verbose=0)
            prob = float(preds[0][0])
            label = "Nodule" if prob >= 0.5 else "Non-nodule"

            prediction = {"result": label, "probability": round(prob, 4)}

        except Exception as e:
            flash(f"Lỗi khi dự đoán ảnh: {e}", "danger")

    return render_template("emr_prediction.html",
                           filename=filename,
                           image_b64=image_b64,
                           prediction=prediction)

# =============================
# Run app
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
