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
import matplotlib.pyplot as plt

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
# Load model from Hugging Face
# =============================
MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
MODEL_FILENAME = "best_weights_model.keras"

model = None
try:
    print("⏳ Đang tải model thật từ Hugging Face…")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = load_model(model_path)
    print("✅ Model thật đã được tải và load thành công.")
except Exception as e:
    print(f"❌ Không thể tải hoặc load model từ HF: {e}")
    model = None

# =============================
# Helper
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
# EMR PROFILE
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này.", "danger")
        return redirect(url_for("login"))

    summary_html = None
    filename = None
    chart_path = None

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
            # Đọc dữ liệu
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            total_rows, total_cols = df.shape

            # Loại cột >80% NA
            missing_ratio = df.isnull().mean()
            df_clean = df.loc[:, missing_ratio < 0.8]

            # Thống kê tổng quan
            info_table = pd.DataFrame({
                "Chỉ số": ["Số hàng", "Số cột", "Số cột bị loại (>80% NA)"],
                "Giá trị": [total_rows, total_cols, total_cols - df_clean.shape[1]]
            })

            # Top 5 cột thiếu dữ liệu
            top_missing = missing_ratio.sort_values(ascending=False).head(5) * 100
            top_missing_table = pd.DataFrame({
                "Cột": top_missing.index,
                "Tỷ lệ thiếu dữ liệu (%)": top_missing.values
            })

            # Mô tả dữ liệu số & phân loại
            desc_num = df_clean.describe().transpose()
            desc_cat = df_clean.describe(include="object").transpose()

            # Biểu đồ histogram
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.figure(figsize=(8, 4))
                df_clean[numeric_cols[:min(2, len(numeric_cols))]].hist(bins=20, figsize=(8, 4))
                plt.tight_layout()
                chart_path = os.path.join(app.config["UPLOAD_FOLDER"], "analysis.png")
                plt.savefig(chart_path)
                plt.close()

            # HTML tổng hợp
            summary_html = (
                "<h3>📊 Tổng quan dữ liệu</h3>" +
                info_table.to_html(classes="table-auto border border-gray-300", index=False, border=0) +
                "<h3>⚠️ Top 5 cột có nhiều giá trị trống</h3>" +
                top_missing_table.to_html(classes="table-auto border border-gray-300", index=False, border=0) +
                "<h3>🔢 Mô tả dữ liệu số</h3>" +
                desc_num.to_html(classes="table-auto border border-gray-300", border=0) +
                "<h3>🔠 Mô tả dữ liệu phân loại</h3>" +
                desc_cat.to_html(classes="table-auto border border-gray-300", border=0)
            )

        except Exception as e:
            flash(f"Lỗi khi đọc hoặc phân tích file: {e}", "danger")

    return render_template("emr_profile.html", summary=summary_html, filename=filename, chart=chart_path)

# =============================
# EMR PREDICTION
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

        # Base64 hiển thị ảnh
        with open(filepath, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        if model is None:
            flash("Model chưa được tải thành công!", "danger")
            return render_template("emr_prediction.html", filename=filename, image_b64=image_b64)

        try:
            img = Image.open(filepath).convert("RGB")  # chấp nhận gray -> RGB
            img = img.resize((224, 224))
            arr = np.array(img) / 255.0
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
# Run
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
