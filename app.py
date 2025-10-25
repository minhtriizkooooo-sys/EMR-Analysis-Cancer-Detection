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
import io
import matplotlib.pyplot as plt

# =============================
# Flask Config
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
ALLOWED_DATA = {'csv', 'xls', 'xlsx'}

# =============================
# Load Model from Hugging Face
# =============================
MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
MODEL_FILENAME = "best_weights_model.keras"

model = None
try:
    print("⏳ Đang tải model thật từ Hugging Face...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = load_model(model_path)
    print("✅ Model thật đã được tải và load thành công.")
except Exception as e:
    print(f"❌ Không thể tải hoặc load model thật từ HF: {e}")
    model = None

# =============================
# Helper
# =============================
def allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext

def df_analysis_summary(df):
    """Tạo bản phân tích chuyên sâu từ dữ liệu EMR"""
    summary = df.describe(include="all").T
    null_count = df.isnull().sum()
    summary["Missing Values"] = null_count
    summary_html = summary.to_html(classes="table table-striped table-bordered", border=0)

    # Nhận xét AI cơ bản
    remarks = []
    if df.shape[0] < 10:
        remarks.append("⚠️ Số lượng bản ghi ít, khó phân tích xu hướng.")
    if (null_count > 0).any():
        remarks.append("⚠️ Dữ liệu còn thiếu, cần làm sạch trước khi huấn luyện AI.")
    if "age" in df.columns and df["age"].mean() > 60:
        remarks.append("🧓 Dữ liệu bệnh nhân chủ yếu là người cao tuổi — nguy cơ ung thư cao hơn trung bình.")
    if "smoking" in df.columns and df["smoking"].mean() > 0.5:
        remarks.append("🚬 Tỷ lệ hút thuốc cao — yếu tố rủi ro chính cần chú ý.")
    if not remarks:
        remarks.append("✅ Dữ liệu đạt yêu cầu cho bước phân tích tiếp theo.")

    return summary_html, remarks

def create_correlation_plot(df):
    """Tạo heatmap tương quan dạng hình ảnh"""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(corr, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="left", fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.colorbar(cax)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

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
# EMR PROFILE (phân tích chuyên sâu CSV/Excel)
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này.", "danger")
        return redirect(url_for("login"))

    summary_html, remarks, corr_b64, filename = None, [], None, None

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

            summary_html, remarks = df_analysis_summary(df)
            corr_b64 = create_correlation_plot(df)

        except Exception as e:
            flash(f"Lỗi khi đọc hoặc phân tích file: {e}", "danger")

    return render_template("emr_profile.html",
                           summary=summary_html,
                           remarks=remarks,
                           corr_b64=corr_b64,
                           filename=filename)

# =============================
# EMR PREDICTION (phân tích ảnh thật)
# =============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này.", "danger")
        return redirect(url_for("login"))

    filename, image_b64, prediction = None, None, None

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

        with open(filepath, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        if model is None:
            flash("❌ Model thật chưa được tải thành công từ Hugging Face!", "danger")
            return render_template("emr_prediction.html", filename=filename, image_b64=image_b64)

        try:
            img = Image.open(filepath).convert("RGB").resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr, verbose=0)
            prob = float(preds[0][0])
            label = "Ung thư (Nodule)" if prob >= 0.5 else "Không ung thư (Non-nodule)"

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
