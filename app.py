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
import traceback

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
    print("⏳ Đang tải model từ Hugging Face…")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, repo_type="model")
    model = load_model(model_path)
    print("✅ Model đã được tải và load thành công.")
except Exception as e:
    print("❌ Không thể tải hoặc load model từ HF:")
    traceback.print_exc()
    model = None

# =============================
# Helper
# =============================
def allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext


def analyze_emr_data(df: pd.DataFrame):
    """Phân tích hồ sơ EMR chuyên sâu, trả về kết quả tổng quan và nhận định."""
    results = {}

    # Thông tin tổng quan
    results["Số bệnh nhân"] = len(df)
    if "Tuổi" in df.columns:
        results["Tuổi trung bình"] = round(df["Tuổi"].mean(), 1)
    if "Giới tính" in df.columns:
        gender_dist = df["Giới tính"].value_counts(normalize=True) * 100
        results["Tỷ lệ giới tính"] = f"Nữ: {gender_dist.get('Nữ', 0):.1f}% | Nam: {gender_dist.get('Nam', 0):.1f}%"

    # Phân tích chỉ số lâm sàng nếu có
    health_notes = []
    if "Glucose" in df.columns:
        avg_glucose = df["Glucose"].mean()
        if avg_glucose > 126:
            health_notes.append(f"⚠️ Mức Glucose trung bình cao ({avg_glucose:.1f}) → Nguy cơ tiểu đường.")
        else:
            health_notes.append(f"✅ Mức Glucose trung bình ổn định ({avg_glucose:.1f}).")

    if "Cholesterol" in df.columns:
        avg_chol = df["Cholesterol"].mean()
        if avg_chol > 200:
            health_notes.append(f"⚠️ Cholesterol trung bình cao ({avg_chol:.1f}) → Nguy cơ tim mạch.")
        else:
            health_notes.append(f"✅ Cholesterol trung bình bình thường ({avg_chol:.1f}).")

    if not health_notes:
        health_notes.append("Không đủ dữ liệu để đánh giá chỉ số sinh học.")

    # Gợi ý y học
    if len(health_notes) >= 2 and any("⚠️" in note for note in health_notes):
        results["Kết luận sơ bộ"] = "⚠️ Hồ sơ có một số dấu hiệu bất thường, cần kiểm tra chuyên sâu."
    else:
        results["Kết luận sơ bộ"] = "✅ Hồ sơ sức khỏe tổng quan bình thường."

    results["Nhận định chi tiết"] = "<br>".join(health_notes)

    # Thống kê cơ bản dạng HTML
    summary_html = df.describe(include="all").to_html(
        classes="table-auto w-full border-collapse border border-gray-300",
        border=0
    )

    return results, summary_html


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
# EMR PROFILE (phân tích file CSV/Excel)
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này.", "danger")
        return redirect(url_for("login"))

    filename = None
    summary_html = None
    analysis = None

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

            analysis, summary_html = analyze_emr_data(df)

        except Exception as e:
            flash(f"Lỗi khi phân tích EMR: {e}", "danger")

    return render_template("emr_profile.html",
                           filename=filename,
                           analysis=analysis,
                           summary=summary_html)


# =============================
# EMR PREDICTION (phân tích hình ảnh)
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

        # Base64 hiển thị ảnh trong HTML
        with open(filepath, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        if model is None:
            flash("Model chưa được tải thành công!", "danger")
            return render_template("emr_prediction.html", filename=filename, image_b64=image_b64)

        try:
            img = Image.open(filepath).convert("RGB").resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr, verbose=0)
            prob = float(preds[0][0])
            label = "Ung thư" if prob >= 0.5 else "Lành tính"

            prediction = {
                "Kết quả": label,
                "Xác suất": f"{prob*100:.2f}%",
                "Nhận định": "⚠️ Dấu hiệu nghi ngờ ác tính, cần kiểm tra sâu hơn." if prob >= 0.5 else "✅ Không phát hiện bất thường rõ ràng."
            }

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
