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
# Đã giữ nguyên key mặc định như trong code gốc
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
        # Giữ nguyên logic lỗi nặng
        raise FileNotFoundError(f"❌ Failed to download model from Hugging Face Space: {e}")

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    # Giữ nguyên logic lỗi nặng
    raise RuntimeError(f"❌ Failed to load model: {e}")


# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    # Truyền cổng vào template để hiển thị trạng thái
    server_port = os.environ.get("PORT", 5000)
    return render_template("dashboard.html", server_port=server_port)

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    # Khởi tạo profile_html
    profile_html = None
    filename = None
    
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filepath)
                else:
                    # Hỗ trợ Excel (.xls, .xlsx)
                    df = pd.read_excel(filepath)
            except Exception as e:
                flash(f"Lỗi đọc file: {e}", "danger")
                return redirect(url_for("emr_profile"))

            try:
                # Dùng to_html() thay vì to_file() để hiển thị trực tiếp trong trang
                profile = ProfileReport(df, title=f"Báo cáo Phân tích Dữ liệu: {filename}", explorative=True)
                profile_html = profile.to_html()
                
                flash(f"✅ Đã tạo báo cáo phân tích dữ liệu EMR cho file '{filename}' thành công!", "success")
                # Xóa file dữ liệu sau khi xử lý (tùy chọn)
                # os.remove(filepath)

            except Exception as e:
                flash(f"❌ Lỗi khi tạo báo cáo: {e}", "danger")
        else:
            flash("⚠️ Vui lòng chọn file CSV hoặc Excel", "warning")

    return render_template("emr_profile.html", 
                           profile_html=profile_html, # Đã sửa: summary -> profile_html
                           filename=filename)

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))
    
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                # Resize ảnh về kích thước mô hình mong muốn
                image = load_img(filepath, target_size=(240, 240))
                image_array = img_to_array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                # Thực hiện dự đoán
                prob = model.predict(image_array)[0][0]
                # Quyết định kết quả (có thể điều chỉnh ngưỡng 0.5)
                result = "Nodule" if prob > 0.5 else "Non-nodule"
                prediction = {"result": result, "probability": float(prob)}

                # Đọc ảnh gốc về base64 để hiển thị
                with open(filepath, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                
                #flash(f"✅ Dự đoán hoàn tất. Kết quả: {result} ({prob*100:.2f}%)", "success")

            except Exception as e:
                flash(f"❌ Lỗi dự đoán hình ảnh. Đảm bảo đây là file hình ảnh hợp lệ. Chi tiết lỗi: {e}", "danger")
            
            # Xóa file ảnh sau khi xử lý (tùy chọn)
            # os.remove(filepath)

        else:
            flash("⚠️ Vui lòng chọn hình ảnh để dự đoán", "warning")

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
            #flash("Chào mừng, Đăng nhập thành công!", "success")
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
