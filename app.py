import os
import io
import base64
import tempfile
import numpy as np
import pandas as pd
import requests
import time
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ydata_profiling import ProfileReport
from functools import wraps

# --- Cấu hình Timeout Tối Đa (Tham khảo) ---
# Dùng để ước tính thời gian chạy tối đa cho tác vụ nặng trong Flask
MAX_PROFILE_TIME = 100  # Giây (nên nhỏ hơn timeout của Gunicorn, ví dụ: 120s)

# --- Flask Setup ---
app = Flask(__name__)
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

# --- Model Loading (EAGER LOADING - KHẮC PHỤC LỖI 502 TIỀM ẨN) ---
# Tải mô hình một lần khi ứng dụng/worker khởi động
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
        print(f"❌ Failed to download model from Hugging Face Space: {e}")
        model = None
        # Không raise FileNotFoundError để ứng dụng vẫn có thể chạy các route khác
else:
    print(f"✅ Model found locally: {MODEL_PATH}")

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None  # Đặt model là None nếu tải thất bại

# --- Decorators & Utility ---
# Decorator kiểm tra đăng nhập (bạn đã sử dụng 'user' trong session)
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required  # Đã thêm decorator để đảm bảo người dùng đã đăng nhập
def dashboard():
    server_port = os.environ.get("PORT", 5000)
    return render_template("dashboard.html", server_port=server_port)

@app.route("/emr_profile", methods=["GET", "POST"])
@login_required  # Đã thêm decorator
def emr_profile():
    """
    Tạo Profile Report và hiển thị.
    LƯU Ý: Đây là đoạn code gây ra lỗi WORKER TIMEOUT.
    Giải pháp bắt buộc là TĂNG TIMEOUT GUNICORN.
    """
    profile_html = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            # Giới hạn kích thước file để tránh timeout (ví dụ: 5MB)
            if 'content_length' in request.files['file'].__dict__ and request.files['file'].content_length > 5 * 1024 * 1024:
                flash("File quá lớn (>5MB). Vui lòng dùng file nhỏ hơn để tránh timeout.", "danger")
                return redirect(url_for("emr_profile"))

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Sử dụng tempfile để tránh vấn đề quyền truy cập hoặc xóa file dễ hơn
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                file.save(temp_file.name)
                filepath = temp_file.name

            try:
                if filename.lower().endswith((".csv", ".txt")):
                    df = pd.read_csv(filepath)
                elif filename.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(filepath)
                else:
                    flash("Lỗi: Chỉ hỗ trợ file CSV, Excel (.xls, .xlsx).", "danger")
                    return redirect(url_for("emr_profile"))

                # Nếu DataFrame lớn, sample để giảm thời gian xử lý
                if len(df) > 1000:
                    df = df.sample(1000)
                    flash("⚠️ File lớn, chỉ phân tích mẫu 1000 dòng để tránh timeout. Vui lòng dùng file nhỏ hơn cho báo cáo đầy đủ.", "warning")

            except Exception as e:
                flash(f"Lỗi đọc file: {e}", "danger")
                return redirect(url_for("emr_profile"))

            try:
                # Cảnh báo người dùng về quá trình chờ
                flash(f"🕒 Đang tạo báo cáo phân tích cho '{filename}'. Quá trình này có thể mất đến 2 phút. Vui lòng chờ...", "info")

                # Bắt đầu tính giờ cho tác vụ nặng
                start_time = time.time()

                # Tác vụ nặng - Tối ưu bằng cách tắt các tính năng nặng
                profile = ProfileReport(
                    df,
                    title=f"Báo cáo Phân tích Dữ liệu: {filename}",
                    explorative=False,  # Tắt explorative để nhanh hơn
                    correlations={"pearson": {"calculate": False}},  # Tắt correlation nặng
                    interactions={"continuous": False},  # Tắt interactions
                    missing_diagrams={"heatmap": False, "dendrogram": False}  # Tắt diagrams nặng
                )
                profile_html = profile.to_html()

                end_time = time.time()

                flash(f"✅ Đã tạo báo cáo thành công trong {end_time - start_time:.2f} giây!", "success")
            except Exception as e:
                # Bắt lỗi nếu quá trình tạo report bị gián đoạn (ví dụ: do timeout quá sớm)
                flash(f"❌ Lỗi khi tạo báo cáo: Quá trình bị ngắt do vượt quá giới hạn thời gian xử lý. Vui lòng thử lại với tập dữ liệu nhỏ hơn, hoặc kiểm tra lại cấu hình Gunicorn timeout. Chi tiết: {e}", "danger")
            finally:
                # Dọn dẹp file tạm thời
                os.remove(filepath)
        else:
            flash("⚠️ Vui lòng chọn file CSV hoặc Excel", "warning")
    return render_template("emr_profile.html",
                           profile_html=profile_html,
                           filename=filename)

@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required  # Đã thêm decorator
def emr_prediction():
    """
    Xử lý dự đoán ảnh.
    Đã khắc phục lỗi 502 do tải model lặp lại (model được tải sẵn).
    """
    prediction = None
    filename = None
    image_b64 = None
    if request.method == "POST":
        if model is None:
            flash("❌ Lỗi dự đoán: Mô hình AI chưa được tải thành công khi khởi động dịch vụ.", "danger")
            return redirect(url_for("emr_prediction"))

        file = request.files.get("file")
        if file and file.filename:
            # Giới hạn kích thước file (ví dụ: 5MB) để tránh overhead
            if 'content_length' in request.files['file'].__dict__ and request.files['file'].content_length > 5 * 1024 * 1024:
                flash("File quá lớn (>5MB). Vui lòng dùng file nhỏ hơn.", "danger")
                return redirect(url_for("emr_prediction"))

            filename = secure_filename(file.filename)

            # Lưu file vào thư mục tạm thời
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                file.save(temp_file.name)
                filepath = temp_file.name
            try:
                # Resize ảnh về kích thước mô hình mong muốn (240, 240)
                image = load_img(filepath, target_size=(240, 240))
                image_array = img_to_array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                # Thực hiện dự đoán trên model đã load sẵn
                prob = model.predict(image_array)[0][0]

                # Quyết định kết quả
                result = "Nodule (Có khối u)" if prob > 0.5 else "Non-nodule (Không có khối u)"
                prediction = {"result": result, "probability": float(prob)}
                # Đọc ảnh gốc về base64 để hiển thị
                with open(filepath, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

            except Exception as e:
                flash(f"❌ Lỗi dự đoán hình ảnh. Chi tiết lỗi: {e}", "danger")
            finally:
                # Dọn dẹp file tạm thời
                os.remove(filepath)
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
    # Sử dụng biến môi trường PORT cho Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
