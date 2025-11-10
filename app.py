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









@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("userID", "").strip()
    password = request.form.get("password", "").strip()
    
    if username == "user_demo" and password == "Test@123456":
        session['user'] = username
        return redirect(url_for("dashboard"))
    flash("Sai ID hoặc mật khẩu.", "danger")
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for("index"))
    # FIXED MODE vì đã loại bỏ model TensorFlow/Keras
    return render_template("dashboard.html", model_status="✅ FIXED MODE")

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
    summary = None
    filename = None
    
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("Không có file nào được tải lên.", "danger")
            return render_template('emr_profile.html', summary=None, filename=None)
            
        filename = file.filename
        
        try:
            file_stream = io.BytesIO(file.read())
            
            # Check file size early (if not already done by Nginx/MAX_CONTENT_LENGTH)
            if len(file_stream.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File quá lớn ({len(file_stream.getvalue())//(1024*1024)}MB > 4MB)")

            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_stream)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_stream)
            else:
                summary = f"<p class='text-red-500 font-semibold'>Chỉ hỗ trợ file CSV hoặc Excel. File: {filename}</p>"
                return render_template('emr_profile.html', summary=summary, filename=filename)

            rows, cols = df.shape
            col_info = []
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                unique_count = df[col].nunique()
                desc_stats = ""
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe().to_dict()
                    desc_stats = (
                        f"Min: {desc.get('min', 'N/A'):.2f}, "
                        f"Max: {desc.get('max', 'N/A'):.2f}, "
                        f"Mean: {desc.get('mean', 'N/A'):.2f}, "
                        f"Std: {desc.get('std', 'N/A'):.2f}"
                    )
                
                col_info.append(f"""
                    <li class="bg-gray-50 p-3 rounded-lg border-l-4 border-primary-green">
                        <strong class="text-gray-800">{col}</strong>
                        <ul class="ml-4 text-sm space-y-1 mt-1 text-gray-600">
                            <li><i class="fas fa-code text-indigo-500 w-4"></i> Kiểu dữ liệu: {dtype}</li>
                            <li><i class="fas fa-exclamation-triangle text-yellow-500 w-4"></i> Thiếu: {missing} ({missing/rows*100:.2f}%)</li>
                            <li><i class="fas fa-hashtag text-teal-500 w-4"></i> Giá trị duy nhất: {unique_count}</li>
                            {'<li class="text-xs text-gray-500"><i class="fas fa-chart-bar text-green-500 w-4"></i> Thống kê mô tả: ' + desc_stats + '</li>' if desc_stats else ''}
                        </ul>
                    </li>
                """)
            
            info = f"""
            <div class='bg-green-50 p-6 rounded-lg shadow-inner'>
                <h3 class='text-2xl font-bold text-product-green mb-4'><i class='fas fa-info-circle mr-2'></i> Thông tin Tổng quan</h3>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-left'>
                    <p class='font-medium text-gray-700'><i class='fas fa-th-list text-indigo-500 mr-2'></i> Số dòng dữ liệu: <strong>{rows}</strong></p>
                    <p class='font-medium text-gray-700'><i class='fas fa-columns text-indigo-500 mr-2'></i> Số cột dữ liệu: <strong>{cols}</strong></p>
                </div>
            </div>
            """
            
            table_html = df.head().to_html(classes="table-auto min-w-full divide-y divide-gray-200", index=False)
            summary = info
            summary += f"<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-cogs mr-2 text-primary-green'></i> Phân tích Cấu trúc Cột ({cols} Cột):</h4>"
            summary += f"<ul class='space-y-3 grid grid-cols-1 md:grid-cols-2 gap-3'>{''.join(col_info)}</ul>"
            summary += "<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-table mr-2 text-primary-green'></i> 5 Dòng Dữ liệu Đầu tiên:</h4>"
            summary += "<div class='overflow-x-auto shadow-md rounded-lg'>" + table_html + "</div>"
            
        except Exception as e:
            summary = f"<p class='text-red-500 font-semibold text-xl'>Lỗi xử lý file EMR: <code class='text-gray-700 bg-gray-100 p-1 rounded'>{e}</code></p>"
            
    return render_template('emr_profile.html', summary=summary, filename=filename)



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








@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/health")
def health():
    # Thêm route Health Check tiêu chuẩn
    return {"status": "healthy"}, 200

if __name__ == "__main__":
    # KHÔNG DÙNG 10000. DÙNG BIẾN MÔI TRƯỜNG $PORT DO Render CUNG CẤP
    port = int(os.environ.get("PORT", 5000)) # Dùng 5000 làm mặc định cho local
    logger.info("🚀 EMR AI - FIXED BASE64 CRASH")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
