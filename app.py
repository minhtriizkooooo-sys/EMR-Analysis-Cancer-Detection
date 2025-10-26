import os
import secrets
import threading
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.optimizers import Adam
from huggingface_hub import hf_hub_download
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==============================
# Flask config
# ==============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_CACHE = "model_cache"
os.makedirs(MODEL_CACHE, exist_ok=True)
CHART_FOLDER = "static/charts"
os.makedirs(CHART_FOLDER, exist_ok=True)

# ==============================
# Model config
# ==============================
HF_MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
HF_MODEL_FILE = "best_weights_model.keras" 
HF_TOKEN = os.environ.get("HF_TOKEN") 
NUM_CLASSES = 1
model = None
model_lock = threading.Lock()

def load_model_once():
    """Tải mô hình AI một lần duy nhất. Đã fix lỗi load trọng số và tối ưu load."""
    global model
    with model_lock:
        if model is not None:
            return model
        try:
            print("⏳ Tải model từ Hugging Face...")
            LOCAL_MODEL_PATH = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILE,
                cache_dir=MODEL_CACHE,
                token=HF_TOKEN
            )

            # Cấu trúc EfficientNetB0
            base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights=None)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            output_layer = Dense(1, activation='sigmoid')(x)
            model_local = Model(inputs=base_model.input, outputs=output_layer)

            # Nạp trọng số tùy chỉnh (EfficientNetB0-B)
            model_local.load_weights(LOCAL_MODEL_PATH) 
            model_local.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

            model = model_local
            print("✅ Model EfficientNet đã load thành công với trọng số tùy chỉnh")
            return model
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            # Thông báo lỗi model cho người dùng (fix lỗi emr_prediction)
            flash(f"Lỗi: Không thể tải mô hình AI ({e}). Chức năng dự đoán không hoạt động.", 'danger')
            model = None
            return None

# ==============================
# Dummy user
# ==============================
USERS = {"user_demo": "Test@123456"}

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        userID = request.form.get("userID")
        password = request.form.get("password")
        if userID in USERS and USERS[userID] == password:
            session["user"] = userID
            return redirect(url_for("dashboard"))
        else:
             flash("ID người dùng hoặc mật khẩu không chính xác.", 'danger')

    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ==============================
# EMR Profile (Phân tích file CSV/Excel) - Đã SỬA TÊN BIẾN CHO HTML
# ==============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    # Sửa lỗi: Dùng biến 'summary' để khớp với template (do không sửa HTML)
    summary = None 
    chart_urls = []

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(file_path, encoding='utf-8', engine='python')
                elif filename.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                else:
                    flash("Định dạng file không được hỗ trợ. Chỉ chấp nhận CSV, XLSX, XLS.", 'danger')
                    return render_template("emr_profile.html", filename=filename, summary=summary)

                # Thống kê tổng quan
                data_summary = df.describe(include='all').transpose()
                # Gán kết quả vào biến 'summary' để khớp với template HTML
                summary = data_summary.to_html(classes="table-auto w-full text-sm", border=0)

                # Vẽ biểu đồ cho các cột số
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                for i, col in enumerate(numeric_cols):
                    if len(df[col].dropna().unique()) > 2 and len(df) > 10: 
                        plt.figure(figsize=(8, 6))
                        df[col].hist(bins=20, color='#4CAF50', edgecolor='black')
                        plt.title(f"Phân bố {col}", fontsize=14)
                        plt.xlabel(col, fontsize=12)
                        plt.ylabel("Tần suất", fontsize=12)
                        plt.tight_layout()
                        
                        chart_file = os.path.join(CHART_FOLDER, f"{col}_{filename}.png")
                        plt.savefig(chart_file)
                        plt.close()
                        chart_urls.append(url_for('static', filename=f"charts/{col}_{filename}.png"))
                
                # Vẽ biểu đồ cho cột Gender
                if 'Gender' in df.columns:
                    plt.figure(figsize=(8, 6))
                    df['Gender'].value_counts().plot(kind='bar', color=['#2e7d32','#81c784'])
                    plt.title("Phân bố giới tính", fontsize=14)
                    plt.xticks(rotation=0)
                    plt.tight_layout()

                    gender_chart = os.path.join(CHART_FOLDER, f"Gender_{filename}.png")
                    plt.savefig(gender_chart)
                    plt.close()
                    chart_urls.append(url_for('static', filename=f"charts/Gender_{filename}.png"))

            except Exception as e:
                flash(f"Lỗi khi đọc file hoặc phân tích dữ liệu: {e}", 'danger')
                summary = f"<p class='text-red-500'>Lỗi khi đọc file: {e}</p>"

    # Truyền biến 'summary'
    return render_template("emr_profile.html", filename=filename, summary=summary, chart_urls=chart_urls)

# ==============================
# EMR Prediction (Dự đoán hình ảnh) - Đã Sửa
# ==============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction_result = None
    uploaded_image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Fix lỗi: Luôn tạo URL ảnh để template hiển thị (fix lỗi không hiện ảnh)
            uploaded_image_url = url_for('uploaded_file', filename=filename) 

            mdl = load_model_once()
            if mdl:
                try:
                    img = Image.open(file_path).convert("RGB")
                    img = img.resize((224, 224))
                    x = np.array(img, dtype=np.float32)/255.0
                    x = np.expand_dims(x, axis=0)

                    pred = mdl.predict(x, verbose=0)[0][0]
                    result = "Nodule" if pred >= 0.5 else "Non-nodule"
                    probability = float(pred if pred >= 0.5 else 1 - pred) 

                    prediction_result = {
                        "result": result,
                        "probability": round(probability, 4),
                        "image_url": uploaded_image_url # Truyền URL ảnh vào kết quả
                    }

                except Exception as e:
                    error_message = f"Lỗi xử lý ảnh hoặc dự đoán: {e}"
                    flash(error_message, 'danger')
                    prediction_result = {
                        "result": error_message,
                        "probability": 0,
                        "image_url": uploaded_image_url
                    }
            else:
                error_message = "Model chưa load được! Vui lòng kiểm tra log server."
                flash(error_message, 'danger')
                prediction_result = {
                    "result": error_message,
                    "probability": 0,
                    "image_url": uploaded_image_url
                }
    
    # Truyền filename và prediction_result
    return render_template("emr_prediction.html", prediction=prediction_result, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve files from the UPLOAD_FOLDER."""
    return send_from_directory(UPLOAD_FOLDER, filename)

# ==============================
# Run Flask
# ==============================

# Fix lỗi 502: Khối lệnh pre-load model để giảm lỗi timeout khi startup
try:
    print("Pre-loading AI Model on startup...")
    load_model_once()
except Exception as e:
    print(f"Lỗi nghiêm trọng khi Pre-load Model: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Chạy với threaded=False để đảm bảo model được load 1 lần
    app.run(host="0.0.0.0", port=port, threaded=False)
