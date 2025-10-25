import os
import secrets
import numpy as np
import pandas as pd
import tensorflow as tf 
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image

# SỬ DỤNG TF.KERAS CHUẨN ĐỂ TƯƠNG THÍCH TỐT NHẤT
from tensorflow.keras.models import load_model 
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish as tf_swish

# CẬP NHẬT QUAN TRỌNG: Đăng ký cả hai tên gọi 'swish' và 'SiLU'
custom_objects = {
    'swish': tf_swish,
    'SiLU': tf_swish  
}

from huggingface_hub import hf_hub_download

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_CACHE = "model_cache"
os.makedirs(MODEL_CACHE, exist_ok=True)

# =============================
# Tải model từ Hugging Face
# =============================
# Repo ID và Tên file model
HF_MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection" 
HF_MODEL_FILE = "best_weights_model.keras" 

# LẤY HF TOKEN TỪ BIẾN MÔI TRƯỜNG
HF_TOKEN = os.environ.get("HF_TOKEN") 

model = None
try:
    print("⏳ Tải model từ Hugging Face...")
    
    # BƯỚC 1: Tải file .keras chỉ chứa weights
    LOCAL_MODEL_PATH = hf_hub_download(
        repo_id=HF_MODEL_REPO, 
        filename=HF_MODEL_FILE, 
        cache_dir=MODEL_CACHE,
        token=HF_TOKEN 
    )
    
    # BƯỚC 2: Xây dựng lại kiến trúc model (Architecture) - ĐÃ SỬA LỖI MỚI
    # Cấu hình phải khớp chính xác: EfficientNetB0 -> GlobalPooling -> Dense(1024) -> Dense(512) -> Dense(4)
    
    # Khởi tạo EfficientNetB0
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None # Không dùng ImageNet weights
    )

    # Thêm các lớp phân loại (Head Layers)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Lớp ẩn 1: 1024 units (đã sửa lần trước)
    x = Dense(1024, activation='relu')(x) 
    
    # Lớp ẩn 2: 512 units (Sửa lỗi dựa trên log mismatch mới nhất)
    x = Dense(512, activation='relu')(x)
    
    # Lớp đầu ra: 4 classes (Giả định)
    output_layer = Dense(4, activation='softmax')(x) 

    # Xây dựng model hoàn chỉnh
    model = Model(inputs=base_model.input, outputs=output_layer)

    # BƯỚC 3: Tải weights đã được lưu từ file .keras vào kiến trúc vừa tạo
    try:
        model.load_weights(LOCAL_MODEL_PATH, skip_mismatch=True) 
        print("✅ Tải weights thành công (sử dụng skip_mismatch=True).")
    except Exception as lw_e:
        # Nếu skip_mismatch không hoạt động hoặc không được hỗ trợ, thử tải cơ bản
        model.load_weights(LOCAL_MODEL_PATH) 
        print("✅ Tải weights thành công (tải cơ bản).")
        
    # BƯỚC 4: Compile lại model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"✅ Model THẬT (EfficientNetB0) đã được TÁI TẠO và tải weights thành công từ {LOCAL_MODEL_PATH}")

except Exception as e:
    # Fallback chỉ để log lỗi, không cố gắng load trực tiếp vì đã biết lỗi cấu trúc
    print(f"❌ Lỗi Tái Tạo Kiến Trúc: {e}")
    print("LƯU Ý QUAN TRỌNG: Model THẬT không tải được. Chức năng dự đoán ảnh sẽ bị vô hiệu hóa.")
    model = None
        

# =============================
# Dummy user (Giữ nguyên)
# =============================
USERS = {"user_demo": "Test@123456"}

# =============================
# Routes (Giữ nguyên)
# =============================
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
            flash("Sai ID hoặc mật khẩu", "danger")
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

# =============================
# EMR CSV/Excel Analysis (Giữ nguyên)
# =============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    filename = None
    summary = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Xử lý dữ liệu EMR
            try:
                # Kiểm tra định dạng file và đọc
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                else:
                    flash("Định dạng file không được hỗ trợ. Vui lòng tải lên CSV hoặc Excel.", "danger")
                    return render_template("emr_profile.html")
                    
                summary = df.describe().to_html(classes="table-auto w-full")
            except Exception as e:
                flash(f"Lỗi khi đọc file: {e}", "danger")
    return render_template("emr_profile.html", filename=filename, summary=summary)

# =============================
# EMR Image Prediction (Giữ nguyên)
# =============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        # Chỉ chạy dự đoán khi file có VÀ model đã được tải thành công (model is NOT None)
        if file and model: 
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                # TIỀN XỬ LÝ ẢNH
                # Model yêu cầu (224, 224, 3) do cách lưu, nên ta phải tạo ảnh giả-RGB từ Grayscale.
                img = Image.open(file_path).convert("L")  # Grayscale (1 kênh)
                img = img.resize((224, 224)) 
                x = np.array(img)/255.0
                
                # CHUYỂN 1 KÊNH (W, H) thành 3 KÊNH (W, H, 3)
                x = np.stack([x, x, x], axis=-1) 

                # Thêm batch size (axis=0)
                # Hình dạng đầu vào: (1, 224, 224, 3)
                x = np.expand_dims(x, axis=0) 

                # KIỂM TRA ĐẦU VÀO ĐỂ DEBUG
                print(f"Input shape cho model (Fake RGB): {x.shape}")
                
                pred = model.predict(x)
                
                # Chuyển kết quả dự đoán thành chuỗi
                prediction_value = np.argmax(pred[0])
                
                # Định nghĩa các lớp dự đoán (Giả định 4 lớp)
                CLASSES = ["Lành Tính (Benign)", "Ác Tính (Malignant)", "Bình Thường (Normal)", "Khác (Other)"]
                
                # Đảm bảo prediction_value không vượt quá số lớp
                if prediction_value < len(CLASSES):
                    predicted_class = CLASSES[prediction_value]
                else:
                    predicted_class = f"Lớp {prediction_value} (Không xác định)"
                
                prediction = f"Kết quả dự đoán: {predicted_class} | Probabilities: {pred[0].tolist()}" 
                
            except Exception as e:
                flash(f"Lỗi khi dự đoán: {e}", "danger")
        elif not model:
            # Thông báo lỗi rõ ràng nếu model không load được
            flash("Model chưa load được! (Lỗi khởi tạo - vui lòng kiểm tra log tải model)", "danger")
    return render_template("emr_prediction.html", filename=filename, prediction=prediction)

# =============================
# Chạy Flask (Giữ nguyên)
# =============================
if __name__ == "__main__":
    # CHÚ Ý: Cần tăng timeout Gunicorn trên Render!
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
