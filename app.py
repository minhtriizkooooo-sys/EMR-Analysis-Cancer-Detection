import os
import secrets
import numpy as np
import pandas as pd
# SỬ DỤNG TF.KERAS CHUẨN ĐỂ TƯƠNG THÍCH TỐT NHẤT
import tensorflow as tf 
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image

# THAY ĐỔI: Sử dụng load_weights thay vì load_model hoàn toàn
from tensorflow.keras.models import load_model 
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation
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
# Repo ID
HF_MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection" 

# Tên file model
HF_MODEL_FILE = "best_weights_model.keras" 

LOCAL_MODEL_PATH = os.path.join(MODEL_CACHE, HF_MODEL_FILE)

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
    
    # BƯỚC 2: Xây dựng lại kiến trúc model (Architecture)
    # Dựa trên lỗi log, model EfficientNetB0 của bạn được lưu với yêu cầu 3 kênh màu (RGB) 
    # trong lớp Conv đầu tiên (stem_conv).
    
    # Khởi tạo EfficientNetB0 với đầu vào 3 kênh màu theo yêu cầu của model lưu.
    # Kích thước đầu vào 224x224x3 (RGB)
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None # Không dùng ImageNet weights
    )

    # Thêm các lớp phân loại (head)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x) # Giả sử có lớp 256
    output_layer = Dense(4, activation='softmax')(x) # Giả sử có 4 lớp đầu ra (tùy vào bài toán)

    # Xây dựng model hoàn chỉnh
    model = Model(inputs=base_model.input, outputs=output_layer)

    # BƯỚC 3: Tải weights đã được lưu từ file .keras vào kiến trúc vừa tạo
    # Sử dụng load_weights thay vì load_model để bỏ qua lỗi cấu hình đầu vào
    model.load_weights(LOCAL_MODEL_PATH, safe_mode=False) 
    
    # BƯỚC 4: Compile lại model (nên có sau khi load weights)
    # LƯU Ý: Nếu model của bạn không cần compile, có thể bỏ qua bước này.
    # Tuy nhiên, để đảm bảo model hoạt động đúng, ta nên compile lại.
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"✅ Model THẬT (EfficientNetB0) đã được TÁI TẠO và tải weights thành công từ {LOCAL_MODEL_PATH}")

except Exception as e:
    # Nếu lỗi load_weights, ta sẽ fallback về load_model để kiểm tra.
    try:
        # Thử lại cách load model trực tiếp, nếu lỗi tái tạo kiến trúc.
        model = load_model(LOCAL_MODEL_PATH, custom_objects=custom_objects, compile=False)
        print("✅ Model load trực tiếp thành công. (Fallback)")
    except Exception as fallback_e:
        print(f"❌ Lỗi load model ban đầu (Load Model): {e}")
        print(f"❌ Lỗi load model fallback (Load Model Trực Tiếp): {fallback_e}")
        print("LƯU Ý QUAN TRỌNG: Model THẬT không tải được. Chức năng dự đoán ảnh sẽ bị vô hiệu hóa.")
        print("Cần kiểm tra lại: Cấu trúc lớp phân loại (Dense Layers) và số lượng Output (4 lớp) có khớp với model đã huấn luyện không?")
        model = None
        

# =============================
# Dummy user
# =============================
USERS = {"user_demo": "Test@123456"}

# =============================
# Routes
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
# EMR CSV/Excel Analysis
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
# EMR Image Prediction
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
                # FIX: Bắt buộc phải tạo ảnh 3 kênh (W, H, 3) để khớp với kiến trúc EfficientNetB0 đã được tái tạo.
                # Do bạn nói model được huấn luyện trên Grayscale, ta sẽ chuyển
                # ảnh Grayscale sang 3 kênh (fake RGB: R=G=B).
                img = Image.open(file_path).convert("L")  # Grayscale (1 kênh)
                img = img.resize((224, 224)) 
                x = np.array(img)/255.0
                
                # CHUYỂN 1 KÊNH (W, H) thành 3 KÊNH (W, H, 3)
                # Stack 3 lần để tạo giả-RGB, thỏa mãn đầu vào 3 kênh của EfficientNetB0.
                # Model sẽ nhận 3 kênh giống hệt nhau, xử lý như ảnh xám 3 lần.
                x = np.stack([x, x, x], axis=-1) 

                # Thêm batch size (axis=0)
                # Hình dạng đầu vào: (1, 224, 224, 3)
                x = np.expand_dims(x, axis=0) 

                # KIỂM TRA ĐẦU VÀO ĐỂ DEBUG
                print(f"Input shape cho model (Fake RGB): {x.shape}")
                
                pred = model.predict(x)
                
                # Chuyển kết quả dự đoán thành chuỗi
                prediction_value = np.argmax(pred[0])
                prediction = f"Kết quả dự đoán: Lớp {prediction_value} | Probabilities: {pred[0]}" 
            except Exception as e:
                flash(f"Lỗi khi dự đoán: {e}", "danger")
        elif not model:
            # Thông báo lỗi rõ ràng nếu model không load được
            flash("Model chưa load được! (Lỗi khởi tạo - vui lòng kiểm tra log tải model)", "danger")
    return render_template("emr_prediction.html", filename=filename, prediction=prediction)

# =============================
# Chạy Flask
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
