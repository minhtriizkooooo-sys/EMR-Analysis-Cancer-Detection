import os
import secrets
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

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
# Model config
# =============================
HF_MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
HF_MODEL_FILE = "best_weights_model.keras"
HF_TOKEN = os.environ.get("HF_TOKEN")
NUM_CLASSES = 128
CLASSES = [f"Lớp Bệnh #{i+1}" for i in range(NUM_CLASSES)]

model = None

def load_model_once():
    """Load EfficientNetB0 model lazy, chỉ 1 lần"""
    global model
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

        # ===== EfficientNetB0 =====
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output_layer)
        model.load_weights(LOCAL_MODEL_PATH, skip_mismatch=True)
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"✅ Model EfficientNetB0 + Top layers đã load thành công")
        return model
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        model = None
        return None

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
            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                else:
                    flash("Định dạng file không được hỗ trợ.", "danger")
                    return render_template("emr_profile.html")
                summary = df.describe().to_html(classes="table-auto w-full")
            except Exception as e:
                flash(f"Lỗi khi đọc file: {e}", "danger")
    return render_template("emr_profile.html", filename=filename, summary=summary)

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction = None
    image_b64 = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Load model từ HF nếu chưa load
            mdl = load_model_once()
            if mdl:
                try:
                    # --- Load ảnh RGB ---
                    img = Image.open(file_path).convert("RGB")
                    img = img.resize((224, 224))
                    x = np.array(img)/255.0
                    x = np.expand_dims(x, axis=0)

                    # --- Dự đoán ---
                    pred = mdl.predict(x)
                    pred_class_index = np.argmax(pred[0])
                    pred_prob = float(pred[0][pred_class_index])

                    # --- Map sang Nodule / Non-nodule ---
                    # TODO: Cập nhật NODULE_CLASSES theo model thật
                    NODULE_CLASSES = list(range(64))  # ví dụ: 0-63 là Nodule
                    result_label = "Nodule" if pred_class_index in NODULE_CLASSES else "Non-nodule"

                    # --- Top 5 class ---
                    top_5_indices = np.argsort(pred[0])[-5:][::-1]
                    top_5_probs = pred[0][top_5_indices]
                    top_5_results = [f"{CLASSES[i]}: {top_5_probs[idx]*100:.2f}%" for idx, i in enumerate(top_5_indices)]

                    # --- Chuẩn bị dữ liệu để hiển thị ---
                    prediction = {
                        "result": result_label,
                        "probability": pred_prob,
                        "top_5": top_5_results
                    }

                    # --- Convert ảnh sang base64 ---
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    image_b64 = base64.b64encode(buffered.getvalue()).decode()

                except Exception as e:
                    flash(f"Lỗi khi dự đoán: {e}", "danger")
            else:
                flash("Model chưa load được!", "danger")

    return render_template("emr_prediction.html", filename=filename,
                           prediction=prediction, image_b64=image_b64)

# =============================
# Run Flask
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
