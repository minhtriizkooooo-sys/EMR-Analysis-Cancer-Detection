import os
import secrets
import threading
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
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
NUM_CLASSES = 1  # nhị phân
model = None
model_lock = threading.Lock()

def load_model_once():
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

            base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights=None)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            output_layer = Dense(1, activation='sigmoid')(x)
            model_local = Model(inputs=base_model.input, outputs=output_layer)

            model_local.load_weights(LOCAL_MODEL_PATH, skip_mismatch=True)
            model_local.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

            model = model_local
            print("✅ Model nhị phân đã load thành công")
            return model
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
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
# EMR Profile - chuyên nghiệp
# ==============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    summary_html = None
    chart_urls = []

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(file_path, encoding='utf-8', engine='python')
                elif filename.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                else:
                    summary_html = "<p>Định dạng file không được hỗ trợ.</p>"
                    return render_template("emr_profile.html", filename=filename, summary_html=summary_html)

                # Thống kê tổng quan
                summary = df.describe(include='all').transpose()
                summary_html = summary.to_html(classes="table-auto w-full")

                # Vẽ biểu đồ cho các cột số và các cột quan trọng nếu có
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                for col in numeric_cols:
                    plt.figure()
                    df[col].hist(bins=20, color='skyblue')
                    plt.title(f"Phân bố {col}")
                    chart_file = os.path.join(CHART_FOLDER, f"{col}_{filename}.png")
                    plt.savefig(chart_file)
                    plt.close()
                    chart_urls.append(url_for('static', filename=f"charts/{col}_{filename}.png"))

                # Vẽ biểu đồ cho cột Gender nếu có
                if 'Gender' in df.columns:
                    plt.figure()
                    df['Gender'].value_counts().plot(kind='bar', color=['skyblue','salmon'])
                    plt.title("Phân bố giới tính")
                    gender_chart = os.path.join(CHART_FOLDER, f"Gender_{filename}.png")
                    plt.savefig(gender_chart)
                    plt.close()
                    chart_urls.append(url_for('static', filename=f"charts/Gender_{filename}.png"))

            except Exception as e:
                summary_html = f"<p>Lỗi khi đọc file: {e}</p>"

    return render_template("emr_profile.html", filename=filename, summary_html=summary_html, chart_urls=chart_urls)

# ==============================
# EMR Prediction
# ==============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction_result = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

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
                        "image_url": uploaded_image_url
                    }

                except Exception as e:
                    prediction_result = {
                        "result": f"Lỗi khi dự đoán: {e}",
                        "probability": 0,
                        "image_url": uploaded_image_url
                    }
            else:
                prediction_result = {
                    "result": "Model chưa load được!",
                    "probability": 0,
                    "image_url": uploaded_image_url
                }

    return render_template("emr_prediction.html", prediction=prediction_result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ==============================
# Run Flask
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
