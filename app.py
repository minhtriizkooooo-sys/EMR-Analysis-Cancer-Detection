import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --------------------------
# LOGGING
# --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------
# FLASK CONFIG
# --------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_key_123")

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "csv", "xlsx"}

# --------------------------
# MODEL SETUP
# --------------------------
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")
MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"

model = None


def ensure_model_exists():
    """Kiểm tra hoặc tải model từ Hugging Face nếu chưa có."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        logger.info("🔽 Model chưa có, đang tải từ Hugging Face...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            total = 0
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        total += len(chunk)
                        f.write(chunk)
            size_mb = round(total / (1024 * 1024), 2)
            logger.info(f"✅ Đã tải model ({size_mb} MB) vào /models.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải model: {e}")
    else:
        logger.info("✅ Model đã có sẵn trong /models.")


def load_keras_model():
    """Load Keras model vào bộ nhớ."""
    global model
    if model is not None:
        return model
    ensure_model_exists()
    try:
        logger.info(f"🔥 Đang load model từ: {MODEL_PATH}")
        model = load_model(MODEL_PATH, compile=False)
        logger.info("✅ Model load thành công.")
    except Exception as e:
        logger.error(f"❌ Không thể load model: {e}")
        model = None
    return model


# Load model khi start app
with app.app_context():
    load_keras_model()


# --------------------------
# HELPER FUNCTIONS
# --------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------
# ROUTES
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("userID", "").strip()
    password = request.form.get("password", "").strip()

    if username == "user_demo" and password == "Test@123456":
        session["user"] = username
        return redirect(url_for("dashboard"))
    flash("Sai ID hoặc mật khẩu.", "danger")
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")


# --------------------------
# EMR PROFILE
# --------------------------
@app.route("/emr_profile", methods=["POST"])
def emr_profile():
    if "file" not in request.files:
        flash("Không có file nào được chọn.")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("Vui lòng chọn file CSV hoặc Excel.")
        return redirect(url_for("dashboard"))

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            summary = df.describe(include="all").to_html(classes="table table-striped table-bordered")
            return render_template("EMR_Profile.html", tables=[summary], titles=["Phân tích Dữ liệu EMR"])
        except Exception as e:
            logger.error(f"Lỗi xử lý EMR: {e}")
            flash("Không thể phân tích file EMR.")
            return redirect(url_for("dashboard"))

    flash("Định dạng file không được hỗ trợ.")
    return redirect(url_for("dashboard"))


# --------------------------
# EMR PREDICTION
# --------------------------
@app.route("/emr_prediction", methods=["POST"])
def emr_prediction():
    if "file" not in request.files:
        flash("Không có file nào được chọn.")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("Vui lòng chọn ảnh để dự đoán.")
        return redirect(url_for("dashboard"))

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        if model is None:
            flash("Model chưa sẵn sàng.")
            return redirect(url_for("dashboard"))

        try:
            img = load_img(file_path, target_size=(240, 240))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr)[0][0]
            label = "Nodule" if pred >= 0.5 else "Non-Nodule"
            prob = round(float(pred if pred >= 0.5 else 1 - pred), 4)

            with open(file_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            return render_template("EMR_Prediction.html", result=label, prob=prob, image_data=image_b64)
        except Exception as e:
            logger.error(f"Lỗi dự đoán: {e}")
            flash("Không thể xử lý ảnh.")
            return redirect(url_for("dashboard"))

    flash("Định dạng file không được hỗ trợ.")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# --------------------------
# RENDER DEPLOY ENTRY POINT
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 100900))
    logger.info(f"🚀 Đang chạy EMR App trên Render (port={port})")
    app.run(host="0.0.0.0", port=port, debug=False)
