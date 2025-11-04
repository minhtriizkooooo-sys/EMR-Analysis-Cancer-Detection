import os
import io
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adamax
from functools import wraps
import logging

# ======================= LOGGER ==========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("EMR-AI")

# ======================= FLASK CONFIG =====================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_key_12345")

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["ALLOWED_EXTENSIONS"] = {"csv", "xlsx", "xls", "png", "jpg", "jpeg", "gif", "bmp"}

# ======================= MODEL CONFIG =====================
MODEL = None
TARGET_SIZE = (240, 240)
MODEL_FILENAME = "best_weights_model.keras"
MODEL_DIR = Path("/models")
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MIN_MODEL_SIZE_MB = 5


def load_keras_model():
    """Load local Keras model if available."""
    global MODEL
    if MODEL is not None:
        return MODEL

    if not MODEL_PATH.exists():
        logger.error(f"❌ Model not found at {MODEL_PATH}")
        return None

    if MODEL_PATH.stat().st_size < MIN_MODEL_SIZE_MB * 1024 * 1024:
        logger.error(f"❌ Model too small or corrupt: {MODEL_PATH.stat().st_size} bytes")
        return None

    try:
        logger.info(f"🔥 Loading Keras model from {MODEL_PATH}")
        MODEL = load_model(str(MODEL_PATH), compile=False, custom_objects={"Adamax": Adamax})
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        MODEL = None
    return MODEL


with app.app_context():
    load_keras_model()

# ======================= HELPERS ==========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Vui lòng đăng nhập để truy cập trang này.", "danger")
            return redirect(url_for("index"))
        return f(*args, **kwargs)

    return decorated_function


def preprocess_image(image_file):
    """Preprocess image to match training."""
    if not MODEL:
        raise RuntimeError("Model not loaded.")
    img = load_img(image_file, target_size=TARGET_SIZE, color_mode="rgb")
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ======================= ROUTES ===========================
@app.route("/", methods=["GET"])
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


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    return render_template("dashboard.html")


# ======================= EMR PROFILE ======================
@app.route("/emr_profile", methods=["GET", "POST"])
@login_required
def emr_profile():
    summary, filename = None, None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Không có file nào được tải lên.", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        try:
            stream = io.BytesIO(file.read())
            if len(stream.getvalue()) > 10 * 1024 * 1024:
                raise ValueError("File quá lớn (>10MB)")

            if filename.lower().endswith(".csv"):
                df = pd.read_csv(stream)
            elif filename.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(stream)
            else:
                flash("Chỉ hỗ trợ CSV hoặc Excel.", "danger")
                return redirect(request.url)

            rows, cols = df.shape
            summary = f"<h3 class='text-xl font-bold text-green-700 mb-4'>Phân tích cấu trúc dữ liệu</h3>"
            summary += f"<p><strong>Số dòng:</strong> {rows} | <strong>Số cột:</strong> {cols}</p>"
            summary += "<ul class='mt-3'>"
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                summary += f"<li><strong>{col}</strong> — kiểu: {dtype}, thiếu: {missing}</li>"
            summary += "</ul>"

        except Exception as e:
            summary = f"<p class='text-red-600 font-semibold'>Lỗi xử lý file: {e}</p>"

    return render_template("emr_profile.html", summary=summary, filename=filename)


# ======================= EMR PREDICTION ====================
@app.route("/emr_prediction", methods=["GET", "POST"])
@login_required
def emr_prediction():
    prediction_result, filename, image_b64 = None, None, None

    if MODEL is None:
        flash("Model AI chưa sẵn sàng. Kiểm tra log.", "danger")
        return render_template("emr_prediction.html")

    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            flash("Vui lòng chọn file hình ảnh.", "danger")
            return redirect(request.url)

        if not allowed_file(uploaded.filename):
            flash("Định dạng file không được hỗ trợ.", "danger")
            return redirect(request.url)

        filename = secure_filename(uploaded.filename)
        data = uploaded.read()
        image_b64 = base64.b64encode(data).decode("utf-8")

        try:
            processed = preprocess_image(io.BytesIO(data))
            preds = MODEL.predict(processed)
            logger.info(f"Model output: {preds.tolist()}")

            if preds.ndim == 2 and preds.shape[1] == 1:
                p = float(preds[0][0])
            else:
                p = float(np.max(preds[0]))

            if p >= 0.5:
                label = "Nodule"
                prob = p
            else:
                label = "Non-nodule"
                prob = 1 - p

            prediction_result = {
                "result": label,
                "probability": float(np.round(prob, 5)),
                "raw_output": float(np.round(p, 5))
            }
            flash("Dự đoán hoàn tất.", "success")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            flash(f"Lỗi khi dự đoán: {e}", "danger")

    return render_template("emr_prediction.html",
                           prediction=prediction_result,
                           filename=filename,
                           image_b64=image_b64)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ======================= RUN APP ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Tự động cho Render & HF
    logger.info(f"🚀 EMR Insight AI Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
