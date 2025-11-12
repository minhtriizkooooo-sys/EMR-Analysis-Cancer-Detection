import os
import io
import logging
from flask import Flask, render_template, request, redirect, flash, send_file
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests

# ------------------------------
# SETUP
# ------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "test_secret")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emr_ai")

MODEL = None
MODEL_PATH = "models/best_weights_model.keras"
MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/resolve/main/models/best_weights_model.keras"

os.makedirs("models", exist_ok=True)


# ------------------------------
# LOAD MODEL (lazy)
# ------------------------------
def load_keras_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        if not os.path.exists(MODEL_PATH):
            logger.info("⬇️ Downloading model from Hugging Face...")
            r = requests.get(MODEL_URL, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        MODEL = load_model(MODEL_PATH, compile=False)
        logger.info("✅ Model loaded successfully.")
        return MODEL
    except Exception as e:
        logger.exception(f"❌ Failed to load model: {e}")
        return None


# ------------------------------
# HOMEPAGE
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------
# PROFILE PAGE
# ------------------------------
@app.route("/profile")
def profile():
    return render_template("profile.html")


# ------------------------------
# PREDICTION PAGE
# ------------------------------
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    global MODEL
    if request.method == "POST":
        try:
            if "image" not in request.files:
                flash("❌ Vui lòng chọn file ảnh.", "danger")
                return redirect(request.url)

            file = request.files["image"]
            if file.filename == "":
                flash("❌ Chưa chọn file ảnh.", "danger")
                return redirect(request.url)

            # Load model (lazy)
            if MODEL is None:
                logger.info("🧩 Loading model on first use...")
                MODEL = load_keras_model()
                if MODEL is None:
                    flash("⚠️ Không thể tải model AI. Vui lòng thử lại sau.", "danger")
                    return redirect(request.url)

            # Process image
            img = Image.open(file.stream).convert("RGB").resize((224, 224))
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predict safely
            logger.info("🩺 Running prediction...")
            preds = MODEL.predict(arr, verbose=0)
            result = "CANCER DETECTED" if preds[0][0] > 0.5 else "NORMAL"

            logger.info(f"✅ Prediction done: {result}")
            flash(f"Kết quả: {result}", "success")
            return render_template("emr_prediction.html", result=result)

        except Exception as e:
            logger.exception("Error during prediction:")
            flash(f"❌ Lỗi trong quá trình dự đoán: {e}", "danger")
            return redirect(request.url)

    return render_template("emr_prediction.html", result=None)


# ------------------------------
# HEALTH CHECK
# ------------------------------
@app.route("/health")
def health():
    return {"status": "ok"}


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 EMR Flask app running on port {port}")
    app.run(host="0.0.0.0", port=port)
