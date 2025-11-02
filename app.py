import os
import io
import base64
import numpy as np
import pandas as pd
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from markupsafe import Markup
from ydata_profiling import ProfileReport

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Flask app setup ---
app = Flask(__name__)
app.secret_key = "emr_secret_key"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Load model ---
MODEL_PATH = os.path.join("models", "best_weights_model.keras")
model = None

try:
    model = load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")

# --- Home page ---
@app.route("/")
def index():
    return render_template("index.html")

# --- Dashboard ---
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# --- CSV profiling ---
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        flash("Không có file được tải lên", "danger")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("Chưa chọn file CSV", "warning")
        return redirect(url_for("dashboard"))

    if file and file.filename.endswith(".csv"):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            profile = ProfileReport(df, title="EMR Profiling Report", explorative=True)
            report_html = profile.to_html()

            with open("templates/EMR_Profile.html", "w", encoding="utf-8") as f:
                f.write(report_html)

            flash("Phân tích CSV thành công!", "success")
            return render_template("EMR_Profile.html")

        except Exception as e:
            flash(f"Lỗi xử lý CSV: {e}", "danger")
            logging.error(e)
            return redirect(url_for("dashboard"))

    else:
        flash("Vui lòng chọn file CSV hợp lệ", "warning")
        return redirect(url_for("dashboard"))

# --- Image prediction ---
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        flash("Không có file ảnh được tải lên", "danger")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("Chưa chọn file ảnh", "warning")
        return redirect(url_for("dashboard"))

    if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(filepath)

        try:
            img = load_img(filepath, target_size=(240, 240))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            if model is None:
                flash("Model chưa được tải hoặc lỗi model!", "danger")
                return redirect(url_for("dashboard"))

            prediction = model.predict(img_array)
            pred_class = "Nodule" if prediction[0][0] > 0.5 else "Non-Nodule"

            img_base64 = base64.b64encode(open(filepath, "rb").read()).decode("utf-8")
            flash(Markup(f"<b>Kết quả dự đoán:</b> {pred_class}"), "info")

            return render_template("EMR_Prediction.html",
                                   image_data=img_base64,
                                   result=pred_class)

        except Exception as e:
            flash(f"Lỗi xử lý ảnh: {e}", "danger")
            logging.error(e)
            return redirect(url_for("dashboard"))
    else:
        flash("Vui lòng chọn file ảnh hợp lệ (PNG, JPG, JPEG)", "warning")
        return redirect(url_for("dashboard"))

# --- Run app (Render uses gunicorn, port auto-set) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port)
