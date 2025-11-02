import os
import io
import gc
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from ydata_profiling import ProfileReport

# -----------------------------------
# Flask Configuration
# -----------------------------------
app = Flask(__name__)
app.secret_key = "emr_secret_key"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------------
# Model Configuration
# -----------------------------------
MODEL_PATH = "models/best_weights_model.keras"
IMG_SIZE = (240, 240)  # ✅ Chuẩn Colab
model = load_model(MODEL_PATH)

# -----------------------------------
# Routes
# -----------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# -----------------------------------
# EMR CSV Profiling
# -----------------------------------
@app.route("/emr_profile", methods=["POST"])
def emr_profile():
    if "file" not in request.files:
        flash("Không có tệp được tải lên.", "error")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("Chưa chọn tệp CSV.", "error")
        return redirect(url_for("dashboard"))

    try:
        df = pd.read_csv(file)
        profile = ProfileReport(df, title="📊 Báo cáo Phân tích EMR", minimal=True)
        html = profile.to_html()

        # Giao diện gọn, scroll mượt
        html_wrapped = f"""
        <div style="max-width:95%;margin:auto;padding:1rem;color:#d7f5d0;font-family:Inter,sans-serif;">
            <h1 style="text-align:center;color:#8fffa3;">📊 Báo cáo Phân tích EMR</h1>
            <div style="background:#0f2415;border-radius:12px;padding:1rem;
                        box-shadow:0 0 10px rgba(0,255,128,0.2);
                        overflow-x:auto;max-height:650px;overflow-y:auto;">
                {html}
            </div>
        </div>
        """
        return render_template("EMR_Profile.html", profile_html=Markup(html_wrapped))

    except Exception as e:
        flash(f"Lỗi khi phân tích file: {e}", "error")
        return redirect(url_for("dashboard"))

# -----------------------------------
# Image Prediction (Nodule / Non-Nodule)
# -----------------------------------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        flash("Không có tệp ảnh được tải lên.", "error")
        return redirect(url_for("dashboard"))

    file = request.files["image"]
    if file.filename == "":
        flash("Chưa chọn ảnh để phân tích.", "error")
        return redirect(url_for("dashboard"))

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # ✅ Resize đúng chuẩn Colab
        img = load_img(file_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Dự đoán
        prediction = model.predict(img_array)
        label = "Nodule" if prediction[0][0] > 0.5 else "Non-Nodule"
        confidence = float(prediction[0][0]) if label == "Nodule" else 1 - float(prediction[0][0])

        # Giải phóng bộ nhớ
        tf.keras.backend.clear_session()
        gc.collect()

        # Encode ảnh để hiển thị lại
        with open(file_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        return render_template(
            "EMR_Prediction.html",
            label=label,
            confidence=f"{confidence*100:.2f}%",
            image_data=encoded_image
        )

    except Exception as e:
        flash(f"Lỗi khi phân tích ảnh: {e}", "error")
        return redirect(url_for("dashboard"))

# -----------------------------------
# Run App
# -----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
