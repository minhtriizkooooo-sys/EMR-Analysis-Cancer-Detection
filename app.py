# app.py — Flask EMR Prediction (load model từ Hugging Face Hub)
from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import logging

# --------------------------------------------------------------
# Cấu hình Flask
# --------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "emr-fixed-2025"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Cấu hình model từ Hugging Face
# --------------------------------------------------------------
HF_REPO_ID = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"        # ✅ bạn sử dụng link này
HF_MODEL_FILENAME = "best_weights_model.keras"                      # ✅ tên file bạn upload
MODEL_INPUT_SIZE = (224, 224)

MODEL = None
MODEL_LOADED = False
IS_DUMMY_MODE = False

# --------------------------------------------------------------
# Tải model từ Hugging Face Hub
# --------------------------------------------------------------
try:
    logger.info(f"🔄 Đang tải model từ Hugging Face: {HF_REPO_ID}/{HF_MODEL_FILENAME}")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
    MODEL = load_model(model_path)
    MODEL_LOADED = True
    logger.info("✅ Model đã tải thành công và được load vào bộ nhớ!")
except Exception as e:
    logger.error(f"❌ LỖI tải model từ Hugging Face: {e}")
    IS_DUMMY_MODE = True

# --------------------------------------------------------------
# Hàm dự đoán hình ảnh
# --------------------------------------------------------------
def predict_image(img_bytes):
    if IS_DUMMY_MODE or not MODEL_LOADED:
        return {
            "result": "Dummy mode",
            "probability": 0.5,
            "message": "Không thể tải model thực. Ứng dụng đang chạy chế độ mô phỏng."
        }

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(MODEL_INPUT_SIZE)
        img_array = np.expand_dims(np.asarray(img) / 255.0, axis=0)

        prediction = MODEL.predict(img_array, verbose=0)[0][0]
        result = "Nodule (U)" if prediction >= 0.5 else "Non-nodule (Không U)"
        prob = prediction if prediction >= 0.5 else 1 - prediction

        return {
            "result": result,
            "probability": float(prob),
            "message": "Dự đoán bằng mô hình thực từ Hugging Face."
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"result": "LỖI", "probability": 0.0, "message": str(e)}

# --------------------------------------------------------------
# Routes chính
# --------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/emr_profile")
def emr_profile():
    if "user" not in session:
        flash("Vui lòng đăng nhập để tiếp tục.", "warning")
        return redirect(url_for("index"))
    return render_template("emr_profile.html")

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if request.method == "POST":
        if "emr_image" not in request.files:
            flash("Không có tệp nào được chọn!", "danger")
            return redirect(request.url)

        file = request.files["emr_image"]
        if file.filename == "":
            flash("Tên tệp không hợp lệ!", "danger")
            return redirect(request.url)

        # Lưu ảnh upload
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Dự đoán ảnh
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        prediction = predict_image(img_bytes)

        flash("✅ Phân tích hồ sơ EMR thành công!", "success")
        return render_template(
            "emr_prediction.html",
            uploaded_image=file.filename,
            prediction=prediction
        )

    return render_template("emr_prediction.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    # => bạn thay logic login nếu muốn
    if username == "admin" and password == "123":
        session["user"] = username
        flash("Đăng nhập thành công!", "success")
        return redirect(url_for("emr_profile"))
    else:
        flash("Sai tài khoản hoặc mật khẩu!", "danger")
        return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("index"))

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 EMR AI – STARTING APP on port {port} – MODEL_LOADED: {MODEL_LOADED}")
    app.run(host="0.0.0.0", port=port, debug=False)
