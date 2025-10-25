from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import secrets
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# ========== Tải model từ Hugging Face ==========
MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
MODEL_FILENAME = "best_weights_model.keras"  # bạn xác nhận lại tên chính xác trong repo

model = None
try:
    print("⏳ Đang tải model từ Hugging Face …")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = load_model(model_path)
    print("✅ Model đã được tải và load thành công.")
except Exception as e:
    print(f"❌ Không thể tải hoặc load model từ HF: {e}")
    model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMG

# ========== Routes ==========
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("userID")
        password = request.form.get("password")
        # Thay đổi thông tin đăng nhập nếu cần
        if user_id == "user_demo" and password == "Test@123456":
            session["user"] = user_id
            print("✅ Đăng nhập thành công!")
            return redirect(url_for("dashboard"))
        else:
            flash("Sai thông tin đăng nhập. Vui lòng thử lại.", "danger")
            return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    print("👋 Người dùng đã đăng xuất.")
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Bạn cần đăng nhập để tiếp tục.", "danger")
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này.", "danger")
        return redirect(url_for("login"))

    filename = None
    image_b64 = None
    prediction = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("Chưa chọn hình ảnh để tải lên!", "danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("Bạn chưa chọn file!", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            print(f"📂 Ảnh đã tải lên: {filepath}")

            if model is None:
                flash("Model chưa được tải thành công!", "danger")
                return render_template("emr_prediction.html", filename=filename)

            try:
                img = Image.open(filepath).convert("RGB").resize((224,224))
                arr = np.array(img)/255.0
                arr = np.expand_dims(arr, axis=0)

                preds = model.predict(arr)
                prob = float(preds[0][0])
                label = "Nodule" if prob > 0.5 else "Non-nodule"
                prediction = {"result": label, "probability": prob}

                print(f"✅ Dự đoán thành công: {label} với xác suất {prob:.4f}")
            except Exception as e:
                flash(f"Lỗi khi phân tích ảnh: {e}", "danger")
                print(f"❌ Lỗi khi phân tích ảnh: {e}")
        else:
            flash("Định dạng ảnh không hợp lệ!", "danger")

    return render_template("emr_prediction.html",
                           filename=filename,
                           image_b64=image_b64,
                           prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
