from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from huggingface_hub import hf_hub_download

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Cần cho session và flash message

# =============================
# Tải model từ Hugging Face
# =============================
MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
MODEL_FILENAME = "best_weights_model.keras"

print("🔽 Đang tải model từ Hugging Face...")
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = load_model(model_path)
    print("✅ Model đã tải thành công.")
except Exception as e:
    print(f"❌ Không thể tải model: {e}")
    model = None

# =============================
# Trang chủ (Login)
# =============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('userID')
    password = request.form.get('password')

    # Kiểm tra thông tin đăng nhập (demo)
    if user_id == 'user_demo' and password == 'Test@123456':
        session['user'] = user_id
        print("✅ Đăng nhập thành công!")  # Chỉ in ra console, không flash
        return redirect(url_for('dashboard'))
    else:
        flash("Sai thông tin đăng nhập. Vui lòng thử lại!", "danger")
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    print("👋 Người dùng đã đăng xuất.")
    return redirect(url_for('index'))

# =============================
# Dashboard
# =============================
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash("Bạn cần đăng nhập để tiếp tục!", "danger")
        return redirect(url_for('index'))
    return render_template('dashboard.html')

# =============================
# Hồ sơ EMR
# =============================
@app.route('/emr_profile')
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập để tiếp tục!", "danger")
        return redirect(url_for('index'))
    return render_template('emr_profile.html')

# =============================
# Phân tích ảnh y tế (EMR Prediction)
# =============================
@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    if 'user' not in session:
        flash("Vui lòng đăng nhập để sử dụng chức năng này!", "danger")
        return redirect(url_for('index'))

    prediction_result = None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            flash("Chưa chọn ảnh để tải lên!", "danger")
            return render_template('emr_prediction.html')

        try:
            img = Image.open(file.stream)

            # Chuyển ảnh grayscale → RGB (nếu cần)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            if model is not None:
                preds = model.predict(img_array)
                label = "Nodule" if preds[0][0] > 0.5 else "Non-nodule"
                prediction_result = f"Kết quả dự đoán: {label} (xác suất: {preds[0][0]:.4f})"
                print(prediction_result)  # ✅ chỉ in ra console, không flash
            else:
                flash("Model chưa được tải thành công!", "danger")

        except Exception as e:
            flash(f"Lỗi xử lý ảnh: {str(e)}", "danger")

    return render_template('emr_prediction.html', prediction_result=prediction_result)

# =============================
# Khởi chạy ứng dụng (Render-compatible)
# =============================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
