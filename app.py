from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import secrets
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
from keras.models import load_model
import requests

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =============================
# Thông tin model Hugging Face
# =============================
HF_MODEL_URL = "https://huggingface.co/minhtriizkooooo/EMR-Analysis-Cancer_Detection/resolve/main/best_weights_model.keras"
LOCAL_MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")

# =============================
# Tài khoản demo
# =============================
DEMO_USER = {
    "user_demo": "Test@123456"
}

# =============================
# Hàm tiện ích
# =============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'xlsx', 'png', 'jpg', 'jpeg'}

def analyze_emr_file(filepath):
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        summary = f"""
        <h3 class='text-2xl font-semibold text-green-700 mb-3'>📊 Tổng quan dữ liệu</h3>
        <p class='text-gray-700 mb-4'>Số dòng: <strong>{df.shape[0]}</strong> | Số cột: <strong>{df.shape[1]}</strong></p>
        {df.describe(include='all').to_html(classes='table-auto w-full border border-gray-300 rounded-lg')}
        """
        return summary
    except Exception as e:
        return f"<p class='text-red-600'>❌ Lỗi khi đọc file: {str(e)}</p>"

def prepare_image_for_model(image_path, target_size=(224, 224)):
    """Chuyển ảnh thành tensor phù hợp cho model"""
    img = Image.open(image_path).convert("RGB").resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def download_model_if_not_exists():
    """Tự động tải model từ Hugging Face nếu chưa có"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("🔽 Đang tải model từ Hugging Face...")
        response = requests.get(HF_MODEL_URL)
        if response.status_code == 200:
            with open(LOCAL_MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Model tải thành công!")
        else:
            raise Exception(f"Không thể tải model (HTTP {response.status_code})")

# =============================
# Tải model Keras khi khởi động
# =============================
try:
    download_model_if_not_exists()
    model = load_model(LOCAL_MODEL_PATH)
    print("✅ Model đã được tải và khởi tạo thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải hoặc load model: {str(e)}")
    model = None

# =============================
# Flask routes
# =============================

@app.route('/')
def home():
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        user = request.form.get('userID')
        pw = request.form.get('password')

        if user in DEMO_USER and DEMO_USER[user] == pw:
            session['user'] = user
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Sai ID hoặc mật khẩu.', 'danger')
            return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Đã đăng xuất khỏi hệ thống.', 'danger')
    return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Vui lòng đăng nhập để tiếp tục.', 'danger')
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    if 'user' not in session:
        flash('Vui lòng đăng nhập để tiếp tục.', 'danger')
        return redirect(url_for('login_page'))

    summary = None
    filename = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Chưa chọn file để tải lên.', 'danger')
            return redirect(request.url)
        
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('✅ File tải lên thành công và đã được phân tích!', 'success')
            summary = analyze_emr_file(file_path)
        else:
            flash('Định dạng file không hợp lệ.', 'danger')

    return render_template('emr_profile.html', summary=summary, filename=filename)

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    if 'user' not in session:
        flash('Vui lòng đăng nhập để tiếp tục.', 'danger')
        return redirect(url_for('login_page'))

    filename = None
    image_b64 = None
    prediction = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Chưa chọn ảnh để tải lên.', 'danger')
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_b64 = image_to_base64(file_path)

            try:
                if model is None:
                    flash('Model chưa được tải hoặc bị lỗi.', 'danger')
                    return redirect(request.url)

                img_arr = prepare_image_for_model(file_path)
                pred = model.predict(img_arr)[0][0]
                result = "Nodule" if pred > 0.5 else "Non-nodule"

                prediction = {
                    'result': result,
                    'probability': round(float(pred), 4)
                }

                flash('✅ Phân tích hình ảnh thành công!', 'success')
            except Exception as e:
                flash(f'❌ Lỗi khi dự đoán ảnh: {str(e)}', 'danger')
        else:
            flash('Định dạng ảnh không hợp lệ.', 'danger')

    return render_template('emr_prediction.html', filename=filename, image_b64=image_b64, prediction=prediction)

# =============================
# Run
# =============================
if __name__ == '__main__':
    app.run(debug=True)
