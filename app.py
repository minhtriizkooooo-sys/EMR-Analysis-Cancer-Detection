import os
import secrets
import numpy as np
import pandas as pd
import base64
import mimetypes
import logging
import gc # Python Garbage Collector
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from huggingface_hub import hf_hub_download
from keras import backend as K

# ==========================================================
# 🧠 SAFE TENSORFLOW CONFIG
# ==========================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    # Disable GPU visibility for CPU usage (good practice on resource-limited environment)
    tf.config.set_visible_devices([], 'GPU')
    K.clear_session()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# 🔧 FLASK CONFIG
# ==========================================================
app = Flask(__name__)
# Thay secrets.token_hex(16) bằng biến môi trường hoặc giá trị cố định an toàn
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['ALLOWED_EMR_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
app.config['ALLOWED_EXTENSIONS'] = app.config['ALLOWED_IMAGE_EXTENSIONS'] | app.config['ALLOWED_EMR_EXTENSIONS']
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==========================================================
# 🔬 GLOBAL MODEL CONFIG AND VARIABLE
# ==========================================================
MODEL_REPO = 'minhtriizkooooo/EMR-Analysis-Cancer_Detection'
MODEL_FILENAME = 'best_weights_model.keras'
IMG_SIZE = (224, 224)

# KHỞI TẠO model = None ở mức toàn cục
model = None 

# ==========================================================
# ⚙️ LOAD MODEL SAFELY
# ==========================================================
def load_keras_model():
    """Load Keras model safely from Hugging Face"""
    global model
    
    try:
        logger.info("⏳ Downloading model from Hugging Face...")
        # Note: hf_hub_download is blocking
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        
        # Note: load_model is blocking
        model = load_model(model_path, compile=False)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")

# ==========================================================
# 🧩 HELPER FUNCTIONS
# ==========================================================
def allowed_file(filename, allowed_extensions=app.config['ALLOWED_EXTENSIONS']):
    """Check allowed file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def preprocess_image(image_path):
    """Prepare image for model input"""
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        return None


def image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        # Lấy MIME type để hiển thị đúng
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'image/jpeg' 

        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}" # Trả về định dạng data URI
    except Exception as e:
        logger.error(f"❌ Error encoding image: {str(e)}")
        return None

# Hàm process_emr_file
def process_emr_file(file_path):
    """Generate basic analysis summary of CSV/XLSX file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        html = ""
        html += "<h3 class='text-xl font-bold mb-2'>📊 Kích thước dữ liệu</h3>"
        html += f"<p>Số hàng: {df.shape[0]}</p>"
        html += f"<p>Số cột: {df.shape[1]}</p>"

        html += "<h3 class='text-xl font-bold mt-4 mb-2'>🔠 Thông tin cột</h3>"
        col_info = pd.DataFrame({
            'Kiểu dữ liệu': df.dtypes,
            'Giá trị thiếu': df.isnull().sum(),
            'Giá trị duy nhất': df.nunique()
        })
        html += col_info.to_html(classes='table-auto w-full border', index=True)

        html += "<h3 class='text-xl font-bold mt-4 mb-2'>📈 Thống kê mô tả</h3>"
        html += df.describe(include='all').to_html(classes='table-auto w-full border')

        html += "<h3 class='text-xl font-bold mt-4 mb-2'>⚠️ Tỷ lệ giá trị thiếu (%)</h3>"
        missing = (df.isnull().mean() * 100).to_frame(name='Tỷ lệ thiếu (%)').round(2)
        html += missing.to_html(classes='table-auto w-full border')

        html += "<h3 class='text-xl font-bold mt-4 mb-2'>🧾 Dữ liệu mẫu (5 hàng đầu)</h3>"
        html += df.head().to_html(classes='table-auto w-full border', index=False)

        return html
    except Exception as e:
        logger.error(f"❌ Error processing EMR file: {str(e)}")
        return None


# ==========================================================
# 🌐 ROUTES
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple demo login"""
    if request.method == 'POST':
        user_id = request.form.get('userID')
        password = request.form.get('password')
        # Placeholder for real authentication
        if user_id == 'user_demo' and password == 'Test@123456':
            session['logged_in'] = True
            session['user'] = user_id
            return redirect(url_for('dashboard'))
        else:
            flash('ID người dùng hoặc mật khẩu không đúng.', 'danger')
            return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập dashboard.', 'danger')
        return redirect(url_for('login'))
    return render_template('dashboard.html')


@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    """Handle EMR profiling"""
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập phân tích hồ sơ EMR.', 'danger')
        return redirect(url_for('login'))

    filename, summary = None, None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Chưa chọn file.', 'danger')
            return redirect(request.url)

        # Check for EMR file extensions (CSV/XLSX)
        if allowed_file(file.filename, app.config['ALLOWED_EMR_EXTENSIONS']):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            summary = process_emr_file(file_path)
            if summary is None:
                flash('Lỗi khi xử lý file EMR.', 'danger')
            else:
                # 💡 ĐÃ SỬA LỖI THỤT LỀ TẠI ĐÂY
                flash(f'Phân tích file EMR ({filename}) hoàn tất.', 'success')
        else:
            flash('Loại file không hợp lệ. Chỉ chấp nhận CSV, XLS, XLSX.', 'danger')

    return render_template('emr_profile.html', filename=filename, summary=summary)


@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    """Handle EMR image prediction"""
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang dự đoán.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Chưa chọn file ảnh.', 'danger')
            return redirect(url_for('emr_prediction'))

        # Check for image file extensions
        if not allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
            flash('Chỉ chấp nhận file ảnh (PNG, JPG, JPEG, GIF, BMP).', 'danger')
            return redirect(url_for('emr_prediction'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        if img_array is None:
            flash('Lỗi khi xử lý hình ảnh.', 'danger')
            return redirect(url_for('emr_prediction'))

        try:
            # Check if model is loaded globally by the Master process
            global model
            if model is None:
                flash('Mô hình AI chưa được tải. Vui lòng kiểm tra logs để biết lỗi tải mô hình.', 'danger')
                return redirect(url_for('emr_prediction'))

            # BƯỚC QUAN TRỌNG: Gọi predict
            input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            pred = model.predict(input_tensor, verbose=0)
            
            # 👇 THÊM DỌN DẸP BỘ NHỚ SAU DỰ ĐOÁN (Chống OOM)
            K.clear_session() 
            gc.collect()
            logger.info("✅ Keras/TF session and Python garbage collected.")
            # 👆 KẾT THÚC DỌN DẸP BỘ NHỚ

            # Assuming binary classification where pred[0][0] is the probability of the positive class
            probability = float(pred[0][0])
            result = 'Nodule' if probability > 0.5 else 'Non-nodule'
            
            # Store prediction data in session
            session['prediction_result'] = {
                'result': result,
                'probability': round(probability * 100, 2),
                'filename': filename,
                'image_b64': image_to_base64(file_path),
                'mime_type': mimetypes.guess_type(file_path)[0] or 'image/jpeg'
            }
            
            flash(f'Dự đoán hoàn tất: {result} với xác suất {round(probability * 100, 2)}%.', 'success')
            return redirect(url_for('emr_prediction'))

        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            flash('Lỗi khi dự đoán hình ảnh. Có thể do timeout.', 'danger')
            
            # DỌN DẸP BỘ NHỚ KỂ CẢ KHI CÓ LỖI
            try:
                K.clear_session()
                gc.collect()
            except:
                pass
            
            return redirect(url_for('emr_prediction'))

    # Retrieve and clear prediction data for GET request (display results)
    prediction_data = session.pop('prediction_result', None)

    return render_template(
        'emr_prediction.html',
        prediction=prediction_data,
        uploaded_image=None, 
        image_b64=None if not prediction_data else prediction_data['image_b64'],
        filename=None if not prediction_data else prediction_data['filename']
    )


# ==========================================================
# 🚀 PRELOAD MODEL (BEFORE GUNICORN FORKS WORKERS)
# ==========================================================
# Gunicorn loads the application file once.
# Code here runs in the master process before workers are forked.
# Workers then inherit the loaded model, preventing load-time timeouts.
logger.info("🔥 Bắt đầu tải mô hình trong tiến trình Master (Gunicorn Preload)...")
load_keras_model()
logger.info("🚀 Tiến trình Master đã hoàn tất tải mô hình.")

# ==========================================================
# 🚀 RUN APP
# ==========================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
