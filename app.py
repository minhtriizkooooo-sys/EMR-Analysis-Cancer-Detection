import os
import secrets
import numpy as np
import pandas as pd
import base64
import mimetypes
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Generate a secure key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'csv', 'xls', 'xlsx'}
app.config['SESSION_TYPE'] = 'filesystem'  # For session management

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_REPO = 'minhtriizkooooo/EMR-Analysis-Cancer_Detection'
MODEL_FILENAME = 'best_weights_model.keras'
IMG_SIZE = (224, 224)  # Model expects 224x224 images

# Load the model from Hugging Face
def load_keras_model():
    try:
        logger.info("⏳ Loading model from Hugging Face...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        model = load_model(model_path)
        logger.info("✅ Model loaded successfully")
        model.summary()  # Log model architecture for debugging
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return None

# Initialize the model as None for lazy loading
model = None

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        return None

# Function to encode image to base64
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"❌ Error encoding image to base64: {str(e)}")
        return None

# Function to process CSV/Excel file and generate detailed summary
def process_emr_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:  # .xls or .xlsx
            df = pd.read_excel(file_path)
        
        # Generate detailed summary
        summary_html = ""
        
        # 1. Data Shape
        summary_html += "<h3 class='text-xl font-bold mb-2'>Kích thước dữ liệu</h3>"
        summary_html += f"<p>Số hàng: {df.shape[0]}</p>"
        summary_html += f"<p>Số cột: {df.shape[1]}</p>"
        
        # 2. Column Info
        summary_html += "<h3 class='text-xl font-bold mt-4 mb-2'>Thông tin cột</h3>"
        col_info = pd.DataFrame({
            'Kiểu dữ liệu': df.dtypes,
            'Giá trị thiếu': df.isnull().sum(),
            'Giá trị duy nhất': df.nunique()
        })
        summary_html += col_info.to_html(classes='table-auto w-full border-collapse', index=True)
        
        # 3. Descriptive Statistics
        summary_html += "<h3 class='text-xl font-bold mt-4 mb-2'>Thống kê mô tả</h3>"
        summary_html += df.describe(include='all').to_html(classes='table-auto w-full border-collapse')
        
        # 4. Missing Values Percentage
        summary_html += "<h3 class='text-xl font-bold mt-4 mb-2'>Tỷ lệ giá trị thiếu (%)</h3>"
        missing_perc = (df.isnull().mean() * 100).to_frame(name='Tỷ lệ thiếu (%)').round(2)
        summary_html += missing_perc.to_html(classes='table-auto w-full border-collapse', index=True)
        
        # 5. Sample Data (first 5 rows)
        summary_html += "<h3 class='text-xl font-bold mt-4 mb-2'>Dữ liệu mẫu (5 hàng đầu)</h3>"
        summary_html += df.head().to_html(classes='table-auto w-full border-collapse', index=False)
        
        return summary_html
    except Exception as e:
        logger.error(f"❌ Error processing EMR file: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Match form fields from index.html
        user_id = request.form.get('userID')
        password = request.form.get('password')
        # Placeholder authentication (replace with actual logic, e.g., database check)
        if user_id == 'user_demo' and password == 'Test@123456':  # Demo credentials
            session['logged_in'] = True
            session['user'] = user_id  # Set user for display in templates
            return redirect(url_for('dashboard'))
        else:
            flash('ID người dùng hoặc mật khẩu không đúng. Vui lòng thử lại.', 'danger')
            return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('logged_in', None)
    session.pop('user', None)  # Clear user data
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập dashboard.', 'danger')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập phân tích hồ sơ EMR.', 'danger')
        return redirect(url_for('login'))
    
    filename = None
    summary = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không tìm thấy file trong yêu cầu.', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file.', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                summary = process_emr_file(file_path)
                if summary is None:
                    flash('Lỗi khi xử lý file EMR.', 'danger')
                    return redirect(request.url)
            except Exception as e:
                logger.error(f"❌ Error saving EMR file: {str(e)}")
                flash('Lỗi khi lưu file EMR.', 'danger')
                return redirect(request.url)
        else:
            flash('Loại file không hợp lệ. Chỉ chấp nhận CSV, XLS, XLSX.', 'danger')
            return redirect(request.url)
    
    return render_template('emr_profile.html', filename=filename, summary=summary)

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang dự đoán.', 'danger')
        return redirect(url_for('login'))
    
    prediction = None
    uploaded_image = None
    image_b64 = None
    filename = None
    mime_type = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không tìm thấy file trong yêu cầu.', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file.', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                uploaded_image = f'uploads/{filename}'
                image_b64 = image_to_base64(file_path)
                if image_b64 is None:
                    flash('Lỗi khi mã hóa hình ảnh.', 'danger')
                    return redirect(request.url)
                mime_type = mimetypes.guess_type(file_path)[0]
                if mime_type is None:
                    mime_type = 'image/jpeg'
            except Exception as e:
                logger.error(f"❌ Error saving file: {str(e)}")
                flash('Lỗi khi lưu file hình ảnh.', 'danger')
                return redirect(request.url)
            
            img_array = preprocess_image(file_path)
            if img_array is None:
                flash('Lỗi khi xử lý hình ảnh.', 'danger')
                return redirect(request.url)
            
            global model
            if model is None:
                model = load_keras_model()
                if model is None:
                    flash('Không thể tải mô hình AI. Vui lòng thử lại sau.', 'danger')
                    return redirect(request.url)
            
            try:
                pred = model.predict(img_array)
                result = 'Nodule' if pred[0][0] > 0.5 else 'Non-nodule'
                probability = float(pred[0][0])
                prediction = {'result': result, 'probability': probability}
                logger.info(f"Dự đoán: {result} (Xác suất: {probability*100:.2f}%)")
            except Exception as e:
                logger.error(f"❌ Lỗi khi dự đoán: {str(e)}")
                flash(f'Lỗi khi thực hiện dự đoán: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Loại file không hợp lệ. Chỉ chấp nhận PNG, JPG, JPEG, GIF, BMP.', 'danger')
            return redirect(request.url)
    
    return render_template('emr_prediction.html', prediction=prediction, uploaded_image=uploaded_image, image_b64=image_b64, filename=filename, mime_type=mime_type)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's PORT or default to 10000
    app.run(host='0.0.0.0', port=port, debug=False)
