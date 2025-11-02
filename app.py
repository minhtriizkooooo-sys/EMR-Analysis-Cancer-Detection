import os
import io
import base64
import numpy as np
import pandas as pd
from pathlib import Path 
import requests 
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging
from pandas.errors import ParserError 

# --- Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s INFO:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Flask config ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_strong_secret_key_12345')

# Cấu hình thư mục
UPLOAD_FOLDER = '/tmp/uploads'
# Dòng này đã được sửa lỗi cú pháp (loại bỏ ký tự U+00A0)
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# --- Model config (Tải từ Hugging Face Space) ---
MODEL = None
TARGET_SIZE = (240, 240)
MAX_FILE_SIZE_MB = 4 

# **URL RAW CỦA MODEL TRÊN HUGGING FACE**
HF_MODEL_URL = "https://huggingface.co/spaces/minhtriizkooooo/EMR-Analysis-Cancer-Detection/raw/main/models/best_weights_model.keras"
MODEL_FILENAME = "best_weights_model.keras"

BASE_DIR = Path(__file__).resolve().parent 
MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME


def load_keras_model():
    """Load model, downloading from HF if necessary."""
    global MODEL
    
    if MODEL is not None:
        return MODEL

    # 1. KIỂM TRA SỰ TỒN TẠI CỦA FILE
    if not MODEL_PATH.exists():
        logger.warning("⚠️ Model file NOT FOUND locally at %s. Attempting to download from Hugging Face...", MODEL_PATH)
        
        # Cố gắng tạo thư mục models/ nếu chưa có
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

        # 2. TẢI FILE TỪ HUGGING FACE
        try:
            logger.info(f"⬇️ Downloading model from: {HF_MODEL_URL}")
            response = requests.get(HF_MODEL_URL, stream=True, timeout=600) # Thêm timeout 10 phút cho file lớn
            response.raise_for_status() # Báo lỗi nếu mã trạng thái không phải 200 OK

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("✅ Model download successful.")

        except requests.exceptions.RequestException as req_e:
            logger.error(f"❌ CRITICAL: Failed to download model from Hugging Face: {req_e}")
            logger.error("Vui lòng kiểm tra lại URL RAW của model: %s", HF_MODEL_URL)
            return None
        except Exception as e:
            logger.error(f"❌ CRITICAL: An unexpected error occurred during download: {e}")
            return None
            
    # 3. TẢI MODEL VÀO BỘ NHỚ
    try:
        logger.info("🔥 Loading Keras model into memory...")
        MODEL = load_model(str(MODEL_PATH), compile=False) 
        logger.info("✅ Model loaded successfully from local file.")
    except Exception as e:
        logger.error(f"❌ Error loading model after download: {e}")
        MODEL = None
        
    return MODEL

# Tải model ngay khi ứng dụng bắt đầu
with app.app_context():
    load_keras_model()

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Vui lòng đăng nhập để truy cập trang này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def preprocess_image(image_file):
    """Preprocessing matched to Colab training (240x240 RGB, no rescale)."""
    if not MODEL:
        raise RuntimeError("Model is not loaded. Cannot preprocess.")
    img = load_img(image_file, target_size=TARGET_SIZE, color_mode='rgb')
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

def profile_csv_data(file_path, max_rows=5000, max_cols=50):
    try:
        if file_path.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin1')
                except Exception:
                    df = pd.read_csv(file_path, encoding='utf-8', engine='python')
            except ParserError as pe:
                 logger.error("Parser Error during CSV read: %s", pe)
                 raise ValueError("CSV format error. Check delimiter, quoting, or missing values.")

        elif file_path.lower().endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file_path)
            except Exception as xe:
                logger.error("Excel Read Error: %s", xe)
                raise ValueError("Excel file read error. Check file corruption or sheet name.")
        else:
            raise ValueError("Unsupported file extension in profiling function.")

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        n_rows, n_cols = df.shape
        
        info_html = f"""
        <div class="bg-blue-50 p-4 rounded-lg mb-4 shadow-sm border-l-4 border-blue-400">
            <h3 class="text-xl font-semibold text-blue-700 mb-2">📊 Tổng quan Dữ liệu</h3>
            <p><strong>Số bản ghi:</strong> {n_rows}</p>
            <p><strong>Số cột:</strong> {n_cols}</p>
        </div>
        """
        
        if n_rows > max_rows or n_cols > max_cols:
            sample = df.sample(min(200, n_rows)).head(50)
            return info_html + "<h4>⚠️ File quá lớn — hiển thị mẫu dữ liệu</h4>" + sample.to_html(classes='table-auto w-full text-sm', index=False)
        
        desc_html = df.describe(include='all', datetime_is_numeric=True).to_html(classes='table-auto w-full text-sm', border=0)
        
        nulls = "<h4 class='mt-4'>Nulls per column</h4><div class='table-responsive'><table class='table-auto w-full text-sm'><thead><tr><th>Column</th><th>Nulls</th><th>% Null</th></tr></thead><tbody>"
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            pct = (null_count / n_rows) * 100 if n_rows else 0
            nulls += f"<tr><td>{col}</td><td>{null_count}</td><td>{pct:.2f}%</td></tr>"
        nulls += "</tbody></table></div>"
        
        return info_html + "<h3 class='mt-4 mb-2'>📈 Thống kê Mô tả</h3>" + desc_html + nulls
    
    except ValueError as ve:
        logger.error("Data Read/Format Error: %s", ve)
        return f"<div class='p-4 bg-red-100 text-red-700 rounded'>❌ Lỗi Định dạng File: {str(ve)}</div>"
    except Exception as e:
        logger.error("General Error profiling EMR: %s", e)
        return f"<div class='p-4 bg-red-100 text-red-700 rounded'>❌ Lỗi Không Xác Định: {str(e)}</div>"

# --- Routes ---
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('userID')
        password = request.form.get('password')
        if user_id == 'user_demo' and password == 'Test@123456':
            session['user'] = user_id
            logger.info("User logged in: %s", user_id)
            return redirect(url_for('dashboard'))
        else:
            flash('ID hoặc mật khẩu không hợp lệ.', 'danger')
            return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Đăng xuất thành công.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=session.get('user'))

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
    summary = None
    filename = None
    
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("Không có file nào được tải lên.", "danger")
            return render_template('emr_profile.html', summary=None, filename=None)
            
        filename = file.filename
        
        try:
            file_stream = io.BytesIO(file.read())
            
            if len(file_stream.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File quá lớn ({len(file_stream.getvalue())//(1024*1024)}MB > {MAX_FILE_SIZE_MB}MB)")

            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_stream)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_stream)
            else:
                summary = f"<p class='text-red-500 font-semibold'>Chỉ hỗ trợ file CSV hoặc Excel. File: {filename}</p>"
                return render_template('emr_profile.html', summary=summary, filename=filename)

            rows, cols = df.shape
            col_info = []
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                unique_count = df[col].nunique()
                desc_stats = ""
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe().to_dict()
                    desc_stats = (
                        f"Min: {desc.get('min', 'N/A'):.2f}, "
                        f"Max: {desc.get('max', 'N/A'):.2f}, "
                        f"Mean: {desc.get('mean', 'N/A'):.2f}, "
                        f"Std: {desc.get('std', 'N/A'):.2f}"
                    )
                
                col_info.append(f"""
                    <li class="bg-gray-50 p-3 rounded-lg border-l-4 border-primary-green">
                        <strong class="text-gray-800">{col}</strong>
                        <ul class="ml-4 text-sm space-y-1 mt-1 text-gray-600">
                            <li><i class="fas fa-code text-indigo-500 w-4"></i> Kiểu dữ liệu: {dtype}</li>
                            <li><i class="fas fa-exclamation-triangle text-yellow-500 w-4"></i> Thiếu: {missing} ({missing/rows*100:.2f}%)</li>
                            <li><i class="fas fa-hashtag text-teal-500 w-4"></i> Giá trị duy nhất: {unique_count}</li>
                            {'<li class="text-xs text-gray-500"><i class="fas fa-chart-bar text-green-500 w-4"></i> Thống kê mô tả: ' + desc_stats + '</li>' if desc_stats else ''}
                        </ul>
                    </li>
                """)
            
            info = f"""
            <div class='bg-green-50 p-6 rounded-lg shadow-inner'>
                <h3 class='text-2xl font-bold text-product-green mb-4'><i class='fas fa-info-circle mr-2'></i> Thông tin Tổng quan</h3>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-left'>
                    <p class='font-medium text-gray-700'><i class='fas fa-th-list text-indigo-500 mr-2'></i> Số dòng dữ liệu: <strong>{rows}</strong></p>
                    <p class='font-medium text-gray-700'><i class='fas fa-columns text-indigo-500 mr-2'></i> Số cột dữ liệu: <strong>{cols}</strong></p>
                </div>
            </div>
            """
            
            table_html = df.head().to_html(classes="table-auto min-w-full divide-y divide-gray-200", index=False)
            summary = info
            summary += f"<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-cogs mr-2 text-primary-green'></i> Phân tích Cấu trúc Cột ({cols} Cột):</h4>"
            summary += f"<ul class='space-y-3 grid grid-cols-1 md:grid-cols-2 gap-3'>{''.join(col_info)}</ul>"
            summary += "<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-table mr-2 text-primary-green'></i> 5 Dòng Dữ liệu Đầu tiên:</h4>"
            summary += "<div class='overflow-x-auto shadow-md rounded-lg'>" + table_html + "</div>"
            
        except Exception as e:
            summary = f"<p class='text-red-500 font-semibold text-xl'>Lỗi xử lý file EMR: <code class='text-gray-700 bg-gray-100 p-1 rounded'>{e}</code></p>"
            
    return render_template('emr_profile.html', summary=summary, filename=filename)

@app.route('/emr_prediction', methods=['GET','POST'])
@login_required
def emr_prediction():
    prediction_result, filename, image_b64 = None, None, None
    
    if MODEL is None:
        flash('Hệ thống AI chưa sẵn sàng. Vui lòng kiểm tra log lỗi tải model.', 'danger')
        return render_template('emr_prediction.html')
    
    if request.method == 'POST':
        uploaded = request.files.get('file')
        if not uploaded or uploaded.filename == '':
            flash('Vui lòng chọn file hình ảnh.', 'danger')
            return redirect(request.url)
        if not allowed_file(uploaded.filename):
            flash('Định dạng file không được hỗ trợ.', 'danger')
            return redirect(request.url)
        filename = secure_filename(uploaded.filename)
        data = uploaded.read()
        image_b64 = base64.b64encode(data).decode('utf-8')
        image_stream = io.BytesIO(data)
        try:
            processed = preprocess_image(image_stream)
            preds = MODEL.predict(processed)
            logger.info("Raw model output: %s", preds.tolist())
            
            p_nodule = float(preds[0][0]) if preds.ndim == 2 and preds.shape[1] >= 1 else float(preds[0])

            label = 'Nodule' if p_nodule >= 0.5 else 'Non-nodule'
            prob = p_nodule if p_nodule >= 0.5 else 1.0 - p_nodule
            prediction_result = {'result': label, 'probability': float(np.round(prob,6)), 'raw_output': float(np.round(p_nodule,6))}
            flash('Dự đoán AI hoàn tất.', 'success')
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            flash(f'Lỗi khi xử lý hình ảnh hoặc dự đoán: {e}', 'danger')
            return redirect(request.url)
    return render_template('emr_prediction.html', prediction=prediction_result, filename=filename, image_b64=image_b64)

# --- Run ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 1000))
    logger.info("Starting Flask on port %s", port)
    app.run(host='0.0.0.0', port=port)
