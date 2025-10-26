import os
import secrets
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
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
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

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

# Initialize the model
model = load_keras_model()

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        return render_template('emr_profile.html')
    return render_template('emr_profile.html')

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    prediction = None
    uploaded_image = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_image = f'uploads/{filename}'
            
            img_array = preprocess_image(file_path)
            if img_array is None:
                flash('Error preprocessing image')
                return redirect(request.url)
            
            if model is None:
                flash('Model not loaded')
                return redirect(request.url)
            try:
                pred = model.predict(img_array)
                prediction = 'Positive' if pred[0][0] > 0.5 else 'Negative'
                logger.info(f"Prediction: {prediction}")
            except Exception as e:
                logger.error(f"❌ Error during prediction: {str(e)}")
                flash(f'Error during prediction: {str(e)}')
                return redirect(request.url)
            
            return render_template('emr_prediction.html', prediction=prediction, uploaded_image=uploaded_image)
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg')
            return redirect(request.url)
    return render_template('emr_prediction.html', prediction=None, uploaded_image=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's PORT or default to 10000
    app.run(host='0.0.0.0', port=port, debug=False)
