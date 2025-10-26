import os
import secrets
import threading
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from huggingface_hub import hf_hub_download
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

# ==============================
# Flask config
# ==============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_CACHE = "model_cache"
os.makedirs(MODEL_CACHE, exist_ok=True)
CHART_FOLDER = "static/charts"
os.makedirs(CHART_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Model config
# ==============================
HF_MODEL_REPO = "minhtriizkooooo/EMR-Analysis-Cancer_Detection"
HF_MODEL_FILE = "best_weights_model.keras"
HF_TOKEN = os.environ.get("HF_TOKEN")
NUM_CLASSES = 1
model = None
model_lock = threading.Lock()

def load_model_once():
    """Load the AI model once, handling errors gracefully."""
    global model
    with model_lock:
        if model is not None:
            return model
        try:
            logger.info("⏳ Loading model from Hugging Face...")
            LOCAL_MODEL_PATH = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILE,
                cache_dir=MODEL_CACHE,
                token=HF_TOKEN
            )

            # Build EfficientNetB0 model
            base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights=None)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            output_layer = Dense(1, activation='sigmoid')(x)
            model_local = Model(inputs=base_model.input, outputs=output_layer)

            # Load weights and compile
            model_local.load_weights(LOCAL_MODEL_PATH)
            model_local.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

            model = model_local
            logger.info("✅ Model EfficientNet loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None

# ==============================
# Dummy user
# ==============================
USERS = {"user_demo": "Test@123456"}

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        userID = request.form.get("userID")
        password = request.form.get("password")
        if userID in USERS and USERS[userID] == password:
            session["user"] = userID
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid user ID or password.", 'danger')

    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ==============================
# EMR Profile (Analyze CSV/Excel files)
# ==============================
@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    summary = None
    chart_urls = []

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(file_path, encoding='utf-8', engine='python')
                elif filename.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                else:
                    flash("Unsupported file format. Only CSV, XLSX, XLS are accepted.", 'danger')
                    return render_template("emr_profile.html", filename=filename, summary=summary)

                data_summary = df.describe(include='all').transpose()
                summary = data_summary.to_html(classes="table-auto w-full text-sm", border=0)

                # Generate charts for numeric columns
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                for i, col in enumerate(numeric_cols):
                    if len(df[col].dropna().unique()) > 2 and len(df) > 10:
                        plt.figure(figsize=(8, 6))
                        df[col].hist(bins=20, color='#4CAF50', edgecolor='black')
                        plt.title(f"Distribution of {col}", fontsize=14)
                        plt.xlabel(col, fontsize=12)
                        plt.ylabel("Frequency", fontsize=12)
                        plt.tight_layout()

                        chart_file = os.path.join(CHART_FOLDER, f"{col}_{filename}.png")
                        plt.savefig(chart_file)
                        plt.close()
                        chart_urls.append(url_for('static', filename=f"charts/{col}_{filename}.png"))

                # Generate chart for Gender column
                if 'Gender' in df.columns:
                    plt.figure(figsize=(8, 6))
                    df['Gender'].value_counts().plot(kind='bar', color=['#2e7d32', '#81c784'])
                    plt.title("Gender Distribution", fontsize=14)
                    plt.xticks(rotation=0)
                    plt.tight_layout()

                    gender_chart = os.path.join(CHART_FOLDER, f"Gender_{filename}.png")
                    plt.savefig(gender_chart)
                    plt.close()
                    chart_urls.append(url_for('static', filename=f"charts/Gender_{filename}.png"))

            except Exception as e:
                flash(f"Error reading file or analyzing data: {e}", 'danger')
                summary = f"<p class='text-red-500'>Error reading file: {e}</p>"

    return render_template("emr_profile.html", filename=filename, summary=summary, chart_urls=chart_urls)

# ==============================
# EMR Prediction (Medical Image Prediction)
# ==============================
@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    filename = None
    prediction_result = None
    uploaded_image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            # Validate file format
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                flash("Only image files (PNG, JPG, JPEG, GIF, BMP) are accepted for prediction.", 'danger')
                return render_template("emr_prediction.html", prediction=None, filename=None)

            try:
                file.save(file_path)
                uploaded_image_url = url_for('uploaded_file', filename=filename)

                mdl = load_model_once()
                if mdl:
                    # Process image
                    img = Image.open(file_path).convert("RGB")
                    img = img.resize((224, 224))
                    x = np.array(img, dtype=np.float32) / 255.0
                    x = np.expand_dims(x, axis=0)

                    # Predict
                    pred = mdl.predict(x, verbose=0)[0][0]
                    result = "Nodule" if pred >= 0.5 else "Non-nodule"
                    probability = float(pred if pred >= 0.5 else 1 - pred)

                    prediction_result = {
                        "result": result,
                        "probability": round(probability, 4),
                        "image_url": uploaded_image_url
                    }
                else:
                    flash("AI model is not available.", 'danger')
                    prediction_result = {
                        "result": "Model not available",
                        "probability": 0,
                        "image_url": uploaded_image_url
                    }

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                flash(f"Prediction error: {e}", 'danger')
                prediction_result = {
                    "result": f"Error: {e}",
                    "probability": 0,
                    "image_url": uploaded_image_url
                }

    return render_template("emr_prediction.html", prediction=prediction_result, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve files from the UPLOAD_FOLDER."""
    return send_from_directory(UPLOAD_FOLDER, filename)

# ==============================
# Initialize Model
# ==============================
if __name__ == "__main__":
    # Pre-load model in a controlled manner
    try:
        logger.info("Pre-loading AI Model on startup...")
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs
        load_model_once()
    except Exception as e:
        logger.error(f"Critical error during model pre-load: {e}")

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=False)
