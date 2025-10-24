# -*- coding: utf-8 -*-
# app.py: EMR AI - FIXED GOOGLE DRIVE DOWNLOAD + REAL KERAS PREDICTION
# CHỈ HIỂN THỊ THUMBNAIL 100x100 thay vì full image

import base64
import os
import io
import logging
import time
import requests
from PIL import Image
from flask import (
    Flask, flash, redirect, render_template, request, session, url_for
)
import pandas as pd
import gdown
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten

# ==========================================================
# KHỞI TẠO BIẾN TOÀN CỤC
# ==========================================================
MODEL = None
MODEL_LOADED = False
IS_DUMMY_MODE = False

# ==========================================================
# CẤU HÌNH GOOGLE DRIVE VÀ MODEL
# ==========================================================
DRIVE_FILE_ID = "1ORV8tDkT03fxjRyaWU5liZ2bHQz3YQC"  # REPLACE WITH VALID ID
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILE_NAME)
MODEL_INPUT_SIZE = (224, 224)
ALTERNATIVE_URL = None  # Set to S3/Dropbox URL if available (e.g., "https://dropbox.com/s/...")

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================================
# HÀM TẢI FILE
# ==========================================================
def download_file_from_gdrive(file_id, destination, max_retries=3):
    """
    Tải file từ Google Drive hoặc URL thay thế. Trả về: (success: bool, is_dummy: bool).
    """
    if os.path.exists(destination):
        logger.info(f"File đã tồn tại: {destination}. Đang kiểm tra tính hợp lệ...")
        try:
            model = load_model(destination)
            model.predict(np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)))
            logger.info("✅ File tồn tại là mô hình Keras hợp lệ.")
            return True, False
        except Exception as e:
            logger.warning(f"❌ File tồn tại nhưng không hợp lệ: {e}. Tải lại...")
            os.remove(destination)

    # Try Google Drive
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    logger.info(f"Đang cố gắng tải model từ GDrive ID: {file_id} về {destination}...")
    
    for attempt in range(max_retries):
        try:
            # Kiểm tra quyền truy cập
            response = requests.head(url, allow_redirects=True)
            logger.info(f"Checking URL: {url}, Status code: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"❌ Không thể truy cập file. Status code: {response.status_code}")
                continue

            # Tải bằng gdown
            gdown.download(id=file_id, output=destination, quiet=False, fuzzy=True)
            if os.path.exists(destination):
                model = load_model(destination)
 @System: * Today's date and time is 04:06 AM +07 on Saturday, October 25, 2025. *
