FROM python:3.11

WORKDIR /app

# 1. Cài đặt các thư viện hệ thống cần thiết cho numpy/pandas/h5py (hỗ trợ load_model)
# libhdf5-dev và hdf5-tools là cần thiết cho các file .keras
RUN apt-get update && apt-get install -y \
    gfortran \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# 2. Cài đặt Python Dependencies:
# Dùng --no-cache-dir để giảm kích thước layer docker.
# ❌ Loại bỏ --only-binary :all: vì nó có thể gây lỗi với các thư viện cần biên dịch nhẹ.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 3. Sử dụng lệnh CMD được tinh chỉnh (giống như Procfile)
# Điều này đảm bảo Docker chạy lệnh Gunicorn đã tối ưu ngay cả khi Procfile bị bỏ qua.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "120", "--max-requests", "500"]
