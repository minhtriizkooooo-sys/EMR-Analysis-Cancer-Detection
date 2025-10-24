FROM python:3.11

WORKDIR /app

# Install system dependencies.
# FIX: Thêm thư viện HDF5 (libhdf5-dev) CẦN THIẾT cho h5py
RUN apt-get update && apt-get install -y \
    gfortran \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Giữ nguyên cờ --only-binary để ưu tiên wheels và tránh lỗi compiler khác
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
