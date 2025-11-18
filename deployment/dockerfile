# Python 3.11 slim tabanlı imaj
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları:
# - libgomp1 : LightGBM'in ihtiyaç duyduğu OpenMP kütüphanesi
# - build-essential : Gerekirse bazı paketleri derlemek için (tsfresh, pyod vs. için güvenli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Uygulama kodu
COPY . .

# Logların anında görünmesi için
ENV PYTHONUNBUFFERED=1

# Cloud Run, PORT env'ini ayarlıyor (8080), main.py zaten aiohttp server başlatıyor
CMD ["python", "-u", "main.py"]

