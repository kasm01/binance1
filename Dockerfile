FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080
ENV PIP_ROOT_USER_ACTION=ignore

# Sistem paketlerini hafif tut
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Önce sadece requirements (layer cache için)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Şimdi tüm proje içeriğini kopyala
COPY . .

# Ana proses
CMD ["python", "main.py"]
