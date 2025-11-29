FROM python:3.11-slim

# Gerekli sistem paketleri (opsiyonel ama faydalı)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce sadece requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 🔥 ÖNEMLİ: Projenin tamamını kopyala ki models/ da gelsin
COPY . .

# Cloud Run PORT
ENV PORT=8080

# Uygulama entrypoint
CMD ["python", "main.py"]

