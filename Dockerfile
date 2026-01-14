FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_USE_LEGACY_KERAS=1   # 🔥 REQUIRED FIX

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "300"]
