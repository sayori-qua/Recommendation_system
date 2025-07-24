FROM python:3.10-slim

WORKDIR /app

#установим системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

#копируем файлы, если requirements не изменился
COPY requirements.txt .

#обновим pip и установим зависимости, если есть изменения в requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=120 -r requirements.txt

COPY . .
#запуск
CMD ["python", "app.py"]