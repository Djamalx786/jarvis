# Life OS Agent — container image for Railway (or any Docker host).
FROM python:3.12-slim

# System libraries:
#  - WeasyPrint needs Pango/Cairo/GDK-Pixbuf + fonts to render the PDF calendar.
#  - ffmpeg converts Telegram voice (.ogg) to .wav before transcription.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        libcairo2 \
        shared-mime-info \
        fonts-dejavu-core \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ holds the SQLite checkpoint DB, RAG index and runtime files.
RUN mkdir -p data

CMD ["python", "main.py"]
