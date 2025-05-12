FROM python:3.10-slim
WORKDIR /app

# 1) ship static ffmpeg / ffprobe
COPY bin/ffmpeg  /usr/local/bin/ffmpeg
COPY bin/ffprobe /usr/local/bin/ffprobe
RUN chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# 2) python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) the Streamlit UI
COPY app.py .

EXPOSE 5000
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
