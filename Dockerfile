FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ core/
COPY scripts/ scripts/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "core.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
