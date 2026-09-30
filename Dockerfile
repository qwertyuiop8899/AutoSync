FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV AUTOSYNC_DATA_DIR=/app/data

EXPOSE 8089

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8089"]
