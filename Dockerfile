# Dockerfile
FROM python:3.11-slim

# Set high timeout for pip
ENV PIP_DEFAULT_TIMEOUT=1200 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Copy the requirements file we just generated
COPY requirements.txt .

# 2. Install dependencies using standard pip
# We use a custom mirror and retries to handle bad internet
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --retries 10

# 3. Copy the rest of the application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]