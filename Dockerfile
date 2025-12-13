# Dockerfile
FROM python:3.11-slim

# Set timeouts for both UV and PIP to handle slow networks
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=1200

WORKDIR /app

# 1. Copy dependency files
COPY pyproject.toml uv.lock* ./

# 2. Install 'uv' with high timeout, then install dependencies
RUN pip install --default-timeout=1000 --no-cache-dir uv \
    && uv pip install --system .

# 3. Copy the rest of the application
COPY . .

# 4. Expose the port
EXPOSE 8000

# 5. Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]