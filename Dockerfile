# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Copy dependency files
COPY pyproject.toml uv.lock* ./

# 2. Install 'uv' and install dependencies SYSTEM-WIDE
RUN pip install --no-cache-dir uv \
    && uv pip install --system .

# 3. Copy the rest of the application
COPY . .

# 4. Expose the port
EXPOSE 8000

# 5. Run the API
# PATH EXPLANATION:
# src  -> Folder
# api  -> Folder
# main -> File (main.py)
# app  -> The FastAPI object inside the file
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]