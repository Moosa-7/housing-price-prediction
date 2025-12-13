# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir uv && uv pip install --system .

# Copy ALL files (Including the 'models' folder!)
COPY . . 

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]