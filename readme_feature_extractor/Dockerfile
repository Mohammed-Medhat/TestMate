# 1. Use a lightweight Python image
FROM python:3.10-slim

# 2. Install essential build tools for llama-cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Install requirements (CPU version)
COPY requirements.txt .
RUN pip install --no-cache-dir llama-cpp-python
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files
COPY src/ ./src/
COPY download_model.py .
COPY run_pipeline.py .

# 5. Prepare models
RUN mkdir -p models
RUN python3 download_model.py


CMD ["python3", "run_pipeline.py"]