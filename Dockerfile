FROM python:3.11-slim

# Install system dependencies FIRST (before switching user)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create user as required by HF
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    WARMUP_ON_STARTUP=true

WORKDIR /app

# Copy requirements and install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY --chown=user . /app

# Create results directory
RUN mkdir -p results

EXPOSE 7860

# HF requires port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]


