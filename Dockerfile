# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    libpng16-16 \
    libtiff6 \
    libopenblas-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . /app

# Create necessary directories
RUN mkdir -p /app/logs /app/cache && chown -R user:user /app

# Switch to user
USER user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port
EXPOSE 7860

# Default command (can be overridden)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

# Alternative commands for different deployment scenarios:
# For production with multiple workers:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]
# 
# For development with auto-reload:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
# 
# For Hugging Face Spaces (single worker recommended):
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
