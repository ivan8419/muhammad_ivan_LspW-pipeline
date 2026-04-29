FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update --allow-releaseinfo-change && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install ONLY the required Python dependencies for serving
RUN pip install --no-cache-dir flask prometheus-client

# Copy application files
COPY . .

# Set environment variables
ENV MODEL_PATH=serving_model/muhammad_ivan_LspW-pipeline
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8080
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the serving application
CMD ["python", "serving.py"]
