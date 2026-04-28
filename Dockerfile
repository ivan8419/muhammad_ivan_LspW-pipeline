FROM tensorflow/tfx:1.14.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
