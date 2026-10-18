# Slim Python base image (matches the training environment so the
# pickled model loads with identical numpy/scikit-learn versions)
FROM python:3.13-slim

# Avoid interactive prompts; make Python output unbuffered for live logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install runtime dependencies first (leverages Docker layer caching)
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application code and the trained model
COPY src/ ./src/
COPY api/ ./api/
COPY models/heart_model.joblib ./models/heart_model.joblib

# Run as a non-root user for better security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Basic container healthcheck hitting the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else sys.exit(1)"

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
