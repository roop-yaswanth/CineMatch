FROM python:3.11-slim

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY --chown=user app.py .
COPY --chown=user .env* ./

USER user

# HF Spaces exposes port 7860
EXPOSE 7860

CMD ["python", "app.py"]
