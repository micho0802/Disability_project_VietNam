# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set PYTHONPATH so tests and app can run
ENV PYTHONPATH=/app

# Default command: run scraper
CMD ["python", "app/main.py"]
