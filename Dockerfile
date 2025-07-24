FROM python:3.10-slim

WORKDIR /app

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirement.txt .
RUN pip install --upgrade pip && pip install -r requirement.txt

COPY app/ .

CMD ["python", "main.py"]