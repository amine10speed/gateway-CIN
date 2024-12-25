# Use the official Python slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Environment variables to avoid issues
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python dependencies
RUN apt-get update -o Acquire::Retries=5 && apt-get install -y --no-install-recommends \
       libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (use Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
