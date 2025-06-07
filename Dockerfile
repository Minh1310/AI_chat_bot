# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip list

# Copy the application code
COPY . .

# Expose the port (optional, for documentation)
EXPOSE 10000

# Run the application with Gunicorn, using $PORT environment variable
CMD ["sh", "-c", "echo 'Starting Gunicorn on PORT: ${PORT}' && gunicorn --bind 0.0.0.0:${PORT} --timeout 300 --workers 1 --log-level debug app:application || echo 'Gunicorn failed with exit code $?'"]