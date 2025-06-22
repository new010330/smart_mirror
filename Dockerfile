# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for TensorFlow and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all requirements files
COPY requirements.txt .
COPY skin_status/requirements.txt ./skin_status/
COPY personal_color_server/requirements.txt ./personal_color_server/

# Install main requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install skin_status requirements
RUN pip install --no-cache-dir -r skin_status/requirements.txt

# Install personal_color_server requirements (including TensorFlow)
RUN pip install --no-cache-dir -r personal_color_server/requirements.txt

# Copy the rest of the application code
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run Django server (assuming personal_color_server is Django-based)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
