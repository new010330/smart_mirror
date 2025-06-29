# Use an official Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies for image processing and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY skin_status/requirements.txt ./skin_status/
COPY personal_color_server/requirements.txt ./personal_color_server/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r skin_status/requirements.txt
RUN pip install --no-cache-dir -r personal_color_server/requirements.txt

# Copy application source code
COPY . .

# Copy model files to appropriate locations
COPY saved_models/mobilenet_skin_best.pth ./skin_status/mobilenet_skin_best.pth
COPY saved_models/final_model_efficientnet.pt ./saved_models/final_model_efficientnet.pt
COPY saved_models/class_names.txt ./saved_models/class_names.txt

# Expose port 8000 for Django development server
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Start Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
