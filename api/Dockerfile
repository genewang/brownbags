# Use official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --upgrade pip && pip install poetry

# Explicitly install numpy first to avoid import errors
RUN pip install numpy

# Copy project files
COPY . .

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy project files
COPY . .

# Copy poetry files
# COPY pyproject.toml poetry.lock ./

# Expose port for Flask
EXPOSE 5000

# Set Flask environment variables
ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
CMD ["flask", "run"]
