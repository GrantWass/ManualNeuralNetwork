# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with uvicorn (based on app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
