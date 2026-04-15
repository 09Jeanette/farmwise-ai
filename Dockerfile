FROM python:3.11-slim

WORKDIR /app

# Set environment variables for memory optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
