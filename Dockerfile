#dockerfile
# Use an official lightweight Python base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install any OS-level dependencies if needed (optional)
# For example, if we needed libffi or other libraries:
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

#Upgrade pip to latest version for improved dependency resolution
RUN pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt

# Copy the application code into the container
COPY . .

# Set environment variables (if any configuration is needed)
ENV PORT=8080

#Create a non-root user for security and switch to it
RUN adduser --disabled-password agentuser \
    && chown -R agentuser /app
USER agentuser

# Expose the port (for local testing; Cloud Run provides the port via $PORT)
EXPOSE 8080

#Optional: add a healthcheck to ensure the application is responsive
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8080/ || exit 1

# Command to start the application (using Uvicorn to run FastAPI server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
