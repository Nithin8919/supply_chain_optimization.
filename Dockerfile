FROM apache/airflow:2.10.3

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /opt/airflow/data /opt/airflow/models

# Set permissions
RUN chown -R airflow:root /opt/airflow
ENV PYTHONPATH="$PYTHONPATH:$PWD"

USER airflow

# Install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt