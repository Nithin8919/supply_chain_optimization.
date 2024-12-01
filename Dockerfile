FROM python:3.11-slim-buster

WORKDIR /app

# Copy your project files to the Docker container
COPY . /app

# Install AWS CLI
RUN apt update -y && apt install awscli -y

# Install Python dependencies
RUN apt-get update && pip install -r requirements.txt

# Set the command to run the training pipeline script
CMD ["python3", "pipelines/training_pipeline.py"]
