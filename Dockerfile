# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a venv for the installs
RUN python3 -m venv venv

# Activate the venv
RUN source venv/bin/activate

RUN setup.sh

# Create a working directory
WORKDIR /workspace

# Copy the project files into the container
COPY . /workspace

# Set the entrypoint for the container
ENTRYPOINT ["/bin/bash"]