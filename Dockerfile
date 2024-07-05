FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python and other dependencies
RUN apt-get update \
 && apt-get install -y \
    libsndfile1  \
    python3-pip \
    python3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install git
RUN apt-get update \
 && apt-get install -y \
    git \
    ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install PyTorch (UDA 11.7)
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install the application
# RUN pip install --no-input --editable .
RUN pip3 install git+https://github.com/m-bain/whisperx.git

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn", "whisperx.api:app", "--host", "0.0.0.0", "--port", "8000"]
