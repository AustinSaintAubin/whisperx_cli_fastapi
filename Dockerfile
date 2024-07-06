FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.10
ENV PYTHONUNBUFFERED=1

# Install Python and other dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 11.7)
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install other items
RUN apt-get update \
 && apt-get install -y \
    git \
    ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 # Set working directory
WORKDIR /app

# Install WisperX
# RUN pip3 install git+https://github.com/m-bain/whisperx.git
COPY ./repositories/whisperx ./whisperx
RUN pip install --no-input --editable ./whisperx

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn

# Copy the web application code
COPY whisperx_cli_fastapi_api.py .

# WhisperX Runtime Environment Variables
ENV OUTPUT_DIR="/tmp/output"
ENV MODEL_DIR="/models"
ENV MODEL="base"
# ENV COMPUTE_TYPE
# ENV DEVICE
# ENV DEVICE_INDEX
# ENV PRINT_PROGRESS
# ENV BATCH_SIZE
# ENV THREADS
# ENV ALIGN_MODEL
# ENV INTERPOLATE_METHOD
# ENV RETURN_CHAR_ALIGNMENTS
# ENV VAD_ONSET
# ENV VAD_OFFSET
# ENV CHUNK_SIZE
# ENV TEMPERATURE
# ENV BEST_OF
# ENV BEAM_SIZE
# ENV PATIENCE
# ENV LENGTH_PENALTY
# ENV SUPPRESS_TOKENS
# ENV SUPPRESS_NUMERALS
# ENV CONDITION_ON_PREVIOUS_TEXT
# ENV FP16
# ENV TEMPERATURE_INCREMENT_ON_FALLBACK
# ENV COMPRESSION_RATIO_THRESHOLD
# ENV LOGPROB_THRESHOLD
# ENV NO_SPEECH_THRESHOLD
# ENV MAX_LINE_WIDTH
# ENV MAX_LINE_COUNT
# ENV HIGHLIGHT_WORDS
# ENV SEGMENT_RESOLUTION

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "whisperx_cli_fastapi_api:app", "--host", "0.0.0.0", "--port", "8000"]
