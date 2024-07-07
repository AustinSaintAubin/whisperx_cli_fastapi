 # WhisperX CLI FastAPI

This repository provides a FastAPI-based Command Line Interface (CLI) for WhisperX, an automatic speech recognition (ASR) system. WhisperX extends the functionality of OpenAI's Whisper ASR model by adding diarization and speaker identification capabilities to it. This makes it more suitable for transcribing conversations between multiple speakers.

## Features

- Transcribe audio files using WhisperX with a CLI interface.
- Supports single and multi-channel audio files.
- Diarize the speech in the audio file, allowing you to identify which speaker said what.
- Built on FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

## Installation using Docker or Docker Compose

To install and run WhisperX CLI FastAPI using Docker or Docker Compose, follow these steps:

1. Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/whisperx_cli_fastapi.git
```
2. Navigate into the project directory:
```bash
cd whisperx_cli_fastapi
```

### Docker

3. Build the Docker image:
```bash
docker build -t whisperx_cli_fastapi .
```
4. Run the Docker container:
```bash
docker run -p 8000:80 whisperx_cli_fastapi
```
The API will be accessible at `http://localhost:8000`.

### Docker Compose

3. Build and start the Docker container using Docker Compose:
```bash
docker-compose up --build
```
The API will be accessible at `http://localhost:8000`.

## Usage

To use the WhisperX CLI FastAPI, you can send a POST request to `http://localhost:8000/transcribe` with an audio file in the body. The API will return the transcription of the audio file, along with the diarization results.

Here's an example using curl:
```bash
curl -X POST "http://localhost:8000/transcribe" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@path/to/your/audio.wav"
```
Replace `path/to/your/audio.wav` with the path to your audio file.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.