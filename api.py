from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import subprocess
import os
import logging
import zipfile
from io import BytesIO
from typing import Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhisperX CLI ASR Webservice")

def str_to_bool(value):
    return value.lower() in ['true', '1', 't', 'y', 'yes']

# Manually define available models, language choices, and other options
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
LANGUAGES = {
    "en": "English",
    "fr": "French",
    # Add other language codes and names here
}
LANGUAGE_CODES = sorted(LANGUAGES.keys()) + sorted([k.title() for k in LANGUAGES.values()])
COMPUTE_TYPE_CHOICES = ["float16", "float32", "int8"]
OUTPUT_FORMAT_CHOICES = ["all", "srt", "vtt", "txt", "tsv", "json", "aud", "dir"]
TASK_CHOICES = ["transcribe", "translate"]

@app.post("/whisperx/")
async def run_whisperx(
    audio: UploadFile = File(...),
    model: Union[str, None] = Query(default="base", enum=MODEL_CHOICES),
    output_format: Union[str, None] = Query(default="txt", enum=OUTPUT_FORMAT_CHOICES),
    task: Union[str, None] = Query(default="transcribe", enum=TASK_CHOICES),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    batch_size: int = Query(default=6),
):
    audio_file_path = f"/tmp/{audio.filename}"
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await audio.read())

    # Retrieve environment variables, default to None if not set
    output_dir = os.getenv("OUTPUT_DIR", "/tmp/output")
    hf_token = os.getenv("HF_TOKEN", "")
    device = os.getenv("DEVICE", "")
    device_index = os.getenv("DEVICE_INDEX", "")
    compute_type = os.getenv("COMPUTE_TYPE", "")
    threads = os.getenv("THREADS", "")
    print_progress = str_to_bool(os.getenv("PRINT_PROGRESS", "false"))

    command = ["whisperx", audio_file_path]

    # Add optional parameters to the command if they are provided
    if model:
        command.extend(["--model", model])
    if output_dir:
        command.extend(["--output_dir", output_dir])
    if output_format:
        command.extend(["--output_format", output_format])
    if task:
        command.extend(["--task", task])
    if language:
        command.extend(["--language", language])
    if batch_size:
        command.extend(["--batch_size", str(batch_size)])
    if device:
        command.extend(["--device", device])
    if device_index:
        command.extend(["--device_index", device_index])
    if compute_type:
        command.extend(["--compute_type", compute_type])
    if threads:
        command.extend(["--threads", threads])
    if hf_token:
        command.extend(["--hf_token", hf_token])
    if print_progress:
        command.append("--print_progress")

    logger.info(f"Running command: {' '.join(command)}")

    # Execute the WhisperX command and log the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Log the output to the terminal
    for stdout_line in iter(process.stdout.readline, ""):
        logger.info(stdout_line.strip())

    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        error_message = process.stderr.read().strip()
        logger.error(error_message)
        return JSONResponse(status_code=500, content={"detail": error_message})

    logger.info(f"Checking output directory: {output_dir}")
    output_files = os.listdir(output_dir)
    logger.info(f"Files in output directory: {output_files}")

    audio_filename_without_extension = os.path.splitext(audio.filename)[0]

    if output_format == "all":
        output_files = [os.path.join(output_dir, f) for f in output_files if f.startswith(audio_filename_without_extension)]
        
        # Create a ZIP file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in output_files:
                file_name = os.path.basename(file_path)
                zip_file.write(file_path, arcname=file_name)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={audio_filename_without_extension}_output.zip"})
    
    possible_file = os.path.join(output_dir, f"{audio_filename_without_extension}.{output_format}")
    if os.path.exists(possible_file):
        return FileResponse(possible_file)
    else:
        logger.error(f"Output file not found: {possible_file}")
        return JSONResponse(status_code=404, content={"detail": "Output file not found."})

