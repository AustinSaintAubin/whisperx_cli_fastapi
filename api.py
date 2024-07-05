from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import subprocess
import os
import logging
import zipfile
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def str_to_bool(value):
    return value.lower() in ['true', '1', 't', 'y', 'yes']

@app.post("/run-whisperx/")
async def run_whisperx(
    audio: UploadFile = File(...),
    model: str = "base",
    output_format: str = "txt",
    task: str = "transcribe",
    language: str = "en",
    batch_size: int = 6,
):
    audio_file_path = f"/tmp/{audio.filename}"
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await audio.read())

    output_dir = os.getenv("OUTPUT_DIR", "/tmp/output")
    hf_token = os.getenv("HF_TOKEN", "")
    device = os.getenv("DEVICE", "cuda")
    device_index = os.getenv("DEVICE_INDEX", "0")
    compute_type = os.getenv("COMPUTE_TYPE", "float32")
    threads = os.getenv("THREADS", "4")
    print_progress = str_to_bool(os.getenv("PRINT_PROGRESS", "false"))

    command = [
        "whisperx",
        audio_file_path,
        "--model", model,
        "--output_dir", output_dir,
        "--output_format", output_format,
        "--task", task,
        "--language", language,
        "--batch_size", str(batch_size),
        "--device", device,
        "--device_index", device_index,
        "--compute_type", compute_type,
        "--threads", threads,
    ]

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

