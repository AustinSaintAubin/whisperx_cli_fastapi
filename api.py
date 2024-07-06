from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
import torch
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

app = FastAPI(
    title="WhisperX CLI ASR Webservice",
    description="A webservice for WhisperX CLI",
    version="0.0.1",
    contact={
        "name": "Austin Saint Aubin", 
        "url": "https://github.com/austinsaintaubin"
    },
    license_info={
        "name": "BSD 4-Clause License",
        "url": "https://opensource.org/licenses/BSD-4-Clause"
    }
)

def str_to_bool(value):
    return value.lower() in ['true', '1', 't', 'y', 'yes']

# Manually define available models, language choices, and other options
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
LANGUAGES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani",
    "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian", "eu": "Basque",
    "fa": "Persian", "fi": "Finnish", "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean",
    "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi",
    "ms": "Malay", "mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "pa": "Panjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen",
    "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese"
}
LANGUAGE_CODES = sorted(LANGUAGES.keys()) + sorted([k.title() for k in LANGUAGES.values()])
COMPUTE_TYPE_CHOICES = ["float16", "float32", "int8"]
OUTPUT_FORMAT_CHOICES = ["all", "srt", "vtt", "txt", "tsv", "json", "aud", "dir"]
TASK_CHOICES = ["transcribe", "translate"]

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

# @app.post("/asr/")
# async def run_whisperx(
#     audio_file: UploadFile = File(...),
#     task: Union[str, None] = Query(default="transcribe", enum=TASK_CHOICES),
#     language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
#     initial_prompt: Union[str, None] = Query(default=None),
#     output: Union[str, None] = Query(default="txt", enum=OUTPUT_FORMAT_CHOICES),
#     output_format = output
# ):

@app.post("/whisperx/")
async def run_whisperx(
    audio_file: UploadFile = File(...),
    model: Union[str, None] = Query(default="base", enum=MODEL_CHOICES),
    initial_prompt: Union[str, None] = Query(default=None),
    output_format: Union[str, None] = Query(default="txt", enum=OUTPUT_FORMAT_CHOICES),
    task: Union[str, None] = Query(default="transcribe", enum=TASK_CHOICES),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    no_align: bool = Query(default=False),
    diarize: bool = Query(default=False),
    min_speakers: Union[str, None] = Query(default=None),
    max_speakers: Union[str, None] = Query(default=None),
    # hf_token: Union[str, None] = Query(default=None)
    hf_token: Union[str, None] = Query(default="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")
):
    # Download the audio file to a temporary directory
    audio_file_path = f"/tmp/{audio_file.filename}"
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await audio_file.read())

    # Retrieve environment variables, default values are provided in comments
    output_dir = os.getenv("OUTPUT_DIR", "/tmp/output")  # Default: "/tmp/output"
    model_dir = os.getenv("MODEL_DIR", "/models")  # Default: "/models"
    
    compute_type = os.getenv("COMPUTE_TYPE", "float16" if torch.cuda.is_available() else "int8")  # Default: "float32"
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # Default: "cuda"

    device_index = os.getenv("DEVICE_INDEX")  # Default: "0"
    print_progress = str_to_bool(os.getenv("PRINT_PROGRESS", "false"))  # Default: False
    batch_size = os.getenv("BATCH_SIZE")  # Default: "6"
    threads = os.getenv("THREADS")  # Default: "4"
    align_model = os.getenv("ALIGN_MODEL")  # Default: ""
    interpolate_method = os.getenv("INTERPOLATE_METHOD")  # Default: "nearest"
    return_char_alignments = str_to_bool(os.getenv("RETURN_CHAR_ALIGNMENTS", "false"))  # Default: False
    vad_onset = os.getenv("VAD_ONSET")  # Default: "0.5"
    vad_offset = os.getenv("VAD_OFFSET")  # Default: "0.363"
    chunk_size = os.getenv("CHUNK_SIZE")  # Default: "30"
    temperature = os.getenv("TEMPERATURE")  # Default: "0"
    best_of = os.getenv("BEST_OF")  # Default: "5"
    beam_size = os.getenv("BEAM_SIZE")  # Default: "5"
    patience = os.getenv("PATIENCE")  # Default: "1.0"
    length_penalty = os.getenv("LENGTH_PENALTY")  # Default: "1.0"
    suppress_tokens = os.getenv("SUPPRESS_TOKENS")  # Default: "-1"
    suppress_numerals = str_to_bool(os.getenv("SUPPRESS_NUMERALS", "false"))  # Default: False
    condition_on_previous_text = str_to_bool(os.getenv("CONDITION_ON_PREVIOUS_TEXT", "false"))  # Default: False
    fp16 = str_to_bool(os.getenv("FP16", "false"))  # Default: Flase
    temperature_increment_on_fallback = os.getenv("TEMPERATURE_INCREMENT_ON_FALLBACK")  # Default: "0.2"
    compression_ratio_threshold = os.getenv("COMPRESSION_RATIO_THRESHOLD")  # Default: "2.4"
    logprob_threshold = os.getenv("LOGPROB_THRESHOLD")  # Default: "-1.0"
    no_speech_threshold = os.getenv("NO_SPEECH_THRESHOLD")  # Default: "0.6"
    max_line_width = os.getenv("MAX_LINE_WIDTH")  # Default: "None"
    max_line_count = os.getenv("MAX_LINE_COUNT")  # Default: "None"
    highlight_words = str_to_bool(os.getenv("HIGHLIGHT_WORDS", "false"))  # Default: False
    segment_resolution = os.getenv("SEGMENT_RESOLUTION")  # Default: "sentence"

    # command = ["whisperx", audio_file_path]
    command = ["whisperx", audio_file_path, "--output_dir", output_dir, "--model_dir", model_dir]

    # Add optional parameters to the command if they are provided
    if model:
        command.extend(["--model", model])
    if initial_prompt:
        command.extend(["--initial_prompt", initial_prompt])
    if output_format:
        command.extend(["--output_format", output_format])
    if task:
        command.extend(["--task", task])
    if language:
        command.extend(["--language", language])
    if no_align:
        command.append("--no_align")
    if diarize:
        command.append("--diarize")
    if min_speakers is not None:
        command.extend(["--min_speakers", str(min_speakers)])
    if max_speakers is not None:
        command.extend(["--max_speakers", str(max_speakers)])
    if hf_token:
        command.extend(["--hf_token", hf_token])
    if device:
        command.extend(["--device", device])
    if device_index:
        command.extend(["--device_index", device_index])
    if batch_size:
        command.extend(["--batch_size", batch_size])
    if compute_type:
        command.extend(["--compute_type", compute_type])
    if threads:
        command.extend(["--threads", threads])
    if align_model:
        command.extend(["--align_model", align_model])
    if interpolate_method:
        command.extend(["--interpolate_method", interpolate_method])
    if return_char_alignments:
        command.append("--return_char_alignments")
    if vad_onset:
        command.extend(["--vad_onset", vad_onset])
    if vad_offset:
        command.extend(["--vad_offset", vad_offset])
    if chunk_size:
        command.extend(["--chunk_size", chunk_size])
    if temperature:
        command.extend(["--temperature", temperature])
    if best_of:
        command.extend(["--best_of", best_of])
    if beam_size:
        command.extend(["--beam_size", beam_size])
    if patience:
        command.extend(["--patience", patience])
    if length_penalty:
        command.extend(["--length_penalty", length_penalty])
    if suppress_tokens:
        command.extend(["--suppress_tokens", suppress_tokens])
    if suppress_numerals:
        command.append("--suppress_numerals")
    if condition_on_previous_text:
        command.append("--condition_on_previous_text")
    if fp16:
        command.append("--fp16")
    if temperature_increment_on_fallback:
        command.extend(["--temperature_increment_on_fallback", temperature_increment_on_fallback])
    if compression_ratio_threshold:
        command.extend(["--compression_ratio_threshold", compression_ratio_threshold])
    if logprob_threshold:
        command.extend(["--logprob_threshold", logprob_threshold])
    if no_speech_threshold:
        command.extend(["--no_speech_threshold", no_speech_threshold])
    if max_line_width:
        command.extend(["--max_line_width", max_line_width])
    if max_line_count:
        command.extend(["--max_line_count", max_line_count])
    if highlight_words:
        command.append("--highlight_words")
    if segment_resolution:
        command.extend(["--segment_resolution", segment_resolution])
    if print_progress:
        command.extend(["--print_progress", "True"])

    logger.info(f"Running command: {' '.join(command)}")
    logger.info(f"NOTE: If command seems to hand, models are likely being downloaded. Please wait.")

    # Execute the WhisperX command and log the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Log the output and error to the terminal
    for stdout_line in iter(process.stdout.readline, ""):
        logger.info(stdout_line.strip())

    for stderr_line in iter(process.stderr.readline, ""):
        logger.error(stderr_line.strip())

    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        error_message = process.stderr.read().strip()
        logger.error(error_message)
        return JSONResponse(status_code=500, content={"detail": error_message})

    logger.info(f"Checking output directory: {output_dir}")
    output_files = os.listdir(output_dir)
    logger.info(f"Files in output directory: {output_files}")

    audio_filename_without_extension = os.path.splitext(audio_file.filename)[0]

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

