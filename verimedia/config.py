import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMP_AUDIO_DIR = os.path.join(UPLOAD_FOLDER, 'temp_audio')
RESULTS_CACHE_DIR = os.path.join(UPLOAD_FOLDER, 'results_cache')
MODEL_ID_FILE = os.path.join(BASE_DIR, 'finetune_model_id.txt')
LEGACY_MODEL_ID_FILE = os.path.join(BASE_DIR, 'fine_tuned_model.txt')

# Size limits
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload size
MAX_VIDEO_SIZE = 500 * 1024 * 1024      # 500MB max video file size
MAX_AUDIO_SIZE = 25 * 1024 * 1024       # 25MB max audio size (OpenAI Whisper limit)
MAX_TEXT_CHARS = 15000                  # ~4000 tokens limit for text analysis

ALLOWED_EXTENSIONS = {
    'text': {'txt', 'pdf', 'doc', 'docx'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'oga', 'mpga'},
    'video': {'mp4', 'webm', 'mpeg', 'mov', 'avi', 'mkv'}
}

AUDIO_SUPPORTED_FORMATS = {'flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'}
VALID_TOXICITY_LEVELS = ('None', 'Mild', 'High', 'Max')

# Ensure required directories exist
for directory in (UPLOAD_FOLDER, TEMP_AUDIO_DIR, RESULTS_CACHE_DIR):
    os.makedirs(directory, exist_ok=True)


def get_fine_tuned_model_id():
    """Retrieve the configured fine-tuned model ID from env or model ID files."""
    env_model = os.environ.get('FINE_TUNED_MODEL', '').strip()
    if env_model:
        return env_model

    for path in (MODEL_ID_FILE, LEGACY_MODEL_ID_FILE):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    model_id = f.read().strip()
                    if model_id:
                        return model_id
            except OSError:
                continue
    return None


def get_env_api_key():
    """Return a valid OpenAI API key from environment variables, ignoring placeholders."""
    raw_key = os.environ.get('OPENAI_API_KEY', '').strip()
    placeholders = {
        '',
        'your_api_key_here',
        'your_openai_api_key_here',
        'sk-your-key-here',
    }
    if raw_key in placeholders:
        return None
    return raw_key


def allowed_file(filename, file_type=None):
    """Check if the filename has an allowed extension for the given file_type."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type:
        return ext in ALLOWED_EXTENSIONS.get(file_type, set())
    return any(ext in extensions for extensions in ALLOWED_EXTENSIONS.values())
