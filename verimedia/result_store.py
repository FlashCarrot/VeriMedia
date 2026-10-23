import json
import os
import time
import uuid
from verimedia.config import RESULTS_CACHE_DIR, TEMP_AUDIO_DIR

CACHE_TTL_SECONDS = 3600  # 1 hour


def save_analysis_result(result_data):
    """
    Persist analysis results on the server filesystem and return a lightweight UUID.
    Prevents the 4KB browser cookie limit overflow caused by storing base64 wordclouds
    and long transcriptions in Flask's client-side session.
    """
    result_id = uuid.uuid4().hex
    cache_path = os.path.join(RESULTS_CACHE_DIR, f"{result_id}.json")
    payload = {
        "created_at": time.time(),
        "result": result_data,
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError as e:
        print(f"Warning: Failed to cache analysis result {result_id}: {e}")
    return result_id


def load_analysis_result(result_id):
    """Load a previously cached analysis result by its UUID."""
    if not result_id or not isinstance(result_id, str):
        return None
    # Sanitize result_id to alphanumeric hex only
    safe_id = "".join(c for c in result_id if c.isalnum())
    if not safe_id:
        return None
    cache_path = os.path.join(RESULTS_CACHE_DIR, f"{safe_id}.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("result")
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to read cached result {safe_id}: {e}")
        return None


def cleanup_temp_files():
    """Remove temporary audio files and cached results older than CACHE_TTL_SECONDS."""
    threshold = time.time() - CACHE_TTL_SECONDS
    for directory in (TEMP_AUDIO_DIR, RESULTS_CACHE_DIR):
        if not os.path.exists(directory):
            continue
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    path = os.path.join(root, file)
                    try:
                        if os.path.getmtime(path) < threshold:
                            os.unlink(path)
                    except OSError:
                        pass
        except OSError:
            pass
