import os
import subprocess
import tempfile
from verimedia.config import MAX_AUDIO_SIZE, TEMP_AUDIO_DIR


def extract_text_from_file(file_path):
    """
    Extract text content from .txt, .docx, or .pdf files.
    Returns (content, error_message).
    """
    if not os.path.exists(file_path):
        return None, "Uploaded file could not be found."

    file_extension = file_path.rsplit('.', 1)[1].lower() if '.' in file_path else ''

    if file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    elif file_extension == 'docx':
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text]
            # Also extract text from tables if present
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            content = "\n".join(paragraphs)
        except ImportError:
            return None, "Error: To analyze Word documents, please install the python-docx library."
        except Exception as e:
            return None, f"Error reading Word document: {str(e)}"
    elif file_extension == 'doc':
        return None, "Error: The legacy .doc format is not supported. Please convert to .docx or .txt."
    elif file_extension == 'pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                pages_text = []
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        pages_text.append(extracted)
                content = "\n".join(pages_text)
        except ImportError:
            return None, "Error: To analyze PDF documents, please install the PyPDF2 library."
        except Exception as e:
            return None, f"Error reading PDF document: {str(e)}"
    else:
        return None, f"Error: The file format .{file_extension} is not supported for text analysis."

    if not content or not content.strip():
        return None, "Error: No text content could be extracted from the file."

    return content, None


def get_ffmpeg_executable():
    """Return path to bundled imageio-ffmpeg binary or system ffmpeg."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def convert_audio_to_wav(file_path):
    """Convert audio (such as .m4a) to a temporary WAV/MP3 file for compatibility."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', dir=TEMP_AUDIO_DIR, delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        ffmpeg_exe = get_ffmpeg_executable()
        cmd = [
            ffmpeg_exe, '-y',
            '-i', file_path,
            '-vn',
            '-acodec', 'libmp3lame',
            '-ac', '1',
            '-ar', '16000',
            '-b:a', '64k',
            temp_path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            return temp_path

        # Fallback to MoviePy AudioFileClip
        from moviepy.editor import AudioFileClip
        audio_clip = AudioFileClip(file_path)
        audio_clip.write_audiofile(temp_path, verbose=False, logger=None)
        audio_clip.close()
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise RuntimeError(f"Audio conversion failed: {e}") from e


def convert_video_to_audio(video_path):
    """
    Extract and compress the audio track from a video file into an MP3 suitable
    for the OpenAI Whisper API (<25MB, mono, 16kHz).
    """
    temp_audio_file = tempfile.NamedTemporaryFile(suffix='.mp3', dir=TEMP_AUDIO_DIR, delete=False)
    temp_audio_path = temp_audio_file.name
    temp_audio_file.close()

    try:
        extracted = False

        # Method 1: Fast direct FFmpeg audio extraction (avoids decoding video frames in Python)
        try:
            ffmpeg_exe = get_ffmpeg_executable()
            cmd = [
                ffmpeg_exe, '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'libmp3lame',
                '-ac', '1',
                '-ar', '16000',
                '-b:a', '48k',
                '-filter:a', 'volume=1.5,loudnorm',
                '-q:a', '9',
                temp_audio_path
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode == 0 and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                extracted = True
        except Exception as ffmpeg_err:
            print(f"Direct FFmpeg extraction note: {ffmpeg_err}")

        # Method 2: Fallback to MoviePy
        if not extracted:
            from moviepy.editor import VideoFileClip
            video_clip = VideoFileClip(
                video_path,
                audio_buffersize=20000,
                verbose=False,
                target_resolution=(360, 640)
            )
            try:
                if not video_clip.audio:
                    raise RuntimeError("No audio track found in video")
                video_clip.audio.write_audiofile(
                    temp_audio_path,
                    verbose=False,
                    logger=None,
                    bitrate='48k',
                    ffmpeg_params=['-ac', '1', '-ar', '16000', '-q:a', '9']
                )
                extracted = True
            finally:
                video_clip.close()

        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            raise RuntimeError("Audio extraction failed - output file is empty")

        # Additional compression if extracted audio exceeds 25MB limit
        audio_size = os.path.getsize(temp_audio_path)
        if audio_size > MAX_AUDIO_SIZE:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(
                temp_audio_path,
                format="mp3",
                bitrate="32k",
                parameters=["-q:a", "9", "-ac", "1", "-ar", "16000"]
            )
            if os.path.getsize(temp_audio_path) > MAX_AUDIO_SIZE:
                raise RuntimeError(
                    f"Audio still too large after compression: {os.path.getsize(temp_audio_path) / (1024 * 1024):.1f}MB"
                )

        return temp_audio_path

    except Exception as e:
        if os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass
        raise RuntimeError(f"Could not extract audio: {e}") from e
